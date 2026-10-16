"""WS4 — planning: turn a question into a typed plan with tool steps.

The planner classifies the question into one of the assignment's six types and
emits an ordered list of tool calls with the arguments filled in.  It is
deterministic by default (regex + lexicon) so the agent, the tests and the
evaluation behave identically on every machine; when a real backend is
configured the *text* of the plan can be produced by the model instead, but the
tool selection stays rule-driven — this is what makes the ablation between
config E (tools without a loop) and config F (the loop) meaningful.

Question types
--------------
``single_hop``      one lookup answers it
``table``           the answer lives in a table (bit field, parameter, procedure)
``figure``          the answer lives in a diagram (state machine, timing, block)
``cross_document``  the answer needs two or more documents chained together
``comparison``      two things are contrasted (SPR vs EPR, Type-C vs PD)
``unanswerable``    the premise is not in the corpus — answered by the loop's
                    evidence check, not by the planner
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

QUESTION_TYPES = ("single_hop", "table", "figure", "cross_document", "comparison", "unanswerable")

FIG_RE = re.compile(r"\b(figure|diagram|state machine|state diagram|transition|waveform|timing diagram|"
                    r"sequence|flow ?chart|block diagram)\b", re.I)
TABLE_RE = re.compile(r"\b(table|bit field|bit-field|bit position|bits?|row|column|steps?|procedure|"
                      r"min(?:imum)?|max(?:imum)?|limit|value|unit)\b", re.I)
COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|differ(?:s|ence)?|instead of|"
                        r"what changed|both)\b", re.I)
CROSS_RE = re.compile(r"\b(which test|which check|which spec|which document|trace|chained|"
                      r"across|references?|points? to|defined (?:by|in)|verif\w+|"
                      r"requirement|stale|superseded|older revision)\b", re.I)
TEST_ID_RE = re.compile(r"\b(?:TEST\.[A-Z0-9.#]+|TD\s?\d+(?:\.\d+)+|COMMON\.CHECK\.[A-Z0-9.]+|"
                        r"\[[A-Z0-9._#]+\])\b")
IDENT_RE = re.compile(r"\b(?:t[A-Z][A-Za-z]+|n[A-Z][A-Za-z]+|vRd-[0-9.]+|[A-Z]{2,}_[A-Za-z0-9_]+|"
                      r"PE_[A-Za-z_]+|PE\.[A-Za-z.]+|[A-Z]{2,}\.[A-Z]{2,}\.[A-Z0-9.]+|"
                      r"[A-Za-z]+_[A-Za-z]+|[A-Z]{2,}[0-9]{2,})\b")
DOC_ALIASES: dict[str, tuple[str, ...]] = {
    "pd": ("pd", "power delivery", "usb pd"),
    "pd_cts": ("pd cts", "compliance test specification", "compliance test"),
    "typec": ("type-c", "typec", "type c"),
    "typec_fts": ("functional test", "fts"),
    "scc_cts": ("shared capacity", "scc"),
    "usb4": ("usb4", "usb 4"),
    "tech2": ("tech 2", "tech2"),
}
STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "what", "which",
             "how", "does", "do", "why", "when", "where", "specification", "spec", "table", "figure"}


@dataclass
class PlanStep:
    """One planned tool call."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "purpose": self.purpose}

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.tool}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


@dataclass
class Plan:
    """A question, its type and the steps the agent should run."""

    question: str
    question_type: str = "single_hop"
    steps: list[PlanStep] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    expected_docs: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        """Human-readable plan lines (kept in the trace)."""
        lines = [f"type={self.question_type}",
                 f"sub-questions={len(self.sub_questions)}",
                 f"expected_documents={','.join(self.expected_docs) or 'any'}"]
        lines += [f"step {i + 1}: {step.tool} ({step.purpose})" for i, step in enumerate(self.steps)]
        lines += [f"note: {n}" for n in self.notes]
        return lines


class Planner:
    """Rule-based classifier and step builder."""

    def __init__(self, *, llm: Any | None = None, max_steps: int = 8) -> None:
        self.llm = llm
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    # classification
    # ------------------------------------------------------------------ #
    def classify(self, question: str) -> str:
        text = question.strip()
        if not text:
            return "single_hop"
        docs = self.documents_in(text)
        if COMPARE_RE.search(text) and len(self.comparison_sides(text)) >= 2:
            return "comparison"
        if "compar" in text.lower() and docs:
            return "comparison"
        if CROSS_RE.search(text) and (len(docs) >= 2 or TEST_ID_RE.search(text)
                                     or re.search(r"\b(which|what)\b.*\b(checks?|verif\w+|references?)\b",
                                                  text, re.I)):
            return "cross_document"
        if FIG_RE.search(text):
            return "figure"
        if TABLE_RE.search(text) or re.search(r"\b\d+\.\d+\b", text) and "table" in text.lower():
            return "table"
        if len(docs) >= 2:
            return "cross_document"
        return "single_hop"

    def documents_in(self, question: str) -> list[str]:
        """Which documents the question names (aliases, not guesses)."""
        lowered = question.lower()
        found: list[str] = []
        for doc_id, aliases in DOC_ALIASES.items():
            if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered) for alias in aliases):
                found.append(doc_id)
        if "typec" in found and "typec_fts" in found and "type-c functional test" not in lowered:
            pass
        return found

    def identifiers(self, question: str) -> list[str]:
        """Identifiers (timers, messages, tests, states) mentioned in the question."""
        out = [m.group(0) for m in TEST_ID_RE.finditer(question)]
        out += [m.group(0) for m in IDENT_RE.finditer(question)]
        seen: list[str] = []
        for token in out:
            if token not in seen and token.lower() not in STOPWORDS:
                seen.append(token)
        return seen

    def comparison_sides(self, question: str) -> list[str]:
        """Split a comparison question into the things being compared."""
        parts = re.split(r"\b(?:compare|versus|vs\.?|and|with|to)\b", question, flags=re.I)
        sides = [p.strip(" ,?.()") for p in parts if len(p.strip(" ,?.()")) > 3]
        return [s for s in sides if not re.fullmatch(r"(what|which|how|the|a|an)", s, re.I)][:3]

    def keywords(self, question: str) -> list[str]:
        from specintel.index.bm25 import tokenize

        return [t for t in tokenize(question) if t not in STOPWORDS and len(t) > 2]

    def decompose(self, question: str) -> list[str]:
        """Split compound questions at conjunctions that introduce new asks."""
        parts = re.split(r"(?:;|\?|,\s*(?=and|which|what|where|how|who)\b|\band\b\s+(?=which|what|where|how|who|"
                         r"in which|does|is)\b)", question, flags=re.I)
        out = [p.strip(" ,.") for p in parts if len(p.strip(" ,.")) > 12]
        return out or [question.strip()]

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def plan(self, question: str, *, docs: Sequence[str] | None = None, k: int = 8) -> Plan:
        question_type = self.classify(question)
        plan = Plan(question=question, question_type=question_type,
                    sub_questions=self.decompose(question),
                    identifiers=self.identifiers(question),
                    expected_docs=list(docs) if docs else self.documents_in(question))
        keywords = " ".join(self.keywords(question)[:8])
        doc_args = plan.expected_docs or None

        if question_type == "table":
            plan.steps += [
                PlanStep("search_text", {"query": question, "k": k, "tables_only": True},
                         "find the table that holds the value"),
                PlanStep("get_table", {"table_id": "$best_table"},
                         "read the whole table, including any continuation page"),
                PlanStep("search_text", {"query": keywords, "k": k}, "surrounding prose for context"),
            ]
        elif question_type == "figure":
            plan.steps += [
                PlanStep("search_text", {"query": question, "k": k,
                                         "element_types": ["figure_description", "figure_caption", "paragraph"]},
                         "find the diagram description"),
                PlanStep("search_pages", {"query": keywords, "k": 5}, "look at the page image itself"),
                PlanStep("get_figure", {"figure_id": "$best_figure"}, "read the structured diagram"),
                PlanStep("search_text", {"query": keywords, "k": k}, "surrounding prose for context"),
            ]
        elif question_type == "cross_document":
            plan.steps.append(PlanStep("search_text", {"query": question, "k": k},
                                       "first hop: the test or requirement"))
            for doc_id in (plan.expected_docs or [])[:2]:
                plan.steps.append(PlanStep("search_text", {"query": keywords, "k": k, "docs": [doc_id]},
                                           f"second hop inside {doc_id}"))
            if plan.identifiers:
                plan.steps.append(PlanStep("resolve_entity", {"name": plan.identifiers[0]},
                                           "canonical node for the identifier"))
                plan.steps.append(PlanStep("graph_query", {"kind": "neighbors", "name": plan.identifiers[0],
                                                           "depth": 2},
                                           "follow the requirement/test/timer links"))
            plan.steps.append(PlanStep("search_text", {"query": " ".join(plan.identifiers[:3]) or keywords, "k": k},
                                       "identifier-focused retrieval"))
        elif question_type == "comparison":
            sides = self.comparison_sides(question) or [question]
            for side in sides:
                plan.steps.append(PlanStep("search_text", {"query": side, "k": k},
                                           f"evidence for '{side[:40]}'"))
            if plan.identifiers:
                plan.steps.append(PlanStep("graph_query", {"kind": "lookup", "name": plan.identifiers[0]},
                                           "canonical entities for the compared items"))
            plan.steps.append(PlanStep("search_text", {"query": question, "k": k}, "overall context"))
        else:
            plan.steps.append(PlanStep("search_text", {"query": question, "k": k, "docs": doc_args},
                                       "primary lookup"))
            if plan.identifiers:
                plan.steps.append(PlanStep("graph_query", {"kind": "lookup", "name": plan.identifiers[0]},
                                           "canonical entity for the identifier"))
            plan.steps.append(PlanStep("search_text", {"query": keywords, "k": k},
                                       "keyword-focused second pass"))

        # never plan more steps than the loop will run
        if len(plan.steps) > self.max_steps:
            plan.notes.append(f"plan truncated from {len(plan.steps)} to {self.max_steps} steps")
            plan.steps = plan.steps[: self.max_steps]
        if plan.identifiers:
            plan.notes.append(f"identifiers: {', '.join(plan.identifiers[:6])}")
        return plan


def classify(question: str) -> str:
    """Module-level convenience used by the evaluation and the tests."""
    return Planner().classify(question)
