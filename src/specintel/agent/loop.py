"""WS4 — the agentic loop: plan → act → observe → reflect → answer or decline.

Loop shape (Task 4, Step 3 of the assignment):

1. **Plan** (:mod:`specintel.agent.planner`): classify the question, decompose it
   and choose tool calls.
2. **Act**: run the tool calls, collecting hits with provenance.  ``$best_table``
   and ``$best_figure`` placeholders are resolved from the previous step's hits,
   which is how "find the table, then read the table" works.
3. **Observe**: every tool result is recorded in the trace with its latency,
   hit count and top score.
4. **Reflect**: check whether the evidence covers the question (identifiers
   present? sub-questions covered?).  If not, run one corrective round with a
   different query (no units, entity names from ``resolve_entity``, neighbours
   from the graph).  A question whose identifiers and numbers simply do not occur
   in the corpus is declined instead of answered.
5. **Answer**: generate from the evidence, verify the citations, drop unsupported
   claims, and return the answer with its citations and trace.
6. **Decline**: when there is no adequate evidence, say so and record what was
   searched — the assignment's honest-decline requirement.

Everything the loop does is bounded: ``max_steps`` tool calls, ``budget_s`` wall
clock, deterministic settings, and the token budget is recorded from the backend
response.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from specintel.agent.prompts import SYSTEM_PROMPT  # noqa: E402 (single source of truth)
from specintel.agent.planner import Plan, PlanStep, Planner
from specintel.agent.tools import ToolResult, citation_from_hit
from specintel.agent.verifier import CitationVerifier, VerificationResult
from specintel.common.llm import ExtractiveLLM, generate, get_llm
from specintel.common.logging import get_logger, log_event, log_exception
from specintel.common.models import AgentTrace, Citation, SearchHit, TraceStep, new_id
from specintel.common.resources import hardware_profile

INSUFFICIENT = "INSUFFICIENT"

# the system prompt lives in agent/prompts.py so it can be reviewed on its own
SYSTEM_PROMPT = SYSTEM_PROMPT


@dataclass
class AgentConfig:
    """Budgets and switches for one agent configuration."""

    max_steps: int = 8
    budget_s: float = 30.0
    top_k: int = 8
    min_evidence_score: float = 0.25
    reflect_rounds: int = 1
    verify_citations: bool = True
    drop_unsupported: bool = True
    allow_decline: bool = True
    config_name: str = "F"
    deterministic: bool = True


@dataclass
class AgentAnswer:
    """The agent's answer with everything needed to audit it."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    declined: bool = False
    decline_reason: str = ""
    confidence: float = 0.0
    evidence: list[SearchHit] = field(default_factory=list)
    trace: AgentTrace | None = None
    verification: VerificationResult | None = None
    plan: Plan | None = None
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "declined": self.declined,
                "decline_reason": self.decline_reason, "confidence": round(self.confidence, 4),
                "citations": [c.to_marker() for c in self.citations],
                "n_evidence": len(self.evidence), "tool_calls": self.tool_calls,
                "verification": self.verification.to_dict() if self.verification else None,
                "plan": [str(s) for s in (self.plan.steps if self.plan else [])],
                "trace": self.trace.to_dict() if self.trace else None}


class SpecAgent:
    """The agentic RAG loop over the tools and the knowledge graph."""

    def __init__(self, toolkit: Any, llm: Any | None = None, *, config: Any | None = None,
                 agent_config: AgentConfig | None = None,
                 verifier: CitationVerifier | None = None) -> None:
        self.tools = toolkit
        self.config = config
        self.agent_config = agent_config or AgentConfig()
        # An agent built from a toolkit alone still has to answer: falling back to
        # the deterministic extractive composer (rather than None) is what makes
        # `SpecAgent(toolkit)` usable in a test, a notebook or an MCP host, and it
        # is the same backend `get_llm("auto")` picks when no local model is up.
        if llm is not None:
            self.llm = llm
        elif config is not None:
            self.llm = get_llm(config)
        else:
            self.llm = ExtractiveLLM()
        # the agent answers compound, table-heavy questions: six units (rows or
        # sentences) per answer, at most three from the table that holds them
        if hasattr(self.llm, "max_sentences"):
            self.llm.max_sentences = max(getattr(self.llm, "max_sentences", 4), 6)
        if hasattr(self.llm, "per_item_limit"):
            self.llm.per_item_limit = max(getattr(self.llm, "per_item_limit", 2), 3)
        self.planner = Planner(max_steps=self.agent_config.max_steps)
        self.verifier = verifier or CitationVerifier(drop_unsupported=self.agent_config.drop_unsupported)
        self.log = get_logger("ws4.agent", config)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def answer(self, question: str, *, docs: Sequence[str] | None = None,
               question_type: str | None = None, k: int | None = None) -> AgentAnswer:
        """Answer one question, or decline with a reason."""
        started = time.perf_counter()
        run_id = new_id("run")
        k = k or self.agent_config.top_k
        plan = self.planner.plan(question, docs=docs, k=k)
        if question_type:
            plan.question_type = question_type
        trace = AgentTrace(run_id=run_id, question=question, question_category=plan.question_type,
                           plan=plan.describe(), config_name=self.agent_config.config_name,
                           model=getattr(self.llm, "name", "none"),
                           hardware=hardware_profile())
        evidence: list[SearchHit] = []
        seen_evidence: set[str] = set()
        tool_calls = 0
        last_table: dict[str, Any] | None = None
        last_figure: dict[str, Any] | None = None
        #: tables and figures the knowledge graph has named but whose content has
        #: not been read yet: the graph knows that Table 6.8 *defines*
        #: tSenderResponse, and the answer needs that table's rows, not just the
        #: relation.  Filled from ``graph_query`` results (see D-19 in TEAM.md).
        graph_refs: dict[str, list[tuple[str, str]]] = {"table": [], "figure": []}

        # -- act ---------------------------------------------------------- #
        # ``docs`` is a promise to the reader ("answer from these documents"):
        # it restricts the final evidence list, including relations the graph
        # contributes, so a filtered answer cannot cite a document the caller
        # excluded.
        for step in plan.steps:
            if tool_calls >= self.agent_config.max_steps:
                trace.warnings.append(f"step budget reached ({self.agent_config.max_steps})")
                break
            if time.perf_counter() - started > self.agent_config.budget_s:
                trace.warnings.append(f"time budget reached ({self.agent_config.budget_s}s)")
                break
            args = self._resolve_args(step, plan, last_table, last_figure)
            if args is None:
                continue
            result = self.tools.call(step.tool, **args)
            tool_calls += 1
            if result.error:
                trace.steps.append(TraceStep(step=tool_calls, kind="tool", name=step.tool, args=args,
                                             observation=result.error, error=result.error,
                                             duration_s=round(result.latency_s, 4)))
                continue
            if result.table:
                last_table = result.table
            if result.figure:
                last_figure = result.figure
            if result.graph is not None:
                _collect_graph_refs(result.graph, graph_refs)
            new_hits = [h for h in result.hits if h.chunk_id not in seen_evidence]
            for hit in new_hits:
                seen_evidence.add(hit.chunk_id)
            evidence.extend(new_hits)
            trace.steps.append(TraceStep(step=tool_calls, kind="tool", name=step.tool, args=args,
                                         n_hits=len(result.hits), top_score=result.top_score() or None,
                                         observation=(result.text or "")[:400],
                                         duration_s=round(result.latency_s, 4),
                                         decision=step.purpose))
            log_event(self.log, "agent.step", run_id=run_id, step=tool_calls, tool=step.tool,
                      hits=len(result.hits), evidence=len(evidence))

        # ``docs`` is a promise to the caller ("answer from these documents"):
        # the graph step can contribute relations from other documents, so the
        # promise is enforced on the evidence list itself — everything after this
        # point (reflection, decline decision, answer and citations) sees only
        # the documents that were asked for.
        if docs:
            allowed = {d.strip() for d in docs if d.strip()}
            dropped = [h for h in evidence if h.doc_id not in allowed]
            evidence = [h for h in evidence if h.doc_id in allowed]
            if dropped:
                trace.warnings.append(f"docs filter removed {len(dropped)} evidence item(s)")
                log_event(self.log, "agent.docs_filter", run_id=run_id, allowed=sorted(allowed),
                          kept=len(evidence), dropped=len(dropped))

        # -- reflect ------------------------------------------------------ #
        gaps = self._gaps(plan, evidence, last_table, last_figure)
        rounds = 0
        while gaps and rounds < self.agent_config.reflect_rounds and tool_calls < self.agent_config.max_steps:
            rounds += 1
            trace.steps.append(TraceStep(step=tool_calls, kind="reflect", name="reflect",
                                         observation="; ".join(gaps),
                                         decision="retrieve again with a different formulation"))
            for call in self._reflection_calls(plan, gaps, k, graph_refs):
                if tool_calls >= self.agent_config.max_steps:
                    break
                result = self.tools.call(call.tool, **call.args)
                tool_calls += 1
                if result.table:
                    last_table = result.table
                if result.figure:
                    last_figure = result.figure
                if result.graph is not None:
                    _collect_graph_refs(result.graph, graph_refs)
                new_hits = [h for h in result.hits if h.chunk_id not in seen_evidence]
                for hit in new_hits:
                    seen_evidence.add(hit.chunk_id)
                evidence.extend(new_hits)
                trace.steps.append(TraceStep(step=tool_calls, kind="tool", name=call.tool, args=call.args,
                                             n_hits=len(result.hits), top_score=result.top_score() or None,
                                             observation=(result.text or "")[:300],
                                             duration_s=round(result.latency_s, 4),
                                             decision=f"reflection round {rounds}: {call.purpose}",
                                             error=result.error or None))
            # the graph is the step that *names* the table or figure holding the
            # answer, and it ran inside this round — so the read is issued now,
            # in the same round, while budget remains
            for call in self._graph_follow_ups(plan, last_table, last_figure, graph_refs):
                if tool_calls >= self.agent_config.max_steps:
                    break
                result = self.tools.call(call.tool, **call.args)
                tool_calls += 1
                if result.table:
                    last_table = result.table
                if result.figure:
                    last_figure = result.figure
                new_hits = [h for h in result.hits if h.chunk_id not in seen_evidence]
                for hit in new_hits:
                    seen_evidence.add(hit.chunk_id)
                evidence.extend(new_hits)
                trace.steps.append(TraceStep(step=tool_calls, kind="tool", name=call.tool, args=call.args,
                                             n_hits=len(result.hits), top_score=result.top_score() or None,
                                             observation=(result.text or "")[:300],
                                             duration_s=round(result.latency_s, 4),
                                             decision=f"reflection round {rounds}: {call.purpose}",
                                             error=result.error or None))
            gaps = self._gaps(plan, evidence, last_table, last_figure)

        # -- decide: answer or decline ------------------------------------ #
        verdict = self._decline_decision(plan, evidence, gaps)
        if verdict and self.agent_config.allow_decline:
            return self._decline(question, trace, evidence, verdict, started, tool_calls, plan)

        if last_table and last_table.get("markdown"):
            evidence = self._with_table_evidence(evidence, last_table)
        if last_figure and last_figure.get("description"):
            evidence = self._with_figure_evidence(evidence, last_figure)

        answer_text, citations, verification = self._compose(question, plan, evidence, trace,
                                                             docs=docs)
        tool_calls += 0
        if answer_text.strip().upper().startswith(INSUFFICIENT) and self.agent_config.allow_decline:
            return self._decline(question, trace, evidence, "the model reported the evidence as insufficient",
                                 started, tool_calls, plan)
        if not answer_text.strip() and self.agent_config.allow_decline:
            return self._decline(question, trace, evidence, "no sentence in the evidence answered the question",
                                 started, tool_calls, plan)

        confidence = min(1.0, max((h.score for h in evidence), default=0.0) / 4.0)
        trace.answer = answer_text
        trace.citations = citations
        trace.confidence = confidence
        trace.latency_s = time.perf_counter() - started
        trace.token_usage = {"total": sum(s.tokens for s in trace.steps)}
        log_event(self.log, "agent.answer", run_id=run_id, question=question[:120], citations=len(citations),
                  evidence=len(evidence), tool_calls=tool_calls, declined=False,
                  duration_s=round(trace.latency_s, 4))
        return AgentAnswer(question=question, answer=answer_text, citations=citations, declined=False,
                           confidence=confidence, evidence=evidence, trace=trace, verification=verification,
                           plan=plan, tool_calls=tool_calls)

    # ------------------------------------------------------------------ #
    # planning helpers
    # ------------------------------------------------------------------ #
    def _resolve_args(self, step: PlanStep, plan: Plan, table: dict[str, Any] | None,
                      figure: dict[str, Any] | None) -> dict[str, Any] | None:
        """Fill placeholder arguments from what previous steps found."""
        args = dict(step.args)
        for key, value in list(args.items()):
            if value == "$best_table":
                if not table:
                    return None
                args.pop(key)
                args["table_id"] = table.get("table_id") or table.get("number")
            elif value == "$best_figure":
                if not figure:
                    return None
                args.pop(key)
                args["figure_id"] = figure.get("figure_id") or figure.get("number")
        return args

    def _reflection_calls(self, plan: Plan, gaps: list[str], k: int,
                          graph_refs: dict[str, list[tuple[str, str]]] | None = None) -> list[PlanStep]:
        """Corrective retrieval: read what the graph named, reformulate, then follow it."""
        calls: list[PlanStep] = []
        refs = graph_refs or {}
        keywords = " ".join(self.planner.keywords(plan.question)[:6])
        calls.append(PlanStep("search_text", {"query": keywords, "k": k},
                              "keyword reformulation (identifiers and units stripped)"))
        for identifier in plan.identifiers[:2]:
            calls.append(PlanStep("resolve_entity", {"name": identifier}, "canonical name for the identifier"))
            calls.append(PlanStep("graph_query", {"kind": "neighbors", "name": identifier, "depth": 2},
                                  "follow the identifier through the knowledge graph"))
        if plan.expected_docs:
            calls.append(PlanStep("search_text", {"query": plan.question, "k": k,
                                                  "docs": plan.expected_docs[:1]},
                                  "restrict to the document the question names"))
        # the graph is queried above, and it is the graph that names the table or
        # figure that holds the answer; read it in the same round (its rows are
        # evidence, a relation between node names is not)
        if any("no table was read" in gap for gap in gaps) and refs.get("table"):
            doc_id, number = refs["table"][0]
            calls.append(PlanStep("get_table", {"doc_id": doc_id, "number": number},
                                  "read the table the knowledge graph points at"))
        if any("no figure was read" in gap for gap in gaps) and refs.get("figure"):
            doc_id, number = refs["figure"][0]
            calls.append(PlanStep("get_figure", {"doc_id": doc_id, "number": number},
                                  "read the figure the knowledge graph points at"))
        return calls[: max(1, self.agent_config.max_steps // 2)]

    def _graph_follow_ups(self, plan: Plan, table: dict[str, Any] | None,
                          figure: dict[str, Any] | None,
                          graph_refs: dict[str, list[tuple[str, str]]]) -> list[PlanStep]:
        """Read the table or figure the knowledge graph just named.

        ``graph_query`` can answer that Table 6.8 *defines* tSenderResponse; it
        cannot answer what the limits are, because the limits are rows.  This
        closes that loop while the round is still running.
        """
        calls: list[PlanStep] = []
        if table is None and plan.question_type in ("table", "single_hop", "comparison") \
                and graph_refs.get("table"):
            doc_id, number = graph_refs["table"][0]
            calls.append(PlanStep("get_table", {"doc_id": doc_id, "number": number},
                                  "read the table the knowledge graph points at"))
        if figure is None and plan.question_type in ("figure", "state_machine", "timing") \
                and graph_refs.get("figure"):
            doc_id, number = graph_refs["figure"][0]
            calls.append(PlanStep("get_figure", {"doc_id": doc_id, "number": number},
                                  "read the figure the knowledge graph points at"))
        return calls

    def _gaps(self, plan: Plan, evidence: Sequence[SearchHit], table: dict[str, Any] | None,
              figure: dict[str, Any] | None) -> list[str]:
        """What is still missing from the evidence?"""
        gaps: list[str] = []
        if not evidence:
            return ["no evidence retrieved"]
        blob = " ".join(h.text for h in evidence).lower()
        missing_ids = [i for i in plan.identifiers if i.lower() not in blob]
        if len(missing_ids) == len(plan.identifiers) and plan.identifiers:
            gaps.append(f"no evidence mentions {', '.join(plan.identifiers[:3])}")
        if plan.question_type == "table" and table is None:
            gaps.append("the question needs a table but no table was read")
        if plan.question_type == "figure" and figure is None:
            gaps.append("the question needs a figure but no figure was read")
        if plan.question_type == "cross_document" and len({h.doc_id for h in evidence}) < 2:
            gaps.append("a cross-document answer needs evidence from more than one document")
        missing_terms = self._uncovered_terms(plan, evidence)
        if missing_terms:
            gaps.append("no evidence covers " + ", ".join(missing_terms[:4]))
        return gaps

    #: words that ask *how* to answer rather than *what* about the corpus; their
    #: absence from the evidence says nothing about whether the corpus can answer
    INSTRUCTION_WORDS = {
        "list", "lists", "give", "state", "explain", "describe", "define", "defined", "according",
        "step", "steps", "tags", "tag", "results", "result", "versions", "version", "related", "compare",
        "comparison", "expected", "value", "values", "table", "figure", "section", "chapter", "specification",
        "spec", "document", "documents", "corpus", "mention", "mentions", "says", "say", "find", "find",
        "identify", "summarise", "summarize", "note", "notes", "used", "uses", "use", "used", "required",
        "requires", "require", "support", "supports", "supported", "allow", "allows", "allowed",
    }

    def _uncovered_terms(self, plan: Plan, evidence: Sequence[SearchHit]) -> list[str]:
        """Distinctive question terms (long or hyphenated, non-stopword) that no hit mentions.

        "Which standards body's Wi-Fi amendment requires multi-link operation" has
        three distinctive terms (ieee, 802.11be, multi-link) and none of them occur
        anywhere in this corpus, which is what makes the question unanswerable from
        the ingested documents rather than merely hard to retrieve.
        """
        terms = [t for t in self.planner.keywords(plan.question)
                 if len(t) > 3 and not t.isdigit() and t not in self.INSTRUCTION_WORDS]
        terms += [m.group(0).lower() for m in _HYPHEN_TERM.finditer(plan.question)]
        terms += [m.group(0).lower() for m in _NUMBERED_RE.finditer(plan.question)]      # 802.11be, R3.2
        terms += [i.lower() for i in plan.identifiers]
        terms = [t for t in dict.fromkeys(terms) if t not in self.INSTRUCTION_WORDS]
        if len(terms) < 2:
            return []
        blob = " ".join(h.text for h in evidence).lower()
        missing = [t for t in terms if t not in blob]
        return missing if len(missing) >= 2 and len(missing) / len(terms) >= 0.6 else []

    # ------------------------------------------------------------------ #
    # evidence shaping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _with_table_evidence(evidence: list[SearchHit], table: dict[str, Any]) -> list[SearchHit]:
        """Make sure the table's own Markdown is citable evidence."""
        chunk_id = f"{table.get('doc_id')}:table:{table.get('number')}:markdown"
        if any(h.chunk_id == chunk_id for h in evidence):
            return evidence
        pages = table.get("pages") or [0]
        hit = SearchHit(chunk_id=chunk_id, doc_id=str(table.get("doc_id", "")), score=1.0,
                        section_number=str(table.get("section_number", "")), page=int(pages[0] or 0),
                        page_end=int(pages[-1] or 0) if len(pages) > 1 else 0, element_type="table",
                        text=str(table.get("markdown", "")), element_ids=list(table.get("element_ids") or []),
                        table_id=f"{table.get('doc_id')}:table:{table.get('number')}",
                        meta={"table_number": table.get("number"), "pages": pages,
                              "stitched": table.get("stitched", False),
                              # the citation marker is [doc, revision, §section, p.page, region];
                              # without the revision here the marker silently degrades to [doc, §…]
                              "revision": str(table.get("revision") or "")})
        return [hit] + list(evidence)

    @staticmethod
    def _with_figure_evidence(evidence: list[SearchHit], figure: dict[str, Any]) -> list[SearchHit]:
        chunk_id = f"{figure.get('doc_id')}:figure:{figure.get('number')}:description"
        if any(h.chunk_id == chunk_id for h in evidence):
            return evidence
        text = "\n".join(x for x in [figure.get("caption"), figure.get("description"),
                                     figure.get("ocr_text")] if x)
        hit = SearchHit(chunk_id=chunk_id, doc_id=str(figure.get("doc_id", "")), score=1.0,
                        section_number=str(figure.get("section_number", "")), page=int(figure.get("page") or 0),
                        element_type="figure", text=text, figure_id=figure.get("figure_id"),
                        meta={"figure_number": figure.get("number"), "kind": figure.get("kind"),
                              "revision": str(figure.get("revision") or "")})
        return [hit] + list(evidence)

    # ------------------------------------------------------------------ #
    # decline
    # ------------------------------------------------------------------ #
    def _decline_decision(self, plan: Plan, evidence: Sequence[SearchHit], gaps: list[str]) -> str:
        """Return a human-readable reason to decline, or an empty string to answer."""
        if not evidence:
            return "no evidence was retrieved from the corpus"
        top = max(h.score for h in evidence)
        strong = [h for h in evidence if h.score >= self.agent_config.min_evidence_score]
        if not strong:
            return (f"the retrieved evidence is too weak (best score {top:.3f} < "
                    f"{self.agent_config.min_evidence_score})")
        text_blob = " ".join(h.text for h in evidence).lower()
        denial = self._false_premise(plan, evidence)
        if denial:
            return denial
        numeric_premises = [m.group(0) for m in _NUMERIC_PREMISE.finditer(plan.question)]
        absent = [n for n in numeric_premises if n.lower() not in text_blob]
        if numeric_premises and len(absent) == len(numeric_premises):
            return f"the question's premise ({', '.join(absent[:3])}) does not occur anywhere in the corpus"
        if plan.identifiers and not any(i.lower() in text_blob for i in plan.identifiers):
            if plan.question_type in ("single_hop", "cross_document"):
                return (f"no evidence mentions {', '.join(plan.identifiers[:3])}; the corpus does not "
                        f"contain this entity")
        missing_terms = self._uncovered_terms(plan, evidence)
        if missing_terms:
            return (f"the corpus does not cover {', '.join(missing_terms[:4])} "
                    f"(searched {len(evidence)} pieces of evidence)")
        return ""

    def _false_premise(self, plan: Plan, evidence: Sequence[SearchHit]) -> str:
        """Detect a premise the corpus explicitly denies rather than merely lacks.

        Two kinds of unanswerable question behave differently:

        * the corpus simply never mentions the subject — decline with "not in
          the corpus";
        * the corpus mentions it *in order to deny it* ("the 60 V contract …
          does not exist in this revision", "wireless PHY requirements are out of
          scope of this document") — decline as well, but quote the denial so the
          user learns *why* the premise fails.  Answering the neighbouring
          question silently ("the highest EPR voltage is 48 V") would leave the
          false premise standing, which is exactly the failure the assignment's
          U-questions test for.
        """
        terms = {t.lower() for t in plan.keywords(plan.question)} if hasattr(plan, "keywords") else set()
        terms |= {i.lower() for i in plan.identifiers}
        terms |= {m.group(0).lower() for m in _NUMERIC_PREMISE.finditer(plan.question)}
        for hit in evidence:
            for sentence in re.split(r"(?<=[.;])\s+", hit.text):
                low = sentence.lower()
                if not any(marker in low for marker in ABSENCE_MARKERS):
                    continue
                if any(term and term in low for term in terms):
                    return (f"the corpus states that this premise does not hold: \"{sentence.strip()[:180]}\" "
                            f"({hit.doc_id} §{hit.section_number} p.{hit.page})")
        return ""

    def _decline(self, question: str, trace: AgentTrace, evidence: Sequence[SearchHit], reason: str,
                 started: float, tool_calls: int, plan: Plan) -> AgentAnswer:
        docs = sorted({h.doc_id for h in evidence})
        kinds = sorted({h.element_type for h in evidence})
        answer = (f"INSUFFICIENT — {reason}. "
                  f"I checked {len(evidence)} pieces of evidence across "
                  f"{len(docs)} document(s) ({', '.join(docs) if docs else 'none'}) and "
                  f"{len(kinds)} element type(s); nothing in the ingested corpus answers this question. "
                  f"Declining rather than answering from model knowledge.")
        trace.answer = answer
        trace.declined = True
        trace.decline_reason = reason
        trace.latency_s = time.perf_counter() - started
        trace.steps.append(TraceStep(step=len(trace.steps) + 1, kind="decline", name="decline",
                                     observation=reason, decision="decline: evidence is insufficient"))
        log_event(self.log, "agent.decline", run_id=trace.run_id, reason=reason, evidence=len(evidence),
                  documents=len(docs))
        return AgentAnswer(question=question, answer=answer, declined=True, decline_reason=reason,
                           evidence=list(evidence), trace=trace, plan=plan, tool_calls=tool_calls)

    # ------------------------------------------------------------------ #
    # answer generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dedup_evidence(evidence: list[SearchHit]) -> list[SearchHit]:
        """One copy of each piece of evidence, keeping the best-scoring copy.

        The agent calls several tools, so the same chunk comes back from the text
        index, the page index and the graph.  Six identical copies of one table
        crowd the answer and inflate the evidence count, and every copy turns
        into the same citation marker.
        """
        best: dict[tuple[Any, ...], SearchHit] = {}
        order: list[tuple[Any, ...]] = []
        for hit in evidence:
            # keyed on the *content*, not on the element id: the chunker emits one
            # chunk per row of a stitched table and every one carries the same
            # table text with a different element id, so keying on the id left
            # seven identical copies in the evidence — which then crowded out the
            # table that actually answered the question.
            key = (hit.doc_id, hit.page, hit.element_type,
                   " ".join(str(hit.text).split())[:200].lower())
            if key not in best:
                best[key] = hit
                order.append(key)
                continue
            if hit.score > best[key].score:
                best[key] = hit
            best[key].element_ids = list(dict.fromkeys([*best[key].element_ids, *hit.element_ids]))
        return [best[k] for k in order]

    def _focused_evidence(self, plan: Plan, evidence: Sequence[SearchHit]) -> list[SearchHit]:
        """Pick the evidence the answer should be written from.

        Table and figure questions are answered from the table or figure itself
        plus a couple of supporting paragraphs; other questions take the top hits.
        Without this, a procedure table that happens to mention the same timer
        competes with the timer table that actually defines it.
        """
        kinds = {h.element_type for h in evidence}
        if plan.question_type == "table" and "table" in kinds:
            terms = {t for t in self.planner.keywords(plan.question) if t not in self.INSTRUCTION_WORDS}
            caption = lambda h: str(h.meta.get("caption") or h.meta.get("table_number") or "").lower()

            def table_rank(hit: SearchHit) -> tuple[float, float]:
                overlap = len({t for t in terms if t in caption(hit) or t in hit.text[:200].lower()})
                return (overlap / max(len(terms), 1), hit.score)

            tables = [h for h in evidence if h.element_type in ("table", "table_caption")]
            others = [h for h in evidence if h.element_type not in ("table", "table_caption")]
            return sorted(tables, key=table_rank, reverse=True)[:2] + others[:3]
        if plan.question_type == "figure" and kinds & {"figure", "figure_description", "figure_caption"}:
            figs = [h for h in evidence if h.element_type.startswith("figure")]
            others = [h for h in evidence if not h.element_type.startswith("figure")]
            return sorted(figs, key=lambda h: -h.score)[:2] + others[:3]
        text_hits = [h for h in evidence if h.element_type != "graph_edge"]
        graph_hits = [h for h in evidence if h.element_type == "graph_edge"]
        body = [h for h in text_hits
                if h.section_number or h.element_type not in ("section", "heading", "page_image")]
        if body:
            text_hits = body
        ranked = sorted(text_hits, key=lambda h: -h.score)[: max(self.agent_config.top_k, 6)]
        if graph_hits and len(ranked) < 3:
            # graph relations support an answer ("tSenderResponse is defined in
            # Table 6.8"); they are used when the text path is thin, never to
            # crowd out text evidence
            ranked = ranked + sorted(graph_hits, key=lambda h: -h.score)[:1]
        return ranked

    def _compose(self, question: str, plan: Plan, evidence: Sequence[SearchHit], trace: AgentTrace,
                 docs: Sequence[str] | None = None
                 ) -> tuple[str, list[Citation], VerificationResult | None]:
        """Generate the answer from the evidence, then verify its citations."""
        ranked = self._dedup_evidence(list(evidence))
        if docs:
            allowed = {d.strip() for d in docs if d.strip()}
            ranked = [h for h in ranked if h.doc_id in allowed]
        ranked = self._focused_evidence(plan, ranked)
        if plan.question_type == "table" and hasattr(self.llm, "per_item_limit"):
            # the rows of the table that answers the question are all part of the
            # answer: "which Rp values advertise 1.5 A and 3.0 A" needs two rows,
            # a bit-field table needs one row per field
            self.llm.per_item_limit = 5
        citations = [citation_from_hit(h) for h in ranked]
        markers = {c.to_marker(): c for c in citations}
        blocks = []
        for hit in ranked:
            marker = citation_from_hit(hit).to_marker()
            blocks.append(f"Evidence [{marker}] ({hit.element_type}, score {hit.score:.3f})\n{hit.text[:1500]}")
        prompt = (f"Question: {question}\nQuestion type: {plan.question_type}\n"
                  f"Sub-questions: {'; '.join(plan.sub_questions)}\n\n"
                  + "\n\n".join(blocks)
                  + "\n\nAnswer the question in two to five sentences, citing each sentence. "
                    "Use numbers and identifiers exactly as they appear in the evidence.")
        evidence_payload = [{"text": h.text, "citation": citation_from_hit(h).to_marker()} for h in ranked]
        text = ""
        response = None
        try:
            response = generate(self.llm, prompt, evidence=evidence_payload, question=question,
                                system=SYSTEM_PROMPT)
            text = (response.text or "").strip()
            # compound questions ("what does X do, and where is it defined?") are
            # answered part by part: one pass scores only the part whose wording
            # overlaps the evidence most, and the other part is silently dropped
            import os as _os
            if len(plan.sub_questions) > 1 and not _os.environ.get("SPECINTEL_ABLATE_MULTIPART"):
                parts = [text]
                for sub in plan.sub_questions[1:]:
                    sub_response = generate(self.llm, f"Question: {sub}\n\n" + "\n\n".join(blocks)
                                            + "\n\nAnswer this part, citing each sentence.",
                                            evidence=evidence_payload, question=sub, system=SYSTEM_PROMPT)
                    sub_text = (sub_response.text or "").strip()
                    if sub_text and sub_text[:80].lower() not in " ".join(parts).lower():
                        parts.append(sub_text)
                    if response is not None:
                        response.prompt_tokens += sub_response.prompt_tokens
                        response.completion_tokens += sub_response.completion_tokens
                text = " ".join(p for p in parts if p)
        except Exception as exc:                                # noqa: BLE001 - degrade, never crash the run
            log_exception(self.log, "agent.generate.failed", exc)
            trace.warnings.append(f"generation failed: {exc}")
            return "", [], None
        trace.steps.append(TraceStep(step=len(trace.steps) + 1, kind="answer", name="generate",
                                     observation=text[:300], tokens=getattr(response, "total_tokens", 0),
                                     decision=f"generated with {getattr(self.llm, 'name', 'none')}"))
        verification: VerificationResult | None = None
        if self.agent_config.verify_citations and text and not text.upper().startswith(INSUFFICIENT):
            verification = self.verifier.verify(text, ranked, citations)
            if verification.dropped:
                trace.steps.append(TraceStep(step=len(trace.steps) + 1, kind="reflect", name="citation_verifier",
                                             observation=f"{len(verification.dropped)} unsupported claim(s): "
                                                         + " | ".join(verification.dropped[:2]),
                                             decision="drop unsupported claims"))
            if verification.kept.strip():
                text = verification.kept.strip()
            kept_markers = {m.strip() for m in _MARKER_RE.findall(text)}
            citations = [c for c in citations if c.to_marker() in kept_markers] or citations[:1]
            citations = _add_twin_citations(citations, ranked)
        elif citations:
            text = text + " " + " ".join(c.to_marker() for c in citations[:1]) if _MARKER_RE.search(text) is None else text
        return text, citations, verification


def _collect_graph_refs(graph: Any, into: dict[str, list[tuple[str, str]]]) -> None:
    """Record the tables and figures a ``graph_query`` result names.

    ``Table:pd_table_6_8`` carries its number in ``properties``; the agent needs
    ``("pd", "6.8")`` so it can call ``get_table`` and read the rows.
    """
    nodes = graph.get("nodes") if isinstance(graph, dict) else getattr(graph, "nodes", None)
    for node in nodes or []:
        if isinstance(node, dict):
            node_type = str(node.get("node_type") or "")
            doc_id = str(node.get("doc_id") or "")
            properties = node.get("properties") or {}
        else:
            node_type = str(getattr(node, "node_type", ""))
            doc_id = str(getattr(node, "doc_id", ""))
            properties = getattr(node, "properties", {}) or {}
        number = str(properties.get("number") or "")
        if not (doc_id and number):
            continue
        if node_type == "Table":
            pair = (doc_id, number)
            if pair not in into["table"]:
                into["table"].append(pair)
        elif node_type == "Figure":
            pair = (doc_id, number)
            if pair not in into["figure"]:
                into["figure"].append(pair)


def _add_twin_citations(citations: list[Citation], ranked: Sequence[SearchHit]) -> list[Citation]:
    """Cite the second copy when the same content exists in another document.

    The same test procedure appears in the conformance test specification and in
    the functional test specification, and the same register list appears in two
    Tech 2 books.  Citing only the copy that happened to rank first hides the
    other from a reader who is holding the other document, so when a cited piece
    of evidence has a near-identical twin, both are cited.
    """
    by_marker = {h_text: h for h_text in []}          # placeholder for type checkers
    keys = {c.to_marker() for c in citations}
    out = list(citations)
    for hit in ranked:
        marker = citation_from_hit(hit).to_marker()
        if marker in keys:
            continue
        twin = any(_jaccard(hit.text, other.text) >= 0.75 for other in ranked
                   if citation_from_hit(other).to_marker() in keys)
        if twin:
            out.append(citation_from_hit(hit))
            keys.add(marker)
        if len(out) >= max(2, len(citations)):
            break
    return out


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


#: sentences that deny a premise rather than support it
ABSENCE_MARKERS = ("does not exist", "does not apply", "not exist in this revision", "is out of scope",
                   "are out of scope", "out of scope of this document", "is not covered", "are not covered",
                   "does not cover", "no such", "is not specified", "are not specified", "is not defined",
                   "is retained from the revision", "not supported in this revision")

#: numeric premises ("60 V", "240 W") that must actually occur in the corpus
_NUMERIC_PREMISE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|ms|µs|us|ns|V|mV|A|mA|W|kW|kΩ|Ω|bits?|bytes?)\b",
                              re.I)
_MARKER_RE = re.compile(r"\[([^\[\]]+?)\]")
_HYPHEN_TERM = re.compile(r"\b([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})\b")
#: numbered standards and revisions: 802.11be, USB4, R3.2, Release 2.5
_NUMBERED_RE = re.compile(r"\b(?:\d{2,}\.\d+[a-z]*|USB\d(?:[.]\d)?|R\d\.\d|Release\s+\d\.\d)\b", re.I)


def build_agent(config: Any, *, toolkit: Any | None = None, llm: Any | None = None,
                agent_config: AgentConfig | None = None) -> SpecAgent:
    """Construct an agent wired to the persisted index, graph and elements."""
    from specintel.agent.tools import Toolkit

    toolkit = toolkit or Toolkit.from_config(config)
    return SpecAgent(toolkit, llm=llm, config=config, agent_config=agent_config)


def ask(question: str, *, config: Any | None = None, docs: Sequence[str] | None = None,
        toolkit: Any | None = None, agent_config: AgentConfig | None = None,
        question_type: str | None = None) -> AgentAnswer:
    """One-shot helper used by the CLI and the tests."""
    if config is None:
        from specintel.common.config import load_config

        config = load_config()
    agent = build_agent(config, toolkit=toolkit, agent_config=agent_config)
    return agent.answer(question, docs=docs, question_type=question_type)
