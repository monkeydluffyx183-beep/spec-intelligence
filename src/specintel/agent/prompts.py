"""WS4 — prompts, kept in one place so they can be reviewed and versioned.

Every prompt in the system lives here instead of being assembled inline at the
call site.  Two reasons: the assignment asks for components that are
explainable, and a prompt that is versioned with the code can be quoted in the
report next to the results it produced.  Prompts are deliberately short and
free of few-shot examples: the model is a local open-weight model, and the
evidence selection (not prompt engineering) is what the pipeline is testing.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Used by the agent loop and the evaluation harness.  The wording is the
#: contract the citation verifier checks: every sentence carries one marker,
#: facts come from the evidence, and "not covered" is an accepted answer.
SYSTEM_PROMPT = (
    "You are a specification analyst answering questions about USB Power Delivery, "
    "USB Type-C, USB4 and Qi/Tech 2 specifications. Answer only from the evidence "
    "given to you; never use outside knowledge. Cite every sentence with the marker "
    "shown in the evidence it came from. Quote numbers, timer names, test ids and "
    "table numbers exactly as they appear. If the evidence does not answer the "
    "question, reply with 'INSUFFICIENT — ' followed by what is missing."
)

PLANNING_PROMPT = """\
Classify the question and list the evidence it needs.

Question: {question}

Answer with JSON: {{"question_type": "single_hop|table|figure|cross_document|comparison|unanswerable",
"documents": ["pd"|"typec"|"usb4"|"tech2"|...], "identifiers": ["..."], "keywords": ["..."],
"sub_questions": ["..."]}}
"""

ANSWER_PROMPT = """\
Question: {question}
Question type: {question_type}
Sub-questions: {sub_questions}

{evidence}

Answer the question in two to five sentences, citing each sentence with the marker of
the evidence it came from. Use numbers and identifiers exactly as they appear.
"""

SUBANSWER_PROMPT = """\
Question: {sub_question}

{evidence}

Answer this part of the question in one or two sentences, citing each sentence.
"""

VERIFY_PROMPT = """\
Is every sentence of the answer supported by the evidence?

Answer: {answer}

Evidence:
{evidence}

Answer with JSON: {{"unsupported": ["sentence", ...], "notes": "..."}}
"""

FIGURE_PROMPT = """\
The image is {kind} from {doc_id} page {page}. Caption: {caption}
Text recovered from the page around this figure (may be empty):
{labels}

Describe the figure and return JSON with the fields that apply:
{{"summary": "...", "states": [], "transitions": [{{"from": "", "to": "", "trigger": ""}}],
"signals": [], "events": [], "timers": [], "notes": []}}
"""

#: The seven tools, described for a model that is choosing one.  The agent's own
#: planner is deterministic (see ``planner.py``); this table is what an
#: LLM-driven planner or an MCP client receives.
TOOL_PROMPTS: list[dict[str, str]] = [
    {"name": "search_text", "description": "Ranked chunk search over prose, table bodies and "
                                           "figure descriptions. Optional filters by document, "
                                           "section, element type, page.",
     "arguments": '{"query": str, "k": int, "docs": [str], "sections": [str], '
                  '"element_types": [str], "tables_only": bool}'},
    {"name": "search_pages", "description": "Vision retrieval: rank page images with MaxSim and "
                                            "return the elements of the winning pages.",
     "arguments": '{"query": str, "k": int}'},
    {"name": "graph_query", "description": "Knowledge graph lookup, neighbours, path, subgraph or "
                                           "statistics over entities such as timers, states, "
                                           "test cases and parameters.",
     "arguments": '{"kind": "lookup|neighbors|path|subgraph|stats", "name": str, "src": str, '
                  '"dst": str, "edge_types": [str], "depth": int}'},
    {"name": "get_section", "description": "Return a whole section by document and section number, "
                                           "in document order.",
     "arguments": '{"doc_id": str, "section_number": str}'},
    {"name": "get_table", "description": "Return one table with headers, rows, page range and the "
                                         "sections that reference it.",
     "arguments": '{"table_id": str}'},
    {"name": "get_figure", "description": "Return one figure: crop path, recovered labels, "
                                          "structured content and description.",
     "arguments": '{"figure_id": str}'},
    {"name": "resolve_entity", "description": "Resolve a name, abbreviation or alias to graph node "
                                              "ids (\"tSenderResponse\", \"PS_RDY\", \"TD 4.2.4\").",
     "arguments": '{"name": str}'},
]


def render_prompt(template: str, **values: Any) -> str:
    """``template.format`` with a readable error when a slot is missing."""
    try:
        return template.format(**values)
    except KeyError as exc:                                     # pragma: no cover - developer error
        raise KeyError(f"prompt template is missing the slot {exc}") from exc


def evidence_block(items: Sequence[dict[str, str]]) -> str:
    """Render retrieved evidence as numbered, cited blocks."""
    lines = []
    for i, item in enumerate(items, start=1):
        marker = item.get("citation", "")
        lines.append(f"[{i}] {marker}\n{item.get('text', '')}")
    return "\n\n".join(lines)
