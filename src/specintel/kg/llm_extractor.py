"""WS3 — pass 2: schema-guided LLM/VLM relation extraction.

Pass 1 (rules) covers everything strictly patterned.  Pass 2 handles the
paragraphs where the relation is expressed in prose, for example
"PE_SRC_Negotiate_Capability starts tSenderResponse while the Source waits for
the exchange to finish" or "PCIe tunnelling is defined in section 7.6".

Design constraints from the assignment:

* the model is **constrained to the schema** and must answer with JSON;
* every returned triple must name its evidence so provenance can be attached;
* anything that violates the ontology is rejected and reported, not inserted;
* the pass is optional — when no local model is reachable the pipeline records
  ``llm_pass: unavailable`` and keeps the rule graph, instead of failing.

The prompt lists the allowed node and edge types, which is what keeps a small
local model inside the ontology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from specintel.common.llm import ExtractiveLLM, generate, get_llm, parse_json_object
from specintel.common.models import Edge, Element, Node, new_id
from specintel.kg.rules import ExtractionResult, RuleExtractor, TIMER_RE, STATE_RE, TEST_ID_RE
from specintel.kg.schema import KG_SCHEMA, NODE_TYPES, validate_edge, validate_node

ALLOWED_EDGES = sorted(KG_SCHEMA)

PROMPT = """You extract knowledge-graph triples from an engineering specification.

Allowed node types: {node_types}
Allowed relation types: {edge_types}

Extract only relations that the text below states explicitly. For each relation return:
  {{"src": {{"type": "<node type>", "name": "<exact name from the text>"}},
    "rel": "<relation type>",
    "dst": {{"type": "<node type>", "name": "<exact name from the text>"}},
    "evidence": "<the sentence you used, verbatim, max 240 chars>",
    "confidence": <0.0-1.0>}}

Rules:
 * copy names exactly as printed (tSenderResponse, PE_SRC_Ready, Source_Capabilities, Table 6.8);
 * src/dst types must fit the relation, for example TRANSITIONS_TO is State -> State;
 * never invent entities that are not in the text;
 * answer with the JSON list only.

Document: {doc_id} (revision {revision}), section {section}, page {page}
Text:
\"\"\"{text}\"\"\"

JSON list:"""


@dataclass
class LLMExtractionReport:
    backend: str = "unavailable"
    model: str = ""
    elements_processed: int = 0
    triples_returned: int = 0
    accepted: int = 0
    rejected: int = 0
    notes: list[str] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "elements_processed": self.elements_processed,
                "triples_returned": self.triples_returned, "accepted": self.accepted,
                "rejected": self.rejected, "notes": self.notes,
                "rejections": self.rejections[:20]}


class LLMExtractor:
    """Schema-constrained extraction over prose elements."""

    def __init__(self, llm: Any | None = None, *, max_elements: int = 0, min_chars: int = 120) -> None:
        self.llm = llm
        self.max_elements = max_elements
        self.min_chars = min_chars

    def available(self) -> bool:
        """False when the resolved backend is the deterministic extractive one."""
        return getattr(self.llm, "backend", "extractive") not in ("extractive", None)

    def run(self, ctx: Any, *, base: ExtractionResult | None = None) -> tuple[ExtractionResult, LLMExtractionReport]:
        """Extract triples from the prose of one document."""
        report = LLMExtractionReport(backend=getattr(self.llm, "backend", "extractive"),
                                     model=getattr(self.llm, "name", "none"))
        result = ExtractionResult()
        if not self.available():
            report.notes.append("no local LLM reachable: pass 2 skipped, rule graph retained "
                                "(the run is fully reproducible without a GPU)")
            return result, report
        candidates = [el for el in ctx.elements
                      if el.element_type in ("paragraph", "note", "list_item") and len(el.text) >= self.min_chars]
        if self.max_elements:
            candidates = candidates[:self.max_elements]
        helper = RuleExtractor()
        for element in candidates:
            report.elements_processed += 1
            prompt = PROMPT.format(node_types=", ".join(NODE_TYPES), edge_types=", ".join(ALLOWED_EDGES),
                                   doc_id=ctx.doc_id, revision=ctx.revision, section=element.section_number,
                                   page=element.page, text=element.text[:2400])
            response = generate(self.llm, prompt, max_tokens=900)
            payload = parse_json_object(response.text)
            triples = _as_triple_list(payload)
            if response.error:
                report.notes.append(f"backend error: {response.error}")
            if not triples:
                continue
            report.triples_returned += len(triples)
            self._insert(base or result, result, triples, element, ctx, report, helper)
        return result, report

    # -- insertion --------------------------------------------------------- #
    def _insert(self, base: ExtractionResult, result: ExtractionResult, triples: Sequence[dict[str, Any]],
                element: Element, ctx: Any, report: LLMExtractionReport, helper: RuleExtractor) -> None:
        for triple in triples:
            src = triple.get("src") or {}
            dst = triple.get("dst") or {}
            relation = str(triple.get("rel") or "").upper()
            src_name, dst_name = str(src.get("name", "")).strip(), str(dst.get("name", "")).strip()
            if not src_name or not dst_name or relation not in KG_SCHEMA:
                report.rejected += 1
                report.rejections.append({"reason": "malformed triple", "triple": triple})
                continue
            src_node = _node_for(base, result, helper, ctx, src.get("type", "Parameter"), src_name, element)
            dst_node = _node_for(base, result, helper, ctx, dst.get("type", "Parameter"), dst_name, element)
            src_node, dst_node = _apply_repairs(src_node, dst_node, relation, element)
            edge = Edge(edge_id=new_id("e"), src_id=src_node.node_id, dst_id=dst_node.node_id,
                        edge_type=relation, doc_id=ctx.doc_id, revision=ctx.revision,
                        section_number=element.section_number, page=element.page,
                        element_id=element.element_id, extraction_method="llm",
                        confidence=float(triple.get("confidence", 0.7) or 0.7),
                        evidence=str(triple.get("evidence", element.text[:200]))[:400])
            node_types = {n.node_id: n.node_type for n in [*base.nodes, *result.nodes]}
            issues = validate_node(src_node) + validate_node(dst_node) + validate_edge(edge, node_types)
            if issues:
                report.rejected += 1
                report.rejections.extend(i.to_dict() for i in issues)
                continue
            if not any(n.node_id == src_node.node_id for n in result.nodes):
                result.nodes.append(src_node)
            if not any(n.node_id == dst_node.node_id for n in result.nodes):
                result.nodes.append(dst_node)
            result.edges.append(edge)
            report.accepted += 1


def _node_for(base: ExtractionResult, result: ExtractionResult, helper: RuleExtractor, ctx: Any,
              node_type: str, name: str, element: Element) -> Node:
    """Reuse an existing node when the rule pass already found the entity."""
    node_type = node_type if node_type in NODE_TYPES else "Parameter"
    candidate_id = helper.node_id(node_type, name)
    if base is not None:
        for node in base.nodes:
            if node.node_id == candidate_id:
                return node
    for node in result.nodes:
        if node.node_id == candidate_id:
            return node
    # the model may mislabel the type; fall back to a rule-known type for the name
    for alt_type in ("Timer", "State", "Message", "TestCase", "Check", "Section", "Table", "Figure"):
        alt_id = helper.node_id(alt_type, name)
        if base is not None and any(n.node_id == alt_id for n in base.nodes):
            return next(n for n in base.nodes if n.node_id == alt_id)
    return Node(node_id=candidate_id, node_type=node_type, name=name, canonical_name=name, doc_id=ctx.doc_id,
                revision=ctx.revision, section_number=element.section_number, page=element.page,
                element_ids=[element.element_id])


def _apply_repairs(src: Node, dst: Node, relation: str, element: Element) -> tuple[Node, Node]:
    """Fix common model mistakes that would otherwise be rejected.

    Example: the model emits ``SENT_BY`` with ``State -> State`` for
    "PE_SRC_Send_Capabilities starts tSenderResponse"; the name is a timer, so
    the node type is corrected from the identifier shape instead of throwing the
    triple away.
    """
    spec = KG_SCHEMA[relation]
    if src.node_type not in spec.src_types:
        src = _retype(src, spec.src_types)
    if dst.node_type not in spec.dst_types:
        dst = _retype(dst, spec.dst_types)
    return src, dst


def _retype(node: Node, allowed: Sequence[str]) -> Node:
    guess = _guess_type(node.name)
    if guess in allowed:
        node.node_type = guess
    elif "Parameter" in allowed:
        node.node_type = "Parameter"
    elif "Section" in allowed:
        node.node_type = "Section"
    elif "Message" in allowed:
        node.node_type = "Message"
    return node


def _guess_type(name: str) -> str:
    if TIMER_RE.fullmatch(name):
        return "Timer"
    if STATE_RE.fullmatch(name):
        return "State"
    if TEST_ID_RE.search(name) or name.startswith("["):
        return "Check" if name.startswith("[") else "TestCase"
    if name.lower().startswith("table "):
        return "Table"
    if name.lower().startswith("figure "):
        return "Figure"
    if name.startswith("§") or name[:1].isdigit():
        return "Section"
    return "Message"


def _as_triple_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in ("triples", "relations", "results", "items"):
            if isinstance(payload.get(key), list):
                return [p for p in payload[key] if isinstance(p, dict)]
        if {"src", "dst", "rel"} <= set(payload):
            return [payload]
    return []


def extraction_available(config: Any = None) -> bool:
    """Report whether pass 2 can run in this environment (used by the report)."""
    llm = get_llm(config)
    return getattr(llm, "backend", "extractive") != "extractive"
