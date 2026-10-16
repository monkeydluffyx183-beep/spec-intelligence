"""WS3 — the ontology: node types, edge types, allowed pairs and validation.

The ontology follows the draft in the assignment (Task 3, Step 1) and refines it
against what the seven documents actually contain.  Two things are deliberate:

* **Allowed pairs.**  Every edge type declares which node types may sit at each
  end.  ``TRANSITIONS_TO`` between a ``State`` and a ``Timer`` is rejected with a
  message naming the offending triple, which is what "report what was rejected"
  means in practice.
* **Provenance is mandatory.**  An edge without ``doc_id`` + ``page`` +
  ``extraction_method`` cannot be cited, so it is invalid by construction.

``KG_SCHEMA`` is exported to ``docs/KG_SCHEMA.md`` by ``scripts/build_graph.py``
so the written deliverable and the code cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.common.models import EDGE_TYPES, NODE_TYPES, Edge, Node


@dataclass(frozen=True)
class EdgeSpec:
    """Declared endpoints of one relation and what the relation means."""

    edge_type: str
    src_types: tuple[str, ...]
    dst_types: tuple[str, ...]
    meaning: str
    symmetric: bool = False
    properties: tuple[str, ...] = ()


KG_SCHEMA: dict[str, EdgeSpec] = {
    "DEFINED_IN": EdgeSpec("DEFINED_IN", tuple(NODE_TYPES), ("Section", "Table", "Figure"),
                           "the entity is defined in this section, table or figure"),
    "REFERENCES": EdgeSpec("REFERENCES", ("Section", "Table", "Figure", "Message", "TestCase", "Assertion",
                                          "Revision", "ECN", "Adapter"),
                           ("Section", "Table", "Figure", "Document", "ECN", "Revision"),
                           "a cross-reference, possibly to another document or an older revision"),
    "SENT_BY": EdgeSpec("SENT_BY", ("Message", "Tech2Packet"), ("Role", "State", "Document"),
                        "which role or state sends the message", properties=("direction", "condition")),
    "RESPONDS_TO": EdgeSpec("RESPONDS_TO", ("Message", "Tech2Packet"), ("Message", "Tech2Packet"),
                            "this message is the answer to that message", properties=("within_timer",)),
    "CONTAINS": EdgeSpec("CONTAINS", ("DataObject", "Message", "Tech2Packet", "Section", "Table", "Tunnel"),
                         ("Field", "DataObject", "Message", "Section", "Tunnel", "Adapter"),
                         "payload or hierarchy containment"),
    "TRANSITIONS_TO": EdgeSpec("TRANSITIONS_TO", ("State",), ("State",),
                               "state machine transition", properties=("trigger", "condition", "actions")),
    "STARTS_TIMER": EdgeSpec("STARTS_TIMER", ("State", "TestCase", "Section"), ("Timer",),
                             "this state or test starts the timer", properties=("condition",)),
    "STOPS_TIMER": EdgeSpec("STOPS_TIMER", ("State", "TestCase", "Section"), ("Timer",),
                            "this state or test stops the timer", properties=("condition",)),
    "HAS_FIELD": EdgeSpec("HAS_FIELD", ("DataObject", "Message", "Tech2Packet", "Table"),
                          ("Field",), "the data object is laid out with this field"),
    "HAS_VALUE": EdgeSpec("HAS_VALUE", ("Field", "Timer", "Counter", "Parameter", "PowerProfile"),
                          ("Parameter", "Timer", "Counter", "Field"),
                          "the entity holds this value, limit or unit",
                          properties=("value", "min", "max", "unit")),
    "TESTS": EdgeSpec("TESTS", ("TestCase",), ("Assertion", "Requirement", "Message", "State", "Timer",
                                               "Parameter", "DataObject", "PowerProfile", "Tech2Packet"),
                      "the test case exercises this requirement or entity"),
    "VERIFIES": EdgeSpec("VERIFIES", ("Check", "TestCase"), ("Assertion", "Requirement", "Timer", "State",
                                                             "Message", "Parameter", "PowerProfile"),
                         "the check verifies this requirement with a stated limit",
                         properties=("limit", "condition")),
    "APPLIES_WHEN": EdgeSpec("APPLIES_WHEN", ("TestCase", "Check", "Assertion"), ("VIFField",),
                             "the test only applies under this condition", properties=("value",)),
    "TUNNELS": EdgeSpec("TUNNELS", ("Adapter", "Tunnel"), ("Protocol", "Adapter", "Document"),
                        "this USB4 adapter tunnels that protocol", properties=("section", "adapter_id")),
    "SUPERSEDES": EdgeSpec("SUPERSEDES", ("Revision", "Document", "ECN"), ("Revision", "Document", "ECN"),
                           "the newer revision replaces the older one", properties=("reason",)),
    "HAS_REVISION": EdgeSpec("HAS_REVISION", ("Document",), ("Revision",),
                             "the document has this revision"),
    "PART_OF": EdgeSpec("PART_OF", ("Section", "Table", "Figure", "Field", "Check"), 
                        ("Section", "Document", "Table", "DataObject", "TestCase"),
                        "hierarchy: this element belongs to that one"),
    "DEPENDS_ON": EdgeSpec("DEPENDS_ON", ("TestCase", "Check", "Message", "State"), 
                           ("Message", "State", "Timer", "Section", "DataObject"),
                           "the entity cannot work without that one",
                           properties=("reason",)),
    "HAS_STATE": EdgeSpec("HAS_STATE", ("Document", "Section", "Figure"), ("State",),
                          "the state machine contains this state"),
}

# node types that must not be created empty (validated before insertion)
REQUIRED_PROPERTIES: dict[str, tuple[str, ...]] = {
    "Timer": (),
    "Counter": (),
    "TestCase": (),
    "Section": ("number",),
    "Table": ("number",),
    "Figure": ("number",),
    "Document": ("revision",),
    "Revision": ("key",),
}

# canonical naming rules, used by entity resolution
CANONICAL_NAME_RULES = (
    (re.compile(r"\bmessage\b", re.I), lambda name: name.replace(" ", "_")),
)


@dataclass
class ValidationIssue:
    """One rejected node or edge, with the reason."""

    kind: str                 # node | edge
    identifier: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.identifier, "reason": self.reason, **self.payload}


def validate_node(node: Node) -> list[ValidationIssue]:
    """Check a node against the ontology."""
    issues: list[ValidationIssue] = []
    if node.node_type not in NODE_TYPES:
        issues.append(ValidationIssue("node", node.node_id, f"unknown node_type '{node.node_type}'"))
    if not node.name.strip():
        issues.append(ValidationIssue("node", node.node_id, "empty name"))
    for prop in REQUIRED_PROPERTIES.get(node.node_type, ()):  # declared properties or node fields
        if prop in (node.properties or {}) or getattr(node, prop, ""):
            continue
        issues.append(ValidationIssue("node", node.node_id,
                                      f"missing required property '{prop}' for {node.node_type}"))
    return issues


def validate_edge(edge: Edge, node_types: dict[str, str], *, require_provenance: bool = True
                  ) -> list[ValidationIssue]:
    """Check an edge against the ontology, the endpoint types and provenance."""
    issues: list[ValidationIssue] = []
    spec = KG_SCHEMA.get(edge.edge_type)
    if spec is None:
        return [ValidationIssue("edge", edge.edge_id, f"unknown edge_type '{edge.edge_type}'",
                                {"src_id": edge.src_id, "dst_id": edge.dst_id})]
    src_type = node_types.get(edge.src_id)
    dst_type = node_types.get(edge.dst_id)
    if src_type is None:
        issues.append(ValidationIssue("edge", edge.edge_id, f"unknown src node '{edge.src_id}'",
                                      {"edge_type": edge.edge_type}))
    elif src_type not in spec.src_types:
        issues.append(ValidationIssue(
            "edge", edge.edge_id,
            f"{edge.edge_type} cannot start at {src_type} (allowed: {', '.join(spec.src_types)})",
            {"src_id": edge.src_id, "dst_id": edge.dst_id, "edge_type": edge.edge_type}))
    if dst_type is None:
        issues.append(ValidationIssue("edge", edge.edge_id, f"unknown dst node '{edge.dst_id}'",
                                      {"edge_type": edge.edge_type}))
    elif dst_type not in spec.dst_types:
        issues.append(ValidationIssue(
            "edge", edge.edge_id,
            f"{edge.edge_type} cannot end at {dst_type} (allowed: {', '.join(spec.dst_types)})",
            {"src_id": edge.src_id, "dst_id": edge.dst_id, "edge_type": edge.edge_type}))
    if require_provenance:
        if not edge.doc_id or not edge.section_number and not edge.page:
            issues.append(ValidationIssue("edge", edge.edge_id,
                                          "missing provenance (doc_id/page/section)",
                                          {"edge_type": edge.edge_type}))
        if edge.extraction_method not in ("rule", "llm", "vlm", "ocr", "parser"):
            issues.append(ValidationIssue("edge", edge.edge_id,
                                          f"unknown extraction_method '{edge.extraction_method}'"))
    if not (0.0 <= edge.confidence <= 1.0):
        issues.append(ValidationIssue("edge", edge.edge_id, f"confidence out of range: {edge.confidence}"))
    return issues


def validate_graph(nodes: Iterable[Node], edges: Iterable[Edge], *, require_provenance: bool = True
                   ) -> tuple[list[Node], list[Edge], list[ValidationIssue]]:
    """Return the accepted nodes/edges plus everything rejected and why."""
    node_list = list(nodes)
    node_types = {n.node_id: n.node_type for n in node_list}
    issues: list[ValidationIssue] = []
    accepted_nodes: list[Node] = []
    for node in node_list:
        node_issues = validate_node(node)
        if node_issues:
            issues.extend(node_issues)
        else:
            accepted_nodes.append(node)
    accepted_edges: list[Edge] = []
    for edge in edges:
        edge_issues = validate_edge(edge, node_types, require_provenance=require_provenance)
        if edge_issues:
            issues.extend(edge_issues)
        else:
            accepted_edges.append(edge)
    return accepted_nodes, accepted_edges, issues


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #
def schema_markdown() -> str:
    """Render the ontology as Markdown (``docs/KG_SCHEMA.md``)."""
    nodes = "\n".join(f"| `{t}` | {_node_meaning(t)} |" for t in NODE_TYPES)
    edges = "\n".join(
        f"| `{spec.edge_type}` | {', '.join(spec.src_types) if len(spec.src_types) < 6 else 'any'} | "
        f"{', '.join(spec.dst_types)} | {spec.meaning} | "
        f"{', '.join(spec.properties) if spec.properties else '—'} |"
        for spec in KG_SCHEMA.values()
    )
    return f"""# Knowledge graph ontology (WS3)

Generated from `src/specintel/kg/schema.py` — do not edit by hand.
Every edge carries provenance: `doc_id`, `revision`, `section_number`, `page`,
`element_id`, `extraction_method` (rule | llm | vlm), `confidence`.

## Node types

| Node type | Meaning |
| --- | --- |
{nodes}

## Edge types

| Relation | From | To | Meaning | Edge properties |
| --- | --- | --- | --- | --- |
{edges}

## Validation

* an unknown node or edge type is rejected;
* an edge whose endpoints are not in the declared types is rejected
  (for example `TRANSITIONS_TO` from a `State` to a `Timer`);
* an edge without provenance, or with a confidence outside `[0, 1]`, is rejected;
* every rejection is written to `data/graph/rejections.jsonl` with its reason and
  counted in the build report.
"""


def _node_meaning(node_type: str) -> str:
    return {
        "Document": "one of the seven source documents, with its revision",
        "Section": "a numbered section or appendix",
        "Table": "a table, including stitched multi-page tables",
        "Figure": "a figure or diagram, with its structured JSON when available",
        "Message": "USB PD control/data/extended message (Source_Capabilities, PS_RDY, Enter_USB …)",
        "DataObject": "PDO/APDO/RDO/VDO type (Fixed Supply, PPS APDO, AVS APDO …)",
        "Field": "a bit field of a data object (B31..30 PDO Type …)",
        "State": "a policy-engine, protocol-layer or Type-C connection state",
        "Timer": "t… timer with min/max/unit when the document states them",
        "Counter": "n… counter with min/max",
        "Parameter": "voltage, current or power limit such as vRd-1.5",
        "Signal": "VBUS, VCONN and other electrical signals",
        "Pin": "CC1, CC2 and other connector pins",
        "Role": "Source, Sink, DRP, cable plug, eMarker, Tech 2 transmitter/receiver",
        "TestCase": "a compliance test case or test description (TEST.PD.*, TD 4.2.4, PTX.CPX.*)",
        "Check": "one check inside a test case ([TEST.PD.PROT.SRC.3#2], [TD.4.2.4.V.3])",
        "Assertion": "a base-specification requirement identifier or a \"shall\" statement",
        "VIFField": "an applicability condition used to select tests",
        "PowerProfile": "BPP, EPP, MPP and other power profiles",
        "Tech2Phase": "Ping, Configuration, Negotiation, Power Transfer",
        "Tech2Packet": "Signal Strength, Control Error, End Power Transfer packets",
        "Router": "USB4 Host Router or Device Router",
        "Adapter": "USB4 Lane Adapter or Protocol Adapter",
        "Tunnel": "a tunnelled protocol path",
        "ECN": "an engineering change notice named in the revision history",
        "Revision": "a document revision (R3.1 V1.6, Release 2.4 …)",
    }.get(node_type, "")
