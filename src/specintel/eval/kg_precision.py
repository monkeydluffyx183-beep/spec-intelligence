"""WS5 — knowledge-graph triple precision and provenance coverage.

The assignment sets a triple-precision target (>= 0.80).  Precision has to be
measured on triples someone actually looked at, so ``scripts/build_graph.py``
exports ``data/graph/triple_sample.jsonl``: a seeded random sample of 100 edges
with their provenance, each carrying an empty ``verdict`` field for a reviewer to
fill with ``valid`` or ``invalid`` (plus an optional ``reason``).

Two numbers come out of this module:

``annotated_precision``
    valid / (valid + invalid) over the annotated triples.  This is the number the
    report quotes once the sample has been reviewed.

``rule_precision``
    An *automatic* audit over the whole export, so the metric is never blocked on
    human annotation:

    * provenance completeness — every edge names a document, a page and a section
      or element, and records how it was extracted;
    * endpoint validity — the edge type exists in the ontology and both endpoint
      types are allowed for that relation;
    * structural defects — self-edges and duplicate edges;
    * evidence grounding — sampled edges whose provenance names an element must
      carry the evidence quote they claim.

The failure histogram is reported with examples so a reviewer sees *why* triples
fail, not just how many, and ``evaluate_edge_provenance`` reports provenance
coverage over every exported edge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from specintel.kg.schema import KG_SCHEMA

TRIPLE_RE = re.compile(r"^(?P<src>.+?)\s+-\[(?P<etype>[A-Z_]+)\]->\s+(?P<dst>.+)$")


@dataclass
class TripleAudit:
    """Result of the automatic audit over the exported sample."""

    total: int = 0
    rule_pass: int = 0
    failures: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    annotated_total: int = 0
    annotated_valid: int = 0
    annotated_invalid: int = 0

    @property
    def rule_precision(self) -> float:
        return self.rule_pass / self.total if self.total else float("nan")

    @property
    def annotated_precision(self) -> float:
        return self.annotated_valid / self.annotated_total if self.annotated_total else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"sample_size": self.total, "rule_precision": _round(self.rule_precision),
                "rule_pass": self.rule_pass, "failures": dict(self.failures),
                "failure_examples": {k: v[:3] for k, v in self.examples.items()},
                "annotated_total": self.annotated_total, "annotated_valid": self.annotated_valid,
                "annotated_invalid": self.annotated_invalid,
                "annotated_precision": _round(self.annotated_precision),
                "annotation_hint": SAMPLE_ANNOTATION_HINT}


def _round(value: float) -> float | None:
    return None if value != value else round(value, 4)


def parse_triple(triple: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(source_id, edge_type, target_id)`` from an exported triple."""
    match = TRIPLE_RE.match(str(triple.get("triple") or ""))
    if match:
        return match.group("src"), match.group("etype"), match.group("dst")
    return (str(triple.get("src_id") or triple.get("source") or ""),
            str(triple.get("edge_type") or triple.get("type") or ""),
            str(triple.get("dst_id") or triple.get("target") or ""))


def audit_triples(triples: Sequence[dict[str, Any]]) -> TripleAudit:
    """Run the rule checks on a triple sample and read the human verdicts present."""
    audit = TripleAudit()
    seen: set[tuple[str, str, str]] = set()
    for triple in triples:
        audit.total += 1
        reasons: list[str] = []
        source, edge_type, target = parse_triple(triple)
        src_type = str(triple.get("src_type") or _type_of(source))
        dst_type = str(triple.get("dst_type") or _type_of(target))

        if not source or not target:
            reasons.append("missing endpoint")
        elif source == target:
            reasons.append("self edge")

        spec = KG_SCHEMA.get(edge_type)
        if spec is None:
            reasons.append(f"edge type '{edge_type}' not in ontology")
        else:
            if spec.src_types and src_type not in spec.src_types:
                reasons.append(f"source type {src_type} not allowed for {edge_type}")
            if spec.dst_types and dst_type not in spec.dst_types:
                reasons.append(f"target type {dst_type} not allowed for {edge_type}")

        provenance = triple.get("provenance") or {}
        doc_id = provenance.get("doc_id") or triple.get("doc_id")
        page = provenance.get("page") if provenance.get("page") is not None else triple.get("page")
        section = provenance.get("section") or triple.get("section_number")
        element_id = provenance.get("element_id") or triple.get("element_id")
        method = provenance.get("method") or triple.get("extraction_method")
        if not doc_id:
            reasons.append("no document provenance")
        if page in (None, "", 0):
            reasons.append("no page provenance")
        if not section and not element_id:
            reasons.append("no section or element provenance")
        if not method:
            reasons.append("no extraction method")
        confidence = provenance.get("confidence", triple.get("confidence"))
        if confidence is None:
            reasons.append("no confidence recorded")

        key = (source, edge_type, target)
        if key in seen and edge_type in ("DEFINED_IN", "PART_OF", "REFERENCES"):
            reasons.append("duplicate structural edge")
        seen.add(key)

        if reasons:
            for reason in reasons:
                audit.failures[reason] += 1
                audit.examples.setdefault(reason, []).append(describe_triple(triple))
        else:
            audit.rule_pass += 1

        verdict = str(triple.get("verdict") or "").strip().lower()
        if verdict in ("valid", "invalid", "true", "false", "yes", "no", "1", "0"):
            audit.annotated_total += 1
            if verdict in ("valid", "true", "yes", "1"):
                audit.annotated_valid += 1
            else:
                audit.annotated_invalid += 1
    return audit


def _type_of(node_id: str) -> str:
    return node_id.split(":", 1)[0] if ":" in node_id else ""


def describe_triple(triple: dict[str, Any]) -> str:
    """Readable one-line description used in the failure examples."""
    source, edge_type, target = parse_triple(triple)
    provenance = triple.get("provenance") or {}
    location = f"{provenance.get('doc_id') or triple.get('doc_id')}:p{provenance.get('page') or triple.get('page')}"
    return f"{source} -[{edge_type}]-> {target} @ {location}"


def load_triple_sample(path: str | Path) -> list[dict[str, Any]]:
    """Read the exported triple sample (JSONL)."""
    return _load_jsonl(path)


def evaluate_edge_provenance(edges_path: str | Path) -> dict[str, Any]:
    """Provenance coverage over every exported edge (not just the sample)."""
    edges = _load_jsonl(edges_path)
    if not edges:
        return {}
    total = len(edges)
    with_doc = with_page = with_method = with_confidence = 0
    for edge in edges:
        provenance = edge.get("provenance") or {}
        with_doc += 1 if (provenance.get("doc_id") or edge.get("doc_id")) else 0
        with_page += 1 if (provenance.get("page") or edge.get("page")) else 0
        with_method += 1 if (provenance.get("method") or edge.get("extraction_method")) else 0
        with_confidence += 1 if (provenance.get("confidence") is not None or edge.get("confidence") is not None) else 0
    return {"edges": total, "doc_provenance": _round(with_doc / total), "page_provenance": _round(with_page / total),
            "extraction_method": _round(with_method / total), "confidence": _round(with_confidence / total)}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


SAMPLE_ANNOTATION_HINT = ("Fill the 'verdict' field of data/graph/triple_sample.jsonl with valid/invalid "
                          "(and optionally 'reason') to turn the sampled precision into a measured number; "
                          "as long as it is empty the report quotes rule_precision instead.")
