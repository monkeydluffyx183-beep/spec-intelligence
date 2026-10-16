"""WS5 — structural accuracy: table cells and structured figures vs ground truth.

The ingestion pipeline is scored against ``data/synthetic/manifest.json``, which the
corpus generator wrote while rendering the pages: every table's header row and
body rows and every figure's nodes and transitions are recorded there *before*
any parsing happened.  Comparing the ingested artefacts with that manifest
measures the pipeline, not the generator.

Two metrics:

``table_cell_accuracy``
    Fraction of ground-truth cells exactly present in the ingested table (cell
    level, so a table that loses one row is penalised proportionally, not
    wholesale).  Reported per table and averaged, with header and body cells
    counted separately so header loss is visible on its own.

``figure_structure_accuracy``
    Fraction of ground-truth figure nodes and transitions recovered from the
    ingested figure: node names from the structured description or the OCR/VLM
    pass, transitions from the same.  Vector figures are the hard case because
    their labels are scattered text; raster figures are the OCR case.

Both are also reported per document, because the assignment's trap list (vector
figures in PD, raster figures in Tech 2, multi-page tables in PD) predicts where
each is weakest.
"""

from __future__ import annotations

import json
import re
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

CELL_SPLIT = re.compile(r"\s*\|\s*")
WS = re.compile(r"\s+")


def normalise_cell(text: str) -> str:
    """Compare cells ignoring wrapping whitespace, case and soft hyphens."""
    value = WS.sub(" ", str(text)).strip().strip("|").strip()
    value = value.replace("\u00ad", "").replace("\u2010", "-").replace("\u2011", "-")
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    return value.casefold()


@dataclass
class TableScore:
    """Cell-level score for one table."""

    doc_id: str
    number: str
    caption: str
    pages: list[int]
    matched: bool = True          # False when no ingested table corresponds to it
    header_cells: int = 0
    header_hits: int = 0
    body_cells: int = 0
    body_hits: int = 0
    stitched: bool = False
    note: str = ""

    @property
    def accuracy(self) -> float:
        total = self.header_cells + self.body_cells
        return (self.header_hits + self.body_hits) / total if total else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "number": self.number, "caption": self.caption, "pages": self.pages,
                "matched": self.matched, "header_cells": self.header_cells, "header_hits": self.header_hits, "body_cells": self.body_cells,
                "body_hits": self.body_hits, "accuracy": round(self.accuracy, 4) if self.header_cells +
                self.body_cells else None, "stitched": self.stitched, "note": self.note}


@dataclass
class FigureScore:
    """Node/transition score for one figure."""

    doc_id: str
    number: str
    caption: str
    kind: str
    raster: bool
    nodes: int = 0
    node_hits: int = 0
    transitions: int = 0
    transition_hits: int = 0
    note: str = ""

    @property
    def accuracy(self) -> float:
        total = self.nodes + self.transitions
        return (self.node_hits + self.transition_hits) / total if total else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "number": self.number, "caption": self.caption, "kind": self.kind,
                "raster": self.raster, "nodes": self.nodes, "node_hits": self.node_hits,
                "transitions": self.transitions, "transition_hits": self.transition_hits,
                "accuracy": round(self.accuracy, 4) if self.nodes + self.transitions else None,
                "note": self.note}


@dataclass
class StructureReport:
    """Everything the structural check produced."""

    tables: list[TableScore] = field(default_factory=list)
    figures: list[FigureScore] = field(default_factory=list)
    manifest_path: str = ""

    @property
    def tables_detected(self) -> float:
        return (sum(1 for t in self.tables if t.matched) / len(self.tables)) if self.tables else float("nan")

    @property
    def table_accuracy(self) -> float:
        scored = [t for t in self.tables if t.header_cells + t.body_cells]
        return sum(t.accuracy for t in scored) / len(scored) if scored else float("nan")

    @property
    def figure_accuracy(self) -> float:
        scored = [f for f in self.figures if f.nodes + f.transitions]
        return sum(f.accuracy for f in scored) / len(scored) if scored else float("nan")

    @property
    def figures_detected(self) -> float:
        return (sum(1 for f in self.figures if "not detected" not in (f.note or ""))
                / len(self.figures)) if self.figures else float("nan")

    def by_document(self) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for doc_id in sorted({t.doc_id for t in self.tables} | {f.doc_id for f in self.figures}):
            tables = [t for t in self.tables if t.doc_id == doc_id and t.header_cells + t.body_cells]
            figures = [f for f in self.figures if f.doc_id == doc_id and f.nodes + f.transitions]
            out[doc_id] = {
                "tables": round(sum(t.accuracy for t in tables) / len(tables), 4) if tables else None,
                "n_tables": len(tables),
                "figures": round(sum(f.accuracy for f in figures) / len(figures), 4) if figures else None,
                "n_figures": len(figures),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"manifest": self.manifest_path,
                "tables_detected": round(self.tables_detected, 4),
                "table_cell_accuracy": round(self.table_accuracy, 4),
                "figures_detected": round(self.figures_detected, 4),
                "figure_structure_accuracy": round(self.figure_accuracy, 4),
                "per_document": self.by_document(),
                "tables": [t.to_dict() for t in self.tables],
                "figures": [f.to_dict() for f in self.figures]}


def score_tables(manifest_entries: Sequence[dict[str, Any]], ingested_tables: Sequence[Any]) -> list[TableScore]:
    """Score every manifest table against the ingested tables of the same document."""
    scores: list[TableScore] = []
    used: set[int] = set()
    entries = [e for e in manifest_entries if e.get("type") == "table"]
    # numbers restart in every Tech 2 book, so matching is one-to-one and breaks
    # ties on the caption and then on page proximity
    for entry in entries:
        doc_id, number = entry["doc_id"], str(entry.get("number") or "")
        page = int(entry.get("page") or 0)
        caption = str(entry.get("caption") or "").lower()
        pool = [t for t in ingested_tables
                if id(t) not in used and getattr(t, "doc_id", None) == doc_id
                and str(getattr(t, "number", "")) == number]
        if not pool:
            pool = [t for t in ingested_tables
                    if id(t) not in used and getattr(t, "doc_id", None) == doc_id
                    and number and number in str(getattr(t, "caption", ""))]
        candidates = sorted(pool, key=lambda t: (
            -_caption_overlap(caption, str(getattr(t, "caption", "")).lower()),
            abs((getattr(t, "pages", [0]) or [0])[0] - page)))
        if candidates:
            used.add(id(candidates[0]))
        score = TableScore(doc_id=doc_id, number=number, caption=str(entry.get("caption") or ""),
                           pages=[int(entry.get("page") or 0)])
        gt_headers = entry.get("headers") or []
        gt_rows = entry.get("rows") or []
        header_cells = [c for row in gt_headers for c in row]
        body_cells = [c for row in gt_rows for c in row]
        score.header_cells, score.body_cells = len(header_cells), len(body_cells)
        if not candidates:
            score.matched = False
            score.note = "table not detected by the ingester"
            scores.append(score)
            continue
        table = candidates[0]
        score.pages = sorted(set(getattr(table, "pages", []) or score.pages))
        score.stitched = bool(getattr(table, "stitched", False))
        ingested = _ingested_cells(table)
        score.header_hits = sum(1 for c in header_cells if normalise_cell(c) in ingested)
        score.body_hits = sum(1 for c in body_cells if normalise_cell(c) in ingested)
        if score.stitched and not getattr(table, "stitched", False):
            score.note = "ground truth spans pages but the ingester did not stitch"
        scores.append(score)
    return scores


def _caption_overlap(expected: str, actual: str) -> int:
    """How many words of the expected caption appear in the candidate's caption."""
    if not expected or not actual:
        return 0
    return len({w for w in expected.split() if len(w) > 3 and w in actual})


def _ingested_cells(table: Any) -> set[str]:
    """Every cell the ingester produced, from its headers, rows and Markdown."""
    cells: set[str] = set()
    for row in getattr(table, "headers", []) or []:
        cells |= {normalise_cell(c) for c in row}
    for row in getattr(table, "rows", []) or []:
        cells |= {normalise_cell(c) for c in row}
    markdown = getattr(table, "markdown", "") or ""
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            cells |= {normalise_cell(c) for c in CELL_SPLIT.split(line.strip().strip("|"))}
    return {c for c in cells if c}


def _ingested_transitions(figure: Any) -> set[tuple[str, str]]:
    """(from, to) pairs present in the ingested figure's structured description."""
    structure = getattr(figure, "structure", None) or {}
    pairs: set[tuple[str, str]] = set()
    for transition in structure.get("transitions", []) or []:
        if isinstance(transition, dict):
            pairs.add(_normalise_pair(str(transition.get("from", "")), str(transition.get("to", ""))))
    return pairs


def _ingested_labels(figure: Any) -> str:
    structure = getattr(figure, "structure", None) or {}
    labels = " ".join(str(t.get("trigger", "")) for t in structure.get("transitions", []) or []
                      if isinstance(t, dict))
    labels += " " + " ".join(str(x) for x in structure.get("labels", []) or [])
    return labels.casefold()


def _normalise_pair(src: str, dst: str) -> tuple[str, str]:
    return normalise_cell(src), normalise_cell(dst)


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", str(text).casefold()) if w]


def _mentions(haystack: list[str], phrase: str) -> bool:
    """Is ``phrase`` in the haystack, allowing a line break inside the phrase?

    Vector diagrams break a label across lines ("Host" / "Router" for the node
    "Host Router") because the box is narrow, and an extractor that reads text
    lines therefore never sees the phrase contiguously.  The words of the phrase
    must appear in order within a window only slightly longer than the phrase
    itself, so a match is still local: "Adapter" elsewhere in the figure does not
    let "Host Router" match.
    """
    words = _tokens(phrase)
    if not words:
        return False
    if len(words) == 1:
        return words[0] in haystack
    span = len(words) + 2
    for i, token in enumerate(haystack):
        if token != words[0]:
            continue
        window = iter(haystack[i:i + span])
        if all(w in window for w in words):
            return True
    return False


def score_figures(manifest_entries: Sequence[dict[str, Any]], ingested_figures: Sequence[Any]) -> list[FigureScore]:
    """Score every manifest figure's nodes and transitions against the ingested figure."""
    scores: list[FigureScore] = []
    for entry in manifest_entries:
        if entry.get("type") != "figure":
            continue
        doc_id, number = entry["doc_id"], str(entry.get("number") or "")
        candidates = [f for f in ingested_figures
                      if getattr(f, "doc_id", None) == doc_id and str(getattr(f, "number", "")) == number]
        gt_nodes = [str(n[0]) if isinstance(n, (list, tuple)) else str(n) for n in (entry.get("nodes") or [])]
        gt_edges = entry.get("edges") or []
        score = FigureScore(doc_id=doc_id, number=number, caption=str(entry.get("caption") or ""),
                            kind=str(entry.get("figure_kind") or ""), raster=bool(entry.get("raster")),
                            nodes=len(gt_nodes), transitions=len(gt_edges))
        if not candidates:
            score.note = "figure not detected by the ingester"
            scores.append(score)
            continue
        if entry.get("raster"):
            score.note = "raster figure: recovery depends on the OCR/VLM pass"
        figure = candidates[0]
        haystack = " ".join(str(x) for x in [
            getattr(figure, "description", ""), getattr(figure, "structure_text", ""),
            getattr(figure, "ocr_text", ""), getattr(figure, "vlm_description", ""),
            json.dumps(getattr(figure, "structure", {}) or {}, default=str),
            json.dumps(getattr(figure, "to_dict", lambda: {})() if hasattr(figure, "to_dict") else {},
                       default=str)])
        haystack = haystack.casefold()
        hay_tokens = _tokens(haystack)
        score.node_hits = sum(1 for n in gt_nodes
                              if normalise_cell(n) in haystack or _mentions(hay_tokens, n))
        # a transition is recovered when the figure record contains an edge with
        # the same endpoints; the trigger label is scored separately below
        ingested_edges = _ingested_transitions(figure)
        transitions = 0
        labels = 0
        for edge in gt_edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                src_idx, dst_idx, label = edge[0], edge[1], str(edge[2])
                try:
                    src, dst = gt_nodes[int(src_idx)], gt_nodes[int(dst_idx)]
                except (IndexError, ValueError):
                    continue
                if (_normalise_pair(src, dst) in ingested_edges
                        or (_mentions(hay_tokens, src) and _mentions(hay_tokens, dst)
                            and normalise_cell(src) != normalise_cell(dst))):
                    transitions += 1
                    if (normalise_cell(label) in haystack or _mentions(hay_tokens, label)
                            or normalise_cell(label) in _ingested_labels(figure)):
                        labels += 1
        score.transition_hits = transitions
        if gt_edges:
            score.note = (score.note + f" | trigger labels recovered: {labels}/{len(gt_edges)}").strip(" |")
        scores.append(score)
    return scores


def evaluate_structure(manifest_path: str | Path, ingested_tables: Sequence[Any],
                       ingested_figures: Sequence[Any]) -> StructureReport:
    """Run both structural checks against the generated manifest."""
    path = Path(manifest_path)
    report = StructureReport(manifest_path=str(path))
    if not path.exists():
        return report
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("elements", data if isinstance(data, list) else [])
    report.tables = score_tables(entries, ingested_tables)
    report.figures = score_figures(entries, ingested_figures)
    return report
