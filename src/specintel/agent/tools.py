"""WS4 — the agent's tools, each over one retrieval or graph surface.

Seven tools, matching the assignment's list:

===========================  =================================================
tool                         what it returns
===========================  =================================================
``search_text(query, k)``    chunk hits (text, table, figure description,
                             parent section) with provenance
``search_pages(query, k)``   page-image hits for the vision path
``graph_query(...)``         nodes, neighbours, paths or stats from the KG
``get_section(doc, n)``      the full text of one section, in order
``get_table(table_id)``      one table: headers, rows, Markdown, source pages
``get_figure(figure_id)``    one figure: caption, structured description, OCR
                             text and the image path
``resolve_entity(name)``     alias -> canonical node(s) with their provenance
===========================  =================================================

Every tool returns a :class:`ToolResult` whose ``hits``/``citations`` carry the
document, revision, section, page and element ids, so an answer built from a
tool call can always be cited back to the page it came from.  Tool calls are
logged in the project log format (``event=tool.call``) with latency and the
result size, which is what the evaluation's latency numbers are computed from.

A :class:`MockToolkit` implements the same interface with fixed fixtures, so the
agent could be developed and tested against the interface before WS2 and WS3
delivered (see ``INTERFACES.md``).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from specintel.common.logging import get_logger, log_event, log_exception
from specintel.common.models import Citation, SearchHit
from specintel.index.search import Filters

TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "search_text",
     "signature": "search_text(query: str, k: int = 8, docs: list[str] | None = None) -> list[SearchHit]",
     "returns": "chunk hits with doc_id, revision, section_number, page, element_ids, text, score",
     "when": "any keyword or phrase lookup over the text, table and figure-description indexes"},
    {"name": "search_pages",
     "signature": "search_pages(query: str, k: int = 5, docs: list[str] | None = None) -> list[dict]",
     "returns": "page-image hits with doc_id, page, patch boxes and image_path",
     "when": "diagrams, waveforms and any content whose layout matters more than its words"},
    {"name": "graph_query",
     "signature": ("graph_query(kind: str, name: str = '', node_id: str = '', src: str = '', dst: str = '', "
                   "edge_types: list[str] | None = None, depth: int = 1, direction: str = 'both') "
                   "-> GraphResult"),
     "returns": "nodes, edges and paths with per-edge provenance",
     "when": "multi-hop questions: which tests verify which timers, which spec defines which requirement"},
    {"name": "get_section",
     "signature": "get_section(doc_id: str, section_number: str) -> dict",
     "returns": "ordered elements of the section with their page numbers",
     "when": "the answer needs a whole section rather than a snippet"},
    {"name": "get_table",
     "signature": "get_table(table_id: str) -> dict",
     "returns": "headers, rows, Markdown, source pages, caption",
     "when": "a specific parameter, bit field or test-procedure table is named"},
    {"name": "get_figure",
     "signature": "get_figure(figure_id: str) -> dict",
     "returns": "caption, kind, structured description, OCR text, image path, source page",
     "when": "a state machine, timing diagram or block diagram is named"},
    {"name": "resolve_entity",
     "signature": "resolve_entity(name: str) -> list[Node]",
     "returns": "canonical nodes that match the name or one of its aliases, with provenance",
     "when": "the question uses a different spelling, an abbreviation or an old name"},
]

TOOL_NAMES = tuple(spec["name"] for spec in TOOL_SPECS)


@dataclass
class ToolResult:
    """Uniform tool output: hits plus an optional structured payload."""

    tool: str
    hits: list[SearchHit] = field(default_factory=list)
    text: str = ""
    table: dict[str, Any] | None = None
    figure: dict[str, Any] | None = None
    graph: dict[str, Any] | None = None
    entity: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def citations(self) -> list[Citation]:
        """Citations for the hits, in retrieval order."""
        return [citation_from_hit(hit) for hit in self.hits]

    def top_score(self) -> float:
        return max((h.score for h in self.hits), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "error": self.error, "latency_s": round(self.latency_s, 4),
                "n_hits": len(self.hits), "text": self.text[:2000], "table": self.table, "figure": self.figure,
                "graph": self.graph, "entity": self.entity, "meta": self.meta,
                "citations": [c.to_marker() for c in self.citations()]}


def _number_from_id(value: str | None, prefix: str) -> str:
    """``pd:table:6.8`` -> ``6.8``; tolerates a bare number too.

    Internal ids are not citations: a page-level hit carries ``2:tbl_f5c576c19099``
    and ``Table tbl_f5c576c19099`` is meaningless to a reader, so an id that is an
    artefact of this pipeline is rejected and the caption inside the hit is used
    instead.
    """
    if not value:
        return ""
    tail = str(value).split(":")[-1]
    if tail in ("", "table", "figure") or tail.startswith(("tbl_", "fig_", "c_")):
        return ""
    return tail


CAPTION_NUMBER_RE = re.compile(r"\b(Table|Figure)\s+([A-Za-z0-9][\w.\-]*)")


def _number_from_text(text: str, kind: str) -> str:
    """The number the evidence itself prints: ``Table 3-4. TD 4.2.4 …``."""
    for match in CAPTION_NUMBER_RE.finditer(str(text)[:400]):
        if match.group(1).lower() == kind:
            return match.group(2).strip(".")
    return ""


def citation_from_hit(hit: SearchHit) -> Citation:
    """Build the assignment's citation marker from a hit's provenance."""
    if hit.table_id:
        number = (str(hit.meta.get("table_number") or "").strip()
                  or _number_from_id(hit.table_id, "table")
                  or _number_from_text(hit.text, "table"))
        region = f"Table {number}".strip()
    elif hit.figure_id or hit.meta.get("figure_number"):
        # the number may arrive in meta without an id (synthetic evidence hits built
        # from the artefact payload) -- the marker must still say "Figure 4-1"
        number = (str(hit.meta.get("figure_number") or "").strip()
                  or _number_from_id(hit.figure_id, "figure")
                  or _number_from_text(hit.text, "figure"))
        region = f"Figure {number}".strip()
    elif hit.element_type in ("table", "table_caption"):
        region = "Table"
    elif hit.element_type in ("figure", "figure_description", "figure_caption"):
        region = "Figure"
    elif hit.element_type == "page_image":
        region = "page image"
    else:
        region = "prose"
    return Citation(doc_id=hit.doc_id, revision=str(hit.meta.get("revision", "")),
                    section_number=hit.section_number, page=hit.page, page_end=hit.page_end,
                    region=region, element_ids=list(hit.element_ids), quote=hit.text[:240])


class Toolkit:
    """The seven tools over the persisted WS2 index and WS3 graph."""

    def __init__(self, index: Any, *, store: Any | None = None, elements: Sequence[Any] = (),
                 tables: Sequence[Any] = (), figures: Sequence[Any] = (), config: Any | None = None) -> None:
        self.index = index
        self.store = store
        self.elements = list(elements)
        self.tables = list(tables)
        self.figures = list(figures)
        self.config = config
        self.log = get_logger("ws4.tools", config)
        self._by_doc_section: dict[tuple[str, str], list[Any]] = {}
        for element in self.elements:
            if element.section_number:
                self._by_doc_section.setdefault((element.doc_id, element.section_number), []).append(element)
        self._tables_by_id = {self._table_key(t): t for t in self.tables}
        self._tables_by_id.update({t.number: t for t in self.tables if t.number})
        self._figures_by_id = {self._figure_key(f): f for f in self.figures}
        self._figures_by_id.update({f.number: f for f in self.figures if f.number})

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Any) -> "Toolkit":
        """Load everything from ``data/``; raises FileNotFoundError with a hint when absent."""
        from specintel.index.build import load_index
        from specintel.kg.build import load_graph
        from specintel.ingest.pipeline import load_elements, load_figures, load_tables

        try:
            index = load_index(config)
        except FileNotFoundError as exc:                       # pragma: no cover - guidance path
            raise FileNotFoundError(f"{exc}\nBuild the index first: python scripts/build_index.py") from exc
        store = None
        try:
            store = load_graph(config)
        except Exception:                                       # the graph is optional for text questions
            store = None
        elements = load_elements(config)
        tables: list[Any] = []
        figures: list[Any] = []
        for doc_id in sorted({e.doc_id for e in elements}):
            tables.extend(load_tables(config, doc_id))
            figures.extend(load_figures(config, doc_id))
        return cls(index, store=store, elements=elements, tables=tables, figures=figures, config=config)

    @staticmethod
    def _table_key(table: Any) -> str:
        return f"{table.doc_id}:table:{table.number or table.table_id}"

    @staticmethod
    def _figure_key(figure: Any) -> str:
        return f"{figure.doc_id}:figure:{figure.number or figure.figure_id}"

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        return {"search_text": self.search_text, "search_pages": self.search_pages,
                "graph_query": self.graph_query, "get_section": self.get_section, "get_table": self.get_table,
                "get_figure": self.get_figure, "resolve_entity": self.resolve_entity}

    def describe(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Dispatch a tool call with logging, timing and error capture.

        The first parameter is named ``tool_name`` (not ``name``) because two
        tools legitimately take a ``name`` keyword — see the note in
        ``INTERFACES.md`` about keeping the dispatcher's own parameters out of the
        tools' namespaces.
        """
        tools = self.registry()
        if tool_name not in tools:
            raise KeyError(f"unknown tool {tool_name!r}; available: {sorted(tools)}")
        started = time.perf_counter()
        try:
            result = tools[tool_name](**kwargs)
        except Exception as exc:                                # noqa: BLE001 - reported to the agent
            log_exception(self.log, "tool.error", exc, tool=tool_name)
            return ToolResult(tool=tool_name, error=f"{type(exc).__name__}: {exc}",
                              latency_s=time.perf_counter() - started)
        result.latency_s = time.perf_counter() - started
        log_event(self.log, "tool.call", tool=tool_name, n_hits=len(result.hits), ok=result.ok,
                  top_score=round(result.top_score(), 4) if result.hits else None,
                  duration_s=round(result.latency_s, 4))
        return result

    # ------------------------------------------------------------------ #
    # retrieval tools
    # ------------------------------------------------------------------ #
    def search_text(self, query: str, k: int = 8, docs: Sequence[str] | None = None,
                    element_types: Sequence[str] | None = None, sections: Sequence[str] | None = None,
                    tables_only: bool = False, figures_only: bool = False, revisions: Sequence[str] | None = None,
                    hybrid: bool | None = None, rerank: bool = True) -> ToolResult:
        filters = Filters(docs=tuple(docs or ()), revisions=tuple(revisions or ()),
                          element_types=tuple(element_types or ()), sections=tuple(sections or ()),
                          tables_only=tables_only, figures_only=figures_only)
        kwargs: dict[str, Any] = {"filters": filters, "rerank": rerank}
        if hybrid is not None:
            kwargs["hybrid"] = hybrid
        hits = self.index.search_text(query, k=k, **kwargs)
        return ToolResult(tool="search_text", hits=list(hits), text=_render_hits(hits),
                          meta={"query": query, "k": k, "filters": filters.to_dict() if hasattr(filters, "to_dict")
                                else {"docs": list(docs or [])}})

    def search_pages(self, query: str, k: int = 5, docs: Sequence[str] | None = None,
                     fuse: bool = False) -> ToolResult:
        """The vision path.

        ``fuse=True`` runs the text+page fusion (:meth:`SearchIndex.search_fused`)
        and returns chunk hits where possible; the default returns raw page hits
        so the caller can cite the page and the image regions directly.
        """
        if fuse:
            hits = self.index.search_fused(query, k=k)[:k]
            return ToolResult(tool="search_pages", hits=list(hits), text=_render_hits(hits),
                              meta={"query": query, "k": k, "fused": True})
        raw = self.index.search_pages(query, k=k, docs=list(docs) if docs else None)
        hits = [_page_hit(item) for item in raw]
        text = "\n".join(f"[{i + 1}] {h.locator} score={h.score:.3f} regions={len(h.element_ids)}"
                         for i, h in enumerate(hits))
        return ToolResult(tool="search_pages", hits=hits, text=text,
                          meta={"query": query, "k": k, "page_patches": len(raw)})

    # ------------------------------------------------------------------ #
    # graph tools
    # ------------------------------------------------------------------ #
    def graph_query(self, kind: str = "neighbors", name: str = "", node_id: str = "", src: str = "",
                    dst: str = "", edge_types: Sequence[str] | None = None, depth: int = 1,
                    direction: str = "both", limit: int = 25) -> ToolResult:
        """Graph lookup / neighbours / path / subgraph / stats.

        ``kind='path'`` takes ``src`` and ``dst``; the other kinds take ``name`` or
        ``node_id``.  Unknown entities come back as an empty result with a note
        rather than an exception, so the agent can report "the graph has no such
        entity" as an honest decline.
        """
        if self.store is None:
            return ToolResult(tool="graph_query", error="no knowledge graph built (run scripts/build_graph.py)",
                              meta={"kind": kind, "name": name})
        query: dict[str, Any] = {"kind": kind, "name": name, "node_id": node_id, "src": src, "dst": dst,
                                 "edge_types": list(edge_types) if edge_types else None, "depth": depth,
                                 "direction": direction, "limit": limit}
        from specintel.kg.build import graph_query as run_graph_query

        payload = run_graph_query(self.config, query, store=self.store) if self.config else \
            self.store.query(query)
        stale_note = ""
        if os.environ.get("SPECINTEL_ABLATE_STALE"):
            # ablation: stop flagging superseded cross-document pointers, so the
            # report can show how many answers would follow a stale reference
            payload = _without_stale_flags(payload)
            stale_note = " (stale-reference flags disabled by SPECINTEL_ABLATE_STALE)"
        text = _render_graph(payload)
        return ToolResult(tool="graph_query", graph=payload, text=text + stale_note,
                          hits=self._graph_hits(payload), meta={"kind": kind, "name": name})

    def _graph_hits(self, payload: dict[str, Any]) -> list[SearchHit]:
        """Turn graph edges into citable hits so graph answers carry page provenance.

        The text is a sentence ("tSenderResponse is defined in Table 6.8"), not an
        arrow diagram: graph hits are evidence the answer is written from, and
        "A -[REL]-> B" is not a sentence a reader or a composer can use.
        """
        names = {str(node.get("node_id")): str(node.get("canonical_name") or node.get("name") or "")
                 for node in payload.get("nodes", [])}
        hits: list[SearchHit] = []
        for i, edge in enumerate(payload.get("edges", [])[:12]):
            evidence = edge.get("evidence") or _edge_sentence(edge, names)
            hits.append(SearchHit(
                chunk_id=f"edge:{edge.get('edge_id', i)}", doc_id=str(edge.get("doc_id", "")),
                section_number=str(edge.get("section_number", "")),
                page=int(edge.get("page") or 0), element_type="graph_edge", source="graph", text=str(evidence),
                score=float(edge.get("confidence") or 0.7),
                element_ids=[edge["element_id"]] if edge.get("element_id") else [],
                meta={"edge_type": edge.get("edge_type"), "src": edge.get("src_id"), "dst": edge.get("dst_id"),
                      "stale": edge.get("stale"), "method": edge.get("extraction_method"),
                      "revision": edge.get("revision", "")}))
        return hits

    def resolve_entity(self, name: str, limit: int = 10) -> ToolResult:
        if self.store is None:
            return ToolResult(tool="resolve_entity", error="no knowledge graph built", meta={"name": name})
        nodes = self.store.lookup(name, limit=limit)
        payload = {"query": name, "nodes": [n.to_dict() if hasattr(n, "to_dict") else n for n in nodes]}
        text = "\n".join(f"{n.get('node_type')}: {n.get('canonical_name')} ({n.get('doc_id')} "
                         f"§{n.get('section_number')} p.{n.get('page')})" for n in payload["nodes"]) or \
            f"no node matches {name!r}"
        return ToolResult(tool="resolve_entity", entity=payload, text=text, meta={"name": name})

    # ------------------------------------------------------------------ #
    # element tools
    # ------------------------------------------------------------------ #
    def get_section(self, doc_id: str, section_number: str) -> ToolResult:
        elements = sorted(self._by_doc_section.get((doc_id, section_number), []),
                          key=lambda e: (e.page, e.reading_order if hasattr(e, "reading_order") else 0))
        if not elements:
            return ToolResult(tool="get_section", error=f"section {doc_id} §{section_number} not found",
                              meta={"doc_id": doc_id, "section_number": section_number})
        texts = [f"{e.element_type}: {e.text}" for e in elements if getattr(e, "text", "")]
        hits = [_element_hit(e) for e in elements if getattr(e, "text", "")]
        payload = {"doc_id": doc_id, "section_number": section_number, "n_elements": len(elements),
                   "pages": sorted({e.page for e in elements}),
                   "title": next((e.section_title for e in elements if e.element_type == "section_heading"), ""),
                   "elements": [{"element_id": e.element_id, "type": e.element_type, "page": e.page,
                                 "text": e.text} for e in elements]}
        return ToolResult(tool="get_section", hits=hits, text="\n".join(texts)[:6000],
                          meta={"doc_id": doc_id, "section_number": section_number, "payload": payload})

    def get_table(self, table_id: str = "", *, doc_id: str = "", number: str = "") -> ToolResult:
        table = self._find_table(table_id, doc_id, number)
        if table is None:
            return ToolResult(tool="get_table", error=f"table {table_id or number!r} not found",
                              meta={"table_id": table_id})
        payload = {"table_id": table.table_id, "doc_id": table.doc_id, "number": table.number,
                   "caption": table.caption, "section_number": table.section_number, "pages": table.pages,
                   "revision": getattr(table, "revision", ""),
                   "headers": table.headers, "rows": table.rows, "markdown": table.markdown,
                   "stitched": getattr(table, "stitched", False),
                   "cell_provenance": getattr(table, "cell_provenance", None)}
        hits = [_element_hit_for_table(table)]
        return ToolResult(tool="get_table", hits=hits, table=payload, text=table.markdown[:6000],
                          meta={"table_id": table.table_id, "pages": table.pages})

    def get_figure(self, figure_id: str = "", *, doc_id: str = "", number: str = "") -> ToolResult:
        figure = self._find_figure(figure_id, doc_id, number)
        if figure is None:
            return ToolResult(tool="get_figure", error=f"figure {figure_id or number!r} not found",
                              meta={"figure_id": figure_id})
        payload = {"figure_id": figure.figure_id, "doc_id": figure.doc_id, "number": figure.number,
                   "caption": figure.caption, "kind": figure.kind, "page": figure.page,
                   "section_number": figure.section_number, "description": figure.description,
                   "revision": getattr(figure, "revision", ""),
                   "ocr_text": getattr(figure, "ocr_text", ""), "image_path": figure.image_path,
                   "structure": getattr(figure, "structure", {}), "labels": getattr(figure, "labels", [])}
        text = "\n".join(x for x in [payload["caption"], payload["description"], payload["ocr_text"]] if x)
        hits = [_element_hit_for_figure(figure)]
        return ToolResult(tool="get_figure", hits=hits, figure=payload, text=text[:6000],
                          meta={"figure_id": figure.figure_id, "page": figure.page})

    # ------------------------------------------------------------------ #
    # lookups
    # ------------------------------------------------------------------ #
    def _find_table(self, table_id: str, doc_id: str, number: str) -> Any | None:
        if table_id:
            for key in (table_id, f"{doc_id}:table:{table_id}" if doc_id else "", table_id.split(":")[-1]):
                if key and key in self._tables_by_id:
                    return self._tables_by_id[key]
        for table in self.tables:
            if number and table.number == number and (not doc_id or table.doc_id == doc_id):
                return table
            if table_id and table.table_id == table_id:
                return table
        return None

    def _find_figure(self, figure_id: str, doc_id: str, number: str) -> Any | None:
        if figure_id:
            for key in (figure_id, f"{doc_id}:figure:{figure_id}" if doc_id else "", figure_id.split(":")[-1]):
                if key and key in self._figures_by_id:
                    return self._figures_by_id[key]
        for figure in self.figures:
            if number and figure.number == number and (not doc_id or figure.doc_id == doc_id):
                return figure
            if figure_id and figure.figure_id == figure_id:
                return figure
        return None


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _page_hit(item: dict[str, Any]) -> SearchHit:
    """Convert a page-image retrieval result into a citable hit."""
    return SearchHit(chunk_id=f"{item.get('doc_id')}:p{item.get('page')}:patch{item.get('patch', '')}",
                     doc_id=str(item.get("doc_id", "")),
                     section_number=", ".join(item.get("sections") or [])[:40],
                     page=int(item.get("page") or 0), element_type="page_image", source="pages",
                     text=str(item.get("text") or ""), score=float(item.get("score") or 0.0),
                     element_ids=list(item.get("element_ids") or []),
                     meta={"image_path": item.get("image_path"), "patch": item.get("patch"),
                           "boxes": item.get("boxes"), "revision": item.get("revision", "")})


def _element_hit(element: Any) -> SearchHit:
    return SearchHit(chunk_id=element.element_id, doc_id=element.doc_id,
                     section_number=element.section_number, page=element.page,
                     element_type=element.element_type, text=getattr(element, "text", ""), score=1.0,
                     element_ids=[element.element_id], table_id=getattr(element, "table_id", None),
                     figure_id=getattr(element, "figure_id", None),
                     meta={"revision": getattr(element, "revision", ""), "section_title": element.section_title})


def _element_hit_for_table(table: Any) -> SearchHit:
    return SearchHit(chunk_id=table.table_id, doc_id=table.doc_id,
                     section_number=table.section_number, page=table.pages[0] if table.pages else 0,
                     page_end=table.pages[-1] if table.pages else None, element_type="table",
                     text=table.markdown, score=1.0, element_ids=list(table.element_ids),
                     table_id=table.table_id,
                     meta={"table_number": table.number, "revision": getattr(table, "revision", ""),
                           "pages": table.pages, "stitched": getattr(table, "stitched", False)})


def _element_hit_for_figure(figure: Any) -> SearchHit:
    return SearchHit(chunk_id=figure.figure_id, doc_id=figure.doc_id,
                     section_number=figure.section_number, page=figure.page, element_type="figure",
                     text="\n".join(x for x in [figure.caption, figure.description,
                                                getattr(figure, "ocr_text", "")] if x),
                     score=1.0, element_ids=[figure.element_id] if getattr(figure, "element_id", None) else [],
                     figure_id=figure.figure_id,
                     meta={"figure_number": figure.number, "revision": getattr(figure, "revision", ""),
                           "kind": figure.kind})


RELATION_PHRASES: dict[str, str] = {
    "DEFINED_IN": "is defined in", "REFERENCES": "references", "SENT_BY": "is sent by",
    "RESPONDS_TO": "responds to", "CONTAINS": "contains", "TRANSITIONS_TO": "transitions to",
    "STARTS_TIMER": "starts", "STOPS_TIMER": "stops", "HAS_FIELD": "has field", "HAS_VALUE": "has value",
    "TESTS": "is tested by", "VERIFIES": "verifies", "SUPERSEDES": "supersedes", "TUNNELS": "tunnels",
    "HAS_STATE": "contains the state", "HAS_REVISION": "has revision", "PART_OF": "is part of",
}


def _human_node(node_id: str) -> str:
    """``Table:pd_table_6_8`` -> ``Table 6.8``, ``Timer:tsenderresponse`` -> ``tSenderResponse``."""
    prefix, _, slug = str(node_id).partition(":")
    slug = slug or prefix
    text = slug.replace("_", " ").strip()
    if prefix in ("Table", "Figure") and text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    return f"{prefix} {text}".strip() if prefix in ("Table", "Figure") else text


def _edge_sentence(edge: dict[str, Any], names: dict[str, str]) -> str:
    """Readable sentence for one graph relation."""
    src = names.get(str(edge.get("src_id"))) or _human_node(str(edge.get("src_id", "")))
    dst = names.get(str(edge.get("dst_id"))) or _human_node(str(edge.get("dst_id", "")))
    phrase = RELATION_PHRASES.get(str(edge.get("edge_type")), "relates to")
    suffix = " (reference to an older revision)" if edge.get("stale") else ""
    return f"{src} {phrase} {dst}{suffix}"


def _render_hits(hits: Iterable[SearchHit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit.locator} ({hit.element_type}, score {hit.score:.3f})\n{hit.text[:600]}")
    return "\n\n".join(lines)


def _without_stale_flags(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy of a graph payload with every ``stale`` flag cleared (ablation)."""
    clone = json.loads(json.dumps(payload, default=str))
    for edge in clone.get("edges", []) or []:
        if isinstance(edge, dict):
            edge["stale"] = False
    return clone


def _render_graph(payload: dict[str, Any]) -> str:
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    out = [f"{len(nodes)} node(s), {len(edges)} edge(s)"]
    for node in nodes[:12]:
        out.append(f"  {node.get('node_type')}: {node.get('canonical_name') or node.get('name')} "
                   f"({node.get('doc_id')} §{node.get('section_number')} p.{node.get('page')})")
    for edge in edges[:16]:
        out.append(f"  {edge.get('src_id')} -[{edge.get('edge_type')}]-> {edge.get('dst_id')} "
                   f"({edge.get('doc_id')} p.{edge.get('page')}"
                   f"{', stale' if edge.get('stale') else ''})")
    if payload.get("paths"):
        for path in payload["paths"][:5]:
            out.append("  path: " + " -> ".join(str(step) for step in path))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# mock toolkit (interface-first development)
# --------------------------------------------------------------------------- #
@dataclass
class MockToolkit:
    """Deterministic tool stub used by the agent tests and by interface-first work.

    It reads ``tests/fixtures/tools.json`` when present, otherwise it answers from
    the small fixtures embedded here.  The point is that the agent, the CLI and
    the evaluation can be exercised end-to-end before the real index and graph
    exist, which is what the assignment means by "contracts before code".
    """

    fixtures_path: str | Path | None = None
    config: Any = None
    fixtures: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fixtures_path and Path(self.fixtures_path).exists():
            self.fixtures = json.loads(Path(self.fixtures_path).read_text(encoding="utf-8"))
        else:
            self.fixtures = DEFAULT_FIXTURES
        self.log = get_logger("ws4.tools.mock", self.config)

    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        return {name: (lambda **kwargs: self.call(name, **kwargs)) for name in TOOL_NAMES}

    def describe(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        name = tool_name
        self.calls.append((name, kwargs))
        spec = self.fixtures.get(name, {})
        if name == "search_text":
            hits = [SearchHit(**hit) for hit in spec.get("hits", [])][: int(kwargs.get("k", 8))]
            return ToolResult(tool=name, hits=hits, text=_render_hits(hits), latency_s=0.0)
        if name == "search_pages":
            hits = [SearchHit(**hit) for hit in spec.get("pages", [])][: int(kwargs.get("k", 5))]
            return ToolResult(tool=name, hits=hits, text="\n".join(h.locator for h in hits))
        if name == "graph_query":
            payload = spec.get("graph", {"nodes": [], "edges": []})
            return ToolResult(tool=name, graph=payload, text=_render_graph(payload))
        if name == "get_table":
            return ToolResult(tool=name, table=spec.get("table"), text=(spec.get("table") or {}).get("markdown", ""))
        if name == "get_figure":
            figure = spec.get("figure") or {}
            return ToolResult(tool=name, figure=figure,
                              text=str(figure.get("text") or figure.get("description") or ""))
        if name == "get_section":
            return ToolResult(tool=name, text=spec.get("section_text", ""))
        if name == "resolve_entity":
            return ToolResult(tool=name, entity=spec.get("entity", {}), text=json.dumps(spec.get("entity", {})))
        return ToolResult(tool=name, error="mock has no fixture for this tool")


DEFAULT_FIXTURES: dict[str, Any] = {
    "search_text": {"hits": [
        {"chunk_id": "pd:table:6.8:row:tSenderResponse", "doc_id": "pd", "meta": {"revision": "R3.2 V1.2"},
         "section_number": "6.6", "page": 3, "element_type": "table", "score": 3.1, "table_id": "pd:t6.8",
         "element_ids": ["pd:table:6.8:v3"], "text": "tSenderResponse | Min 24 ms | Max 30 ms | Table 6.8",
         "meta": {"revision": "R3.2 V1.2", "table_number": "6.8"}},
        {"chunk_id": "pd:6.6.2:para", "doc_id": "pd", "meta": {"revision": "R3.2 V1.2"}, "section_number": "6.6.2",
         "page": 3, "element_type": "paragraph", "score": 1.2,
         "element_ids": ["pd:paragraph:3:64"],
         "text": "A Sender shall respond to a Request within tSenderResponse.", "meta": {"revision": "R3.2 V1.2"}},
    ]},
    "graph_query": {"graph": {"nodes": [{"node_id": "pd:t6.8", "node_type": "Table", "name": "Table 6.8"}],
                              "edges": [{"edge_id": "e1", "src_id": "pd:timer:tSenderResponse",
                                         "dst_id": "pd:t6.8", "edge_type": "DEFINED_IN", "doc_id": "pd",
                                         "page": 3, "confidence": 1.0, "element_id": "pd:table:6.8:v3"}]}},
    "get_section": {
        "section_text": ("6.6 Timers and Counters | Timer definitions and limits for the protocol "
                         "layer. tSenderResponse: 24 ms minimum, 30 ms maximum. "
                         "tPSTransition: 450 ms minimum, 550 ms maximum."),
    },
    "search_pages": {"pages": [
        {"chunk_id": "pd:page:3", "doc_id": "pd", "section_number": "6.6",
         "page": 3, "element_type": "page_image", "score": 0.81, "source": "pages",
         "text": "Timers and Counters — Table 6.8 (page image)",
         "meta": {"revision": "R3.2 V1.2", "image_path": "pd_p0003.png"}},
    ]},
    "get_figure": {"figure": {
        "figure_id": "pd:fig:8-1", "doc_id": "pd", "number": "8-1", "page": 6,
        "caption": "Source policy engine states", "kind": "state_machine",
        "description": "Source policy engine: PE_SRC_Startup to PE_SRC_Send_Capabilities.",
        "structure": {"states": ["PE_SRC_Startup", "PE_SRC_Send_Capabilities"],
                      "transitions": [{"from": "PE_SRC_Startup", "to": "PE_SRC_Send_Capabilities",
                                       "trigger": "VBUS valid"}]},
        "text": "Source policy engine states (Figure 8-1, pd p.6)",
    }},
    "resolve_entity": {"entity": {
        "name": "tSenderResponse", "query": "tSenderResponse",
        "nodes": [{"node_id": "Timer:tsenderresponse", "node_type": "Timer", "name": "tSenderResponse",
                   "doc_id": "pd", "page": 3, "section_number": "6.6", "revision": "R3.2 V1.2",
                   "definition": ""}],
    }},
    "get_table": {"table": {"table_id": "pd:t6.8", "doc_id": "pd", "number": "6.8", "pages": [3, 4],
                            "caption": "Timer definitions and limits", "section_number": "6.6",
                            "markdown": "| Timer | Min | Max | Unit |\n|---|---|---|---|\n"
                                        "| tSenderResponse | 24 | 30 | ms |"}},
}

def write_mock_fixtures(path: str | Path, *, fixtures: dict[str, Any] | None = None) -> Path:
    """Write the default fixtures so a teammate can stub the tools immediately."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fixtures or DEFAULT_FIXTURES, indent=2), encoding="utf-8")
    return target


__all__ = ["ToolResult", "Toolkit", "MockToolkit", "TOOL_SPECS", "TOOL_NAMES", "citation_from_hit",
           "write_mock_fixtures"]
