"""WS2 — structure-aware chunking with parent–child links.

Rules (Task 2, Step 5):

* a section becomes a *parent* chunk carrying the heading and the whole text, so
  a small matching child can return its parent section for the answer;
* paragraphs are packed into *child* chunks of ``target_tokens`` with a small
  overlap, never crossing a section boundary;
* tables, figure descriptions and captions are **atomic**: a bit-field table is
  useless if half its rows are in another chunk, so they are never split and
  always keep their number, caption and page range;
* the heading path is prepended to every chunk's embedding text so an isolated
  paragraph still carries "USB PD §6.6 timers" context;
* exact identifiers found in the text are recorded per chunk, which is what lets
  the identifier-aware BM25 boost and the graph tool line up.

The chunk schema is the contract in INTERFACES.md; ``meta`` carries whatever the
producer knows and the consumers may use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.common.models import Chunk, Element, Figure, Table, new_id
from specintel.index.bm25 import identifiers_in

TOKEN_ESTIMATE = 4.0        # characters per token for English technical prose
ATOMIC_TYPES = {"table", "figure", "figure_description"}
CAPTION_TYPES = {"table_caption", "figure_caption"}


@dataclass
class ChunkOptions:
    target_tokens: int = 320
    overlap_tokens: int = 60
    min_tokens: int = 24
    include_headings_in_text: bool = True
    parent_sections: bool = True


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: characters/4 with a floor of the word count."""
    return max(len(text) // int(TOKEN_ESTIMATE), len(text.split()))


def chunk_elements(elements: Sequence[Element], *, tables: Sequence[Table] = (),
                   figures: Sequence[Figure] = (), options: ChunkOptions | None = None) -> list[Chunk]:
    """Turn one document's element stream into chunks with provenance."""
    opts = options or ChunkOptions()
    tables_by_id = {t.table_id: t for t in tables}
    figures_by_id = {f.figure_id: f for f in figures}

    chunks: list[Chunk] = []
    current_section_key: tuple[str, str, str] | None = None
    section_parent: Chunk | None = None
    buffer: list[Element] = []

    def flush_children() -> None:
        nonlocal buffer
        if not buffer:
            return
        groups = _pack(buffer, opts)
        for group in groups:
            chunks.append(_child_chunk(group, opts, section_parent))
        buffer = []

    for element in elements:
        if element.element_type in ("header", "footer", "toc"):
            continue
        section_key = (element.doc_id, element.section_number, element.section_title)
        if section_key != current_section_key:
            flush_children()
            if section_parent is not None:
                section_parent.child_chunk_ids = [c.chunk_id for c in chunks
                                                  if c.parent_chunk_id == section_parent.chunk_id]
            section_parent = None
            current_section_key = section_key

        if element.element_type == "section_heading":
            if opts.parent_sections:
                section_parent = _parent_chunk(element, opts)
                chunks.append(section_parent)
            continue

        if element.element_type in ATOMIC_TYPES or element.table_id or element.figure_id:
            flush_children()
            chunks.append(_atomic_chunk(element, opts, section_parent, tables_by_id, figures_by_id))
            continue

        if element.element_type in CAPTION_TYPES:
            flush_children()
            chunks.append(_caption_chunk(element, opts, section_parent))
            continue

        buffer.append(element)

    flush_children()
    if section_parent is not None:
        section_parent.child_chunk_ids = [c.chunk_id for c in chunks if c.parent_chunk_id == section_parent.chunk_id]

    # a parent whose section had no body still needs its own retrievable text
    for chunk in chunks:
        if chunk.parent_chunk_id is None and not chunk.text.strip():
            chunk.text = chunk.section_title
    return chunks


# --------------------------------------------------------------------------- #
# chunk builders
# --------------------------------------------------------------------------- #
def _heading_prefix(element: Element) -> str:
    parts = [p for p in (element.section_path or []) if p]
    if not parts and element.section_number:
        parts = [f"{element.section_number} {element.section_title}".strip()]
    return " > ".join(parts)


def _parent_chunk(element: Element, opts: ChunkOptions) -> Chunk:
    return Chunk(
        chunk_id=f"{element.doc_id}:sec:{element.section_number or new_id('s')}",
        doc_id=element.doc_id, revision=element.revision, element_type="section",
        text=element.text, page=element.page, page_end=element.page,
        section_number=element.section_number, section_title=element.section_title,
        section_path=list(element.section_path), element_ids=[element.element_id],
        bbox=element.bbox, n_tokens=estimate_tokens(element.text),
        meta={"kind": "parent_section", "level": element.meta.get("level")},
    )


def _pack(buffer: Sequence[Element], opts: ChunkOptions) -> list[list[Element]]:
    """Greedily pack elements into groups of about ``target_tokens``."""
    groups: list[list[Element]] = []
    current: list[Element] = []
    size = 0
    for element in buffer:
        tokens = estimate_tokens(element.text) or 1
        # an atomic-ish element (long note) larger than the target gets its own group
        if tokens >= opts.target_tokens:
            if current:
                groups.append(current)
                current, size = [], 0
            groups.append([element])
            continue
        if size + tokens > opts.target_tokens and current:
            groups.append(current)
            # carry a tail of the previous group for overlap
            tail: list[Element] = []
            tail_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev.text)
                if tail_tokens + prev_tokens > opts.overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_tokens
            current, size = list(tail), tail_tokens
        current.append(element)
        size += tokens
    if current:
        groups.append(current)
    return groups


def _child_chunk(group: Sequence[Element], opts: ChunkOptions, parent: Chunk | None) -> Chunk:
    first, last = group[0], group[-1]
    body = "\n".join(el.text for el in group if el.text.strip())
    prefix = _heading_prefix(first) if opts.include_headings_in_text else ""
    text = f"{prefix}\n{body}".strip() if prefix else body
    ids = identifiers_in(text)
    return Chunk(
        chunk_id=f"{first.doc_id}:chunk:{new_id('c')}",
        doc_id=first.doc_id, revision=first.revision, element_type=_dominant_type(group),
        text=text, page=first.page, page_end=max(el.page for el in group),
        section_number=first.section_number, section_title=first.section_title,
        section_path=list(first.section_path), element_ids=[el.element_id for el in group],
        parent_chunk_id=parent.chunk_id if parent else None,
        bbox=first.bbox, n_tokens=estimate_tokens(text),
        meta={"kind": "text", "identifiers": sorted(ids), "n_elements": len(group)},
    )


def _atomic_chunk(element: Element, opts: ChunkOptions, parent: Chunk | None,
                  tables: dict[str, Table], figures: dict[str, Figure]) -> Chunk:
    table = tables.get(element.table_id or "") if element.table_id else None
    figure = figures.get(element.figure_id or "") if element.figure_id else None
    if table is not None:
        text = f"{_heading_prefix(element)}\nTable {table.number}. {table.caption}\n{table.markdown}"
        element_type = "table"
    elif figure is not None:
        text = f"{_heading_prefix(element)}\nFigure {figure.number}. {figure.caption}\n{figure.description}"
        element_type = "figure_description"
    else:
        text = element.text
        element_type = element.element_type
    pages = [element.page] + ([p for p in element.meta.get("pages", [])] if element.meta else [])
    identifiers = identifiers_in(text)
    return Chunk(
        chunk_id=f"{element.doc_id}:{element_type}:{new_id('c')}",
        doc_id=element.doc_id, revision=element.revision, element_type=element_type,
        text=text, page=min(pages), page_end=max(pages),
        section_number=element.section_number, section_title=element.section_title,
        section_path=list(element.section_path), element_ids=[element.element_id],
        parent_chunk_id=parent.chunk_id if parent else None,
        table_id=element.table_id, figure_id=element.figure_id, bbox=element.bbox,
        n_tokens=estimate_tokens(text),
        meta={"kind": element_type, "identifiers": sorted(identifiers),
              "table_number": getattr(table, "number", None), "figure_number": getattr(figure, "number", None),
              "image_path": getattr(figure, "image_path", None)},
    )


def _caption_chunk(element: Element, opts: ChunkOptions, parent: Chunk | None) -> Chunk:
    kind = "table" if element.element_type == "table_caption" else "figure"
    return Chunk(
        chunk_id=f"{element.doc_id}:caption:{new_id('c')}", doc_id=element.doc_id,
        revision=element.revision, element_type=element.element_type, text=element.text,
        page=element.page, page_end=element.page, section_number=element.section_number,
        section_title=element.section_title, section_path=list(element.section_path),
        element_ids=[element.element_id], parent_chunk_id=parent.chunk_id if parent else None,
        table_id=element.table_id, figure_id=element.figure_id, bbox=element.bbox,
        n_tokens=estimate_tokens(element.text),
        meta={"kind": f"{kind}_caption", "identifiers": sorted(identifiers_in(element.text))},
    )


def _dominant_type(group: Sequence[Element]) -> str:
    counts: dict[str, int] = {}
    for el in group:
        counts[el.element_type] = counts.get(el.element_type, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else "paragraph"


# --------------------------------------------------------------------------- #
# helpers used by WS4/WS5
# --------------------------------------------------------------------------- #
def expand_to_parents(chunks: Sequence[Chunk], parent_ids: Iterable[str]) -> list[Chunk]:
    """Replace child chunks with their parents (parent–child retrieval)."""
    wanted = set(parent_ids)
    by_id = {c.chunk_id: c for c in chunks}
    return [by_id[pid] for pid in wanted if pid in by_id]


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, Any]:
    sizes = [c.n_tokens for c in chunks] or [0]
    by_type: dict[str, int] = {}
    for c in chunks:
        key = c.meta.get("kind", c.element_type)
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "chunks": len(chunks),
        "mean_tokens": round(sum(sizes) / len(sizes), 1),
        "max_tokens": max(sizes),
        "min_tokens": min(sizes),
        "by_kind": dict(sorted(by_type.items())),
        "with_parent": sum(1 for c in chunks if c.parent_chunk_id),
        "identifiers": sum(1 for c in chunks if c.meta.get("identifiers")),
    }


def chunk_corpus(elements: Sequence[Element], *, tables_by_doc: dict[str, Sequence[Table]] | None = None,
                 figures_by_doc: dict[str, Sequence[Figure]] | None = None,
                 options: ChunkOptions | None = None) -> list[Chunk]:
    """Chunk every document, keeping document order so page ranges stay sane."""
    tables_by_doc = tables_by_doc or {}
    figures_by_doc = figures_by_doc or {}
    by_doc: dict[str, list[Element]] = {}
    order: list[str] = []
    for element in elements:
        if element.doc_id not in by_doc:
            by_doc[element.doc_id] = []
            order.append(element.doc_id)
        by_doc[element.doc_id].append(element)
    chunks: list[Chunk] = []
    for doc_id in order:
        doc_elements = sorted(by_doc[doc_id], key=lambda e: (e.page, e.bbox[1] if e.bbox else 0, e.order))
        chunks.extend(chunk_elements(doc_elements, tables=tables_by_doc.get(doc_id, ()),
                                     figures=figures_by_doc.get(doc_id, ()), options=options))
    return chunks
