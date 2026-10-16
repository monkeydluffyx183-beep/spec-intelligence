"""WS1 — tables: detection, Markdown/JSON export, and stitching across pages.

Two entry points:

``detect_tables(path, options)``
    Uses PyMuPDF's line/grid based detector to turn table regions into
    :class:`~specintel.common.models.Table`, pairing each region with its
    caption ("Table 6.8. Timer definitions and limits").

``stitch_tables(tables, repeat_header_rows)``
    Glues the parts of a table that continues on the next page into one table.
    A continuation is recognised by a caption marked "(continued)", or by a
    part whose first row repeats the previous part's header.  Header rows are
    kept on every part so a split table stays readable in retrieval, and the
    stitched table records every page it touches for citation.

The exported Markdown is what the LLM sees; the rows/headers are what the code
sees.  Both keep the table number and caption, because the assignment's
questions are usually phrased as "in which table is X defined".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.models import Table, new_id
from specintel.ingest.parser import CAPTION_RE, CONTINUED_RE, open_pdf, page_text_lines

TABLE_CAPTION_ABOVE = re.compile(r"^\s*Table\s+[\w.\-@]+\s*[.:]", re.IGNORECASE)


@dataclass
class RawTablePart:
    """A table region detected on one page, before stitching."""

    page: int
    bbox: list[float]
    headers: list[list[str]]
    rows: list[list[str]]
    number: str = ""
    caption: str = ""
    continued: bool = False
    section_number: str = ""

    @property
    def n_cols(self) -> int:
        sizes = [len(r) for r in self.rows] + [len(h) for h in self.headers]
        return max(sizes) if sizes else 0


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, str):
        return " ".join(cell.split())
    return " ".join(str(cell).split())


def _clean_rows(rows: Sequence[Sequence[Any]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_cell_text(c) for c in row]
        if any(cells):
            cleaned.append(cells)
    # drop fully empty leading/trailing rows
    while cleaned and not any(cleaned[0]):
        cleaned.pop(0)
    while cleaned and not any(cleaned[-1]):
        cleaned.pop()
    return cleaned


def _find_caption(page_lines: Sequence[Any], bbox: Sequence[float], prefer_above: bool = True) -> tuple[str, str, bool]:
    """Return ``(number, caption, continued)`` for the table at ``bbox``."""
    y0, y1 = float(bbox[1]), float(bbox[3])
    best: tuple[float, str] | None = None
    for line in page_lines:
        match = CAPTION_RE.match(line.text)
        if not match or match.group("kind").lower() != "table":
            continue
        if prefer_above and line.y1 <= y0 + 4:
            distance = y0 - line.y1
        elif not prefer_above and line.y0 >= y1 - 4:
            distance = line.y0 - y1
        else:
            continue
        # the caption baseline may overlap the grid it belongs to by a fraction
        # of a point, which is normal in vector PDFs
        if distance < -4 or distance > 60:
            continue
        if best is None or distance < best[0]:
            best = (distance, line.text)
    if best is None:
        return "", "", False
    text = best[1]
    match = CAPTION_RE.match(text)
    number = match.group("number") if match else ""
    caption = match.group("text") if match else text
    return number, caption.strip(), bool(CONTINUED_RE.search(text))


def detect_tables(path: str | Path, *, doc_id: str, revision: str, max_pages: int | None = None) -> list[RawTablePart]:
    """Detect table regions in a PDF and pair them with their captions."""
    doc = open_pdf(path)
    parts: list[RawTablePart] = []
    pages = range(doc.page_count if max_pages is None else min(doc.page_count, max_pages))
    for pno in pages:
        page = doc[pno]
        lines = page_text_lines(page, pno + 1)
        try:
            finder = page.find_tables()
            found = list(getattr(finder, "tables", []) or [])
        except Exception:
            found = []
        for table in found:
            rows = _clean_rows(table.extract() or [])
            if not rows:
                continue
            bbox = [float(v) for v in (table.bbox or (0, 0, 0, 0))]
            number, caption, continued = _find_caption(lines, bbox, prefer_above=True)
            if not number:                      # caption below the grid
                number, caption, continued = _find_caption(lines, bbox, prefer_above=False)
            # the first row is treated as the header when the caption sits above
            headers = [rows[0]] if rows else []
            body = rows[1:] if len(rows) > 1 else []
            parts.append(RawTablePart(page=pno + 1, bbox=bbox, headers=headers, rows=body,
                                      number=number, caption=caption, continued=continued))
    doc.close()
    return parts


# --------------------------------------------------------------------------- #
# stitching
# --------------------------------------------------------------------------- #
def _row_signature(row: Sequence[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\W+", "", c).lower() for c in row)


def stitch_tables(parts: Sequence[RawTablePart], *, doc_id: str, revision: str,
                  repeat_header_rows: int = 1, require_caption_match: bool = True) -> list[Table]:
    """Merge multi-page table parts into :class:`Table` objects.

    A part is treated as a continuation of the previous part when it sits on
    the immediately following page and either its caption says "continued",
    or its header row repeats the previous header while its caption matches.
    """
    stitched: list[Table] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        headers = current["headers"][:max(1, repeat_header_rows)] if current["headers"] else []
        stitched.append(Table(
            table_id=current["table_id"], doc_id=doc_id, revision=revision, number=current["number"],
            caption=current["caption"], pages=sorted(set(current["pages"])), headers=headers,
            rows=current["rows"], section_number=current.get("section_number", ""), bbox=current["bbox"],
            element_ids=[], stitched=len(set(current["pages"])) > 1, extractor="pymupdf",
        ))
        current = None

    for part in parts:
        if current is None:
            current = _new_current(part, doc_id)
            continue
        prev_page = max(current["pages"])
        same_table = (
            part.page == prev_page + 1
            and not current.get("closed")
            and (
                part.continued
                or (not require_caption_match and _same_header(current["headers"], part.headers))
                or (bool(part.number) and part.number == current["number"])
            )
        )
        if same_table:
            current["rows"].extend(part.rows)
            current["pages"].append(part.page)
            if not current["number"] and part.number:
                current["number"] = part.number
            continue
        flush()
        current = _new_current(part, doc_id)
    flush()
    return stitched


def _new_current(part: RawTablePart, doc_id: str) -> dict[str, Any]:
    return {
        # the document id comes first: ids are the join key between WS1, WS2, WS3
        # and WS4, and they must be unique across the whole corpus, not just
        # inside one document
        "table_id": f"{doc_id}:table:{new_id('tbl')}",
        "number": part.number,
        "caption": part.caption,
        "headers": [list(h) for h in part.headers],
        "rows": [list(r) for r in part.rows],
        "pages": [part.page],
        "bbox": list(part.bbox),
        "section_number": part.section_number,
        "closed": False,
    }


def _same_header(a: Sequence[Sequence[str]], b: Sequence[Sequence[str]]) -> bool:
    if not a or not b:
        return False
    return _row_signature(a[0]) == _row_signature(b[0])


def attach_sections(tables: Sequence[Table], elements: Iterable[Any]) -> None:
    """Fill ``section_number`` from the parsed elements that sit above each table."""
    by_doc_page: dict[tuple[str, int], list[Any]] = {}
    for el in elements:
        by_doc_page.setdefault((el.doc_id, el.page), []).append(el)
    for table in tables:
        candidates = [el for el in by_doc_page.get((table.doc_id, table.pages[0]), [])
                      if el.element_type == "section_heading"
                      and (not table.bbox or (el.bbox or [0, 0, 0, 0])[1] <= table.bbox[1] + 2)]
        if candidates:
            table.section_number = max(candidates, key=lambda e: (e.bbox or [0, 0, 0, 0])[1]).section_number


# --------------------------------------------------------------------------- #
# export helpers
# --------------------------------------------------------------------------- #
def tables_to_json(tables: Sequence[Table]) -> dict[str, Any]:
    """Export tables as JSON (the code-facing form of Task 2, Step 3)."""
    return {"tables": [t.to_dict() for t in tables]}


def table_markdown_bundle(tables: Sequence[Table]) -> str:
    """One Markdown document with every table, number, caption and pages."""
    chunks = []
    for t in tables:
        pages = f"p.{t.pages[0]}" if len(t.pages) == 1 else f"p.{t.pages[0]}-{t.pages[-1]}"
        chunks.append(f"<!-- table_id={t.table_id} doc={t.doc_id} section={t.section_number} {pages} -->\n"
                      f"### Table {t.number or '?'}. {t.caption}\n\n{t.markdown}\n")
    return "\n".join(chunks)


def table_cell_accuracy(gold: Table, pred: Table) -> dict[str, float]:
    """Cell-level accuracy of a predicted table against a gold table.

    Reports exact-match cell accuracy plus row alignment, which is what the
    hand-check of ten tables and the OCR benchmark report.
    """
    n_cols = max(gold.n_cols, pred.n_cols, 1)
    gold_cells = [c for r in gold.rows for c in r]
    hits = sum(1 for i, row in enumerate(pred.rows)
               if i < len(gold.rows)
               for j, cell in enumerate(row)
               if j < len(gold.rows[i]) and _norm(cell) == _norm(gold.rows[i][j]))
    total = max(len(gold_cells), 1)
    row_hits = sum(1 for i, row in enumerate(pred.rows)
                   if i < len(gold.rows) and [_norm(c) for c in row] == [_norm(c) for c in gold.rows[i]])
    return {
        "cell_accuracy": hits / total,
        "row_accuracy": row_hits / max(len(gold.rows), 1),
        "col_match": 1.0 if gold.n_cols == pred.n_cols else 0.0,
        "gold_cells": float(len(gold_cells)),
        "pred_cells": float(sum(len(r) for r in pred.rows)),
    }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00b5", "u").replace("\u2013", "-")).strip().lower()
