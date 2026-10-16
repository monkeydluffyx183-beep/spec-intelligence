"""Flow layout for the synthetic corpus: blocks in, paginated PDFs + manifest out.

The generator keeps a *manifest* of everything it wrote (section, table cells,
figure nodes, page numbers).  That manifest is the ground truth used by the
ingestion tests, the OCR benchmark labels and the gold set, so the numbers in
the evaluation are checked against what is actually on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.synthetic.pdfwrite import (
    MARGIN_L as ML,
    PAGE_H,
    PAGE_W,
    TEXT_W,
    ImageAsset,
    Page,
    PdfWriter,
    draw_boxes_diagram,
    text_width,
    wrap,
)

COL_W = TEXT_W / 12.0


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    kind: str
    anchor: str = ""          # stable id used by the manifest and the gold set
    section: str = ""
    section_title: str = ""


@dataclass
class Heading(Block):
    number: str = ""
    title: str = ""
    level: int = 1
    kind: str = "heading"


@dataclass
class Paragraph(Block):
    text: str = ""
    kind: str = "paragraph"


@dataclass
class Note(Block):
    text: str = ""
    kind: str = "note"


@dataclass
class Bullets(Block):
    items: list[str] = field(default_factory=list)
    kind: str = "bullets"


@dataclass
class TableBlock(Block):
    number: str = ""
    caption: str = ""
    headers: list[list[str]] = field(default_factory=list)   # one or more header rows
    rows: list[list[str]] = field(default_factory=list)
    section: str = ""
    repeat_headers: bool = True
    col_widths: list[float] | None = None      # relative weights; None = auto-fit
    max_cell_lines: int = 3
    kind: str = "table"
    # computed
    pages: list[int] = field(default_factory=list)


@dataclass
class FigureBlock(Block):
    number: str = ""
    caption: str = ""
    kind: str = "figure"
    figure_kind: str = "state_machine"       # state_machine | timing | block | raster
    nodes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    raster: tuple[int, int] = (0, 0)
    description: str = ""                    # ground-truth VLM description
    structure: dict[str, Any] = field(default_factory=dict)
    section: str = ""
    page: int = 0


@dataclass
class PageBreak(Block):
    kind: str = "pagebreak"


# --------------------------------------------------------------------------- #
# flow engine
# --------------------------------------------------------------------------- #
class Flow:
    """Places blocks onto pages, emitting headers, footers and a manifest."""

    def __init__(self, writer: PdfWriter, doc_id: str, revision: str, running_title: str,
                 *, top_y: float = 96.0, bottom_y: float = PAGE_H - 84.0, start_page_no: int = 0) -> None:
        self.writer = writer
        self.doc_id = doc_id
        self.revision = revision
        self.running_title = running_title
        self.top_y = top_y
        self.bottom_y = bottom_y
        self.manifest: list[dict[str, Any]] = []
        self._page: Page | None = None
        self._y = top_y
        # ``start_page_no`` is the number of pages already written (the title page),
        # so the page numbers recorded here are the *PDF* page numbers.  The gold
        # set is built from this manifest and the ingester reports PDF page
        # numbers; if the two disagreed the evaluation would silently measure a
        # one-page offset.
        self._page_no = start_page_no
        self._section = ""
        self._section_title = ""
        self._section_path: list[str] = []

    # -- page handling ----------------------------------------------------- #
    @property
    def page_no(self) -> int:
        return self._page_no

    def new_page(self) -> Page:
        self._page_no += 1
        page = self.writer.add_page()
        self._page = page
        self._y = self.top_y
        self._header(page)
        return page

    def _header(self, page: Page) -> None:
        page.text(ML, 40, self.running_title, font="bold", size=8.0)
        page.text(ML, 54, f"Revision {self.revision}", size=7.5)
        page.line(ML, 62, PAGE_W - 72, 62, gray=0.4, line_width=0.5)
        page.line(ML, PAGE_H - 66, PAGE_W - 72, PAGE_H - 66, gray=0.4, line_width=0.5)
        page.text(ML, PAGE_H - 60, "Copyright © 2026 USB Implementers Forum, Inc. — "
                                   "Licensed material; redistribution prohibited.", size=7.0)
        page.text(PAGE_W - 150, PAGE_H - 60, f"Page {self._page_no}", size=7.0)

    def _ensure(self, needed: float) -> None:
        if self._page is None or self._y + needed > self.bottom_y:
            self.new_page()

    def _record(self, entry: dict[str, Any]) -> None:
        """Record a manifest entry.

        ``page`` defaults to the page being written, but a block that spans pages
        (a stitched table, a tall figure) may pass the page it *starts* on, because
        that is the page a citation to it should name and the page the ingester
        reports for the table.
        """
        entry.setdefault("doc_id", self.doc_id)
        entry.setdefault("page", self._page_no)
        entry.setdefault("section", self._section)
        entry.setdefault("section_title", self._section_title)
        self.manifest.append(entry)

    # -- blocks ------------------------------------------------------------ #
    def add(self, block: Block) -> None:
        if isinstance(block, Heading):
            self._heading(block)
        elif isinstance(block, Paragraph):
            self._paragraph(block)
        elif isinstance(block, Note):
            self._note(block)
        elif isinstance(block, Bullets):
            self._bullets(block)
        elif isinstance(block, TableBlock):
            self._table(block)
        elif isinstance(block, FigureBlock):
            self._figure(block)
        elif isinstance(block, PageBreak):
            self.new_page()
        else:  # pragma: no cover - defensive
            raise TypeError(f"unknown block {block!r}")

    def _heading(self, block: Heading) -> None:
        size = {1: 13.0, 2: 11.0, 3: 9.5}.get(block.level, 9.5)
        self._ensure(size * 3.2)
        if block.level == 1:
            self._section_path = [block.number]
        elif block.level == 2:
            self._section_path = self._section_path[:1] + [block.number]
        else:
            self._section_path = self._section_path[:2] + [block.number]
        self._section = block.number
        self._section_title = block.title
        if self._page is None:
            self.new_page()
        assert self._page is not None
        self._y += 6
        self._page.text(ML, self._y, f"{block.number} {block.title}", font="bold", size=size)
        self._y += size * 1.6
        self._record({"type": "section_heading", "anchor": block.anchor or f"sec:{block.number}",
                      "level": block.level, "number": block.number, "title": block.title,
                      "text": f"{block.number} {block.title}", "section_path": list(self._section_path)})

    def _paragraph(self, block: Paragraph) -> None:
        size = 9.0
        lines = wrap(block.text, size)
        for i, line in enumerate(lines):
            self._ensure(12.0)
            assert self._page is not None
            self._page.text(ML, self._y, line, size=size)
            self._y += 11.5
        self._y += 3.5
        self._record({"type": "paragraph", "anchor": block.anchor, "text": block.text})

    def _note(self, block: Note) -> None:
        size = 8.5
        lines = wrap(block.text, size, width=TEXT_W - 18)
        height = len(lines) * 10.8 + 10
        self._ensure(height)
        assert self._page is not None
        self._page.rect(ML, self._y - 3, TEXT_W, height - 2, fill=0.95, stroke=0.5)
        for i, line in enumerate(lines):
            self._page.text(ML + 6, self._y + 3 + i * 10.8, line, size=size)
        self._y += height + 4
        self._record({"type": "note", "anchor": block.anchor, "text": block.text})

    def _bullets(self, block: Bullets) -> None:
        size = 9.0
        for item in block.items:
            lines = wrap(item, size, width=TEXT_W - 14)
            for i, line in enumerate(lines):
                self._ensure(11.5)
                assert self._page is not None
                prefix = "• " if i == 0 else "  "
                self._page.text(ML + 6, self._y, prefix + line, size=size)
                self._y += 11.2
        self._y += 4
        self._record({"type": "list_item", "anchor": block.anchor, "text": "\n".join(block.items)})

    # -- tables ------------------------------------------------------------ #
    ROW_H = 12.0
    HEADER_H = 12.5
    CELL_FONT = 7.5
    CELL_LEADING = 9.5

    #: minimum horizontal space between two columns' text runs
    GUTTER = 5.0

    def _column_widths(self, block: TableBlock, n_cols: int, width: float) -> list[float]:
        """Column widths: explicit weights, else auto-fit to *rendered* text width.

        Sizing by character count is not enough: ``COMMON.CHECK.PD.2`` is 17
        characters and needs about 75 pt at 7.5 pt, and a column one point too
        narrow makes a PDF reader run the next column's text into this one
        ("COMMON.CHECK.PD.2Verify"), which corrupts every downstream table cell.
        Real specification tables are typeset to fit their content, so this
        measures the content and adds a gutter.
        """
        weights = block.col_widths
        if not weights:
            weights = []
            for c in range(n_cols):
                cells = [str(r[c]) for r in block.rows if c < len(r)]
                cells += [str(h[c]) for h in block.headers if c < len(h)]
                widest = max((text_width(cell, self.CELL_FONT, "bold") for cell in cells), default=30.0)
                weights.append(float(min(320.0, max(28.0, widest + self.GUTTER))))
        total = sum(weights) or 1.0
        cols = [width * w / total for w in weights]
        # hand any room left over to the widest column.  The slack is measured in
        # points *after* scaling: a first version compared weight units with the
        # point width and added the difference in points, which made a narrow
        # table's first column 470 pt wide and pushed the remaining columns off
        # the page — the cells were simply not drawn, so nothing downstream could
        # recover them.
        leftover = width - sum(cols)
        if leftover > 1.0:
            widest = max(range(len(weights)), key=lambda i: weights[i])
            cols[widest] += leftover
        return cols

    def _cell_lines(self, cell: object, col_w: float, max_lines: int) -> list[str]:
        return wrap(str(cell), self.CELL_FONT, width=max(18.0, col_w - 6))[:max_lines]

    def _table(self, block: TableBlock) -> None:
        n_cols = max([len(r) for r in block.rows] + [len(h) for h in block.headers] or [1])
        width = min(TEXT_W, max(300.0, TEXT_W * min(1.0, 0.62 + 0.075 * n_cols)))
        col_ws = self._column_widths(block, n_cols, width)
        if block.number:
            self._ensure(30)
            assert self._page is not None
            cap = f"Table {block.number}" + (f". {block.caption}" if block.caption else "")
            for line in wrap(cap, 8.5, "bold", width):
                self._page.text(ML, self._y, line, font="bold", size=8.5)
                self._y += 10.5
            self._y += 2
        header_rows = block.headers or [[]]
        part = 0
        page_start = None
        idx = 0
        while True:
            self._ensure(self.HEADER_H * len(header_rows) + self.ROW_H + 8)
            if page_start is None:
                page_start = self._page_no
            page = self._page
            assert page is not None
            if part > 0 and block.number:
                page.text(ML, self._y, f"Table {block.number} (continued)", font="bold", size=8.5)
                self._y += 10.5
            # header rows (repeated on every page part)
            for hr in (header_rows if (part == 0 or block.repeat_headers) else [[]]):
                if hr:
                    lines = [self._cell_lines(hr[c] if c < len(hr) else "", col_ws[c], 2) for c in range(n_cols)]
                    height = max(self.HEADER_H, self.CELL_LEADING * max(len(l) for l in lines) + 4)
                    page.rect(ML, self._y - 2, width, height, fill=0.9, stroke=0.4)
                    for c, cell_lines in enumerate(lines):
                        x = ML + sum(col_ws[:c])
                        for li, line in enumerate(cell_lines):
                            page.text(x + 3, self._y + 2.5 + li * self.CELL_LEADING, line, font="bold",
                                      size=self.CELL_FONT)
                    # column rules on the header row too: without them a table
                    # detector sees the header as one merged cell
                    for c in range(n_cols + 1):
                        x = ML + sum(col_ws[:c])
                        page.line(x, self._y - 2, x, self._y + height - 2, gray=0.6, line_width=0.4)
                    page.line(ML, self._y + height - 2, ML + width, self._y + height - 2, gray=0.6,
                              line_width=0.4)
                    self._y += height
            # body rows
            while idx < len(block.rows):
                row = block.rows[idx]
                cell_lines = [self._cell_lines(row[c] if c < len(row) else "", col_ws[c], block.max_cell_lines)
                              for c in range(n_cols)]
                row_h = max(self.ROW_H, self.CELL_LEADING * max(len(l) for l in cell_lines) + 4)
                if self._y + row_h > self.bottom_y - 6:
                    break
                for c, lines in enumerate(cell_lines):
                    x = ML + sum(col_ws[:c])
                    for li, line in enumerate(lines):
                        page.text(x + 3, self._y + 2.5 + li * self.CELL_LEADING, line, size=self.CELL_FONT)
                for c in range(n_cols + 1):
                    x = ML + sum(col_ws[:c])
                    page.line(x, self._y - 2, x, self._y + row_h - 2, gray=0.75, line_width=0.35)
                page.line(ML, self._y + row_h - 2, ML + width, self._y + row_h - 2, gray=0.75, line_width=0.35)
                self._y += row_h
                idx += 1
            part += 1
            if idx >= len(block.rows):
                break
            self.new_page()
        if page_start is not None:
            block.pages = list(range(page_start, self._page_no + 1))
        self._y += 8
        self._record({"type": "table", "anchor": block.anchor or f"table:{block.number}",
                      "number": block.number, "caption": block.caption, "headers": block.headers,
                      "rows": block.rows, "pages": block.pages, "n_cols": n_cols,
                      "page": (block.pages[0] if block.pages else self._page_no),
                      "text": f"Table {block.number}. {block.caption}"})

    # -- figures ----------------------------------------------------------- #
    def _figure(self, block: FigureBlock) -> None:
        height = 190.0 if block.figure_kind != "timing" else 150.0
        self._ensure(height + 30)
        page = self._page
        assert page is not None
        block.page = self._page_no
        x, y = ML, self._y
        if block.raster[0]:
            asset = ImageAsset.from_pattern(block.raster[0], block.raster[1], seed=abs(hash(block.number)) % 97,
                                            kind="boxes")
            page.image(x, y, TEXT_W, height, asset)
        else:
            draw_boxes_diagram(page, x, y, TEXT_W, height, block.nodes, block.edges,
                               title=f"Figure {block.number} — {block.caption[:60]}")
        self._y += height + 4
        cap = f"Figure {block.number}. {block.caption}"
        for line in wrap(cap, 8.5, width=TEXT_W - 10):
            page.text(ML + 10, self._y, line, size=8.5)
            self._y += 10.2
        self._y += 8
        self._record({"type": "figure", "anchor": block.anchor or f"figure:{block.number}",
                      "number": block.number, "caption": block.caption, "figure_kind": block.figure_kind,
                      "description": block.description, "structure": block.structure,
                      "nodes": [n[0] for n in block.nodes], "edges": block.edges,
                      "raster": bool(block.raster[0]),
                      "text": cap})


def section_path(flow: Flow) -> list[str]:
    return list(flow._section_path)


# --------------------------------------------------------------------------- #
# document assembly helper
# --------------------------------------------------------------------------- #
def title_page(writer: PdfWriter, title: str, revision: str, doc_id: str, date: str,
               license_line: str, toc: Sequence[tuple[str, str]]) -> None:
    page = writer.add_page()
    page.text(ML, 150, title, font="bold", size=18.0)
    page.text(ML, 180, revision, size=12.0)
    page.text(ML, 200, date, size=10.0)
    page.text(ML, 240, license_line, size=8.5)
    page.text(ML, 300, "Table of Contents", font="bold", size=11.0)
    y = 320.0
    for number, name in toc:
        page.text(ML, y, f"{number}  {name}", size=9.0)
        y += 12.0
    page.text(ML, PAGE_H - 90, f"Document id {doc_id} — synthetic stand-in generated by "
                              f"specintel.synthetic (not the licensed USB-IF document).", size=7.5)


def iter_blocks(blocks: Iterable[Block]) -> Iterable[Block]:
    return blocks
