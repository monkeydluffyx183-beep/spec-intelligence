"""WS1 — layout-aware parsing: PDF pages → typed elements with provenance.

Two parsers are provided so the assignment can compare them on the same pages
(Task 2, Step 1):

``PymupdfParser``      font/size/position aware layout parser: classifies
                       headings, paragraphs, captions, notes, headers and
                       footers, keeps the section hierarchy and a bbox per
                       element.
``PymupdfPlainParser`` the "naive" baseline: raw text blocks in reading order
                       with no classification, which is what a plain text
                       extractor gives you.

Both emit the same :class:`~specintel.common.models.Element` stream, so WS2/WS3
do not care which parser produced it — only the ``extractor`` field differs.
Table and figure detection live in :mod:`specintel.ingest.tables` and
:mod:`specintel.ingest.figures`; this module provides the text path and the
shared page utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from specintel.common.models import ELEMENT_TYPES, Element, new_id

# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #
SECTION_RE = re.compile(r"^(?P<number>(?:\d+(?:\.\d+)*)|[A-Z](?:\.\d+)*)\s+(?P<title>\S.{2,120})$")
#: Caption forms seen in the four spec families: "Table 6.8. Timer definitions
#: and limits" (USB-PD), "Table 4-27 Configuration Channel" (Type-C writes no
#: punctuation after the number), "Table A.1. Revision history", "Figure 7-2
#: Timing of an EPR mode entry".
#:
#: The hard case is telling a caption from a cross-reference in prose: "Table
#: 4-27 shows the CC terminations" is a sentence, not a caption.  Three guards do
#: it — the number may not be a prefix of a longer number (``(?!\.\d)``), the
#: number may not run into a word (``(?![\w\-])``), and what follows may not
#: begin with a lowercase word, because a caption names something and a
#: reference predicates something ("shows", "gives", "is", "and", "below").
CAPTION_RE = re.compile(
    r"^(?P<kind>Table|Figure)\s+"
    # a caption number always contains a digit: without this, the title page's
    # "Table of Contents" is read as a table caption
    r"(?=[\w.\-]*\d)(?P<number>[A-Za-z0-9][\w\-]*(?:\.[0-9A-Za-z]+)*)"
    # the lowercase test is scoped with (?-i:…) on purpose: the pattern is
    # compiled with re.IGNORECASE, which would otherwise make [a-z] match an
    # uppercase letter and reject every real caption
    r"(?P<sep>[.:])?(?![\w\-])(?!\.\d)(?!\s+(?-i:[a-z]))\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE)

CONTINUED_RE = re.compile(r"\(?\s*continued\s*\)?", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^\s*(?:[•\-\u2022]\s+|[a-z]\)\s+|\d+\.\s+)")
NOTE_RE = re.compile(r"^\s*(?:NOTE|Note)\s*[:.]", )
BOILERPLATE_RE = re.compile(
    r"(copyright|all rights reserved|licensed material|redistribution|usb implementers forum|"
    r"wireless power consortium|page \d+\s*(of|/)\s*\d+|^\s*page\s+\d+\s*$)",
    re.IGNORECASE,
)
SENTENCE_END = (".", ":", ";", "!", "?")


@dataclass
class ParseOptions:
    """Knobs for the layout parser (mirrors ``ingest:`` in config.yaml)."""

    doc_id: str
    revision: str
    extractor: str = "pymupdf"
    header_zone: float = 62.0        # y below which text is a running header
    footer_zone: float = 72.0        # distance from the bottom for the footer
    min_heading_size: float = 9.4
    max_paragraph_gap: float = 6.0   # vertical gap that still joins lines of a paragraph


@dataclass
class RawLine:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    block_no: int
    line_no: int

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class RawBlock:
    page: int
    lines: list[RawLine] = field(default_factory=list)
    bbox: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _join_lines([l.text for l in self.lines])

    @property
    def size(self) -> float:
        return max((l.size for l in self.lines), default=0.0)

    @property
    def bold(self) -> bool:
        return any(l.bold for l in self.lines[:2])


def _join_lines(lines: Sequence[str]) -> str:
    """Join wrapped lines, removing hyphenation introduced by the layout."""
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if out.endswith("-") and not out.endswith("--") and piece[:1].islower():
            out = out[:-1] + piece
        elif out:
            out = f"{out} {piece}"
        else:
            out = piece
    return out.strip()


# --------------------------------------------------------------------------- #
# pdf access helpers
# --------------------------------------------------------------------------- #
def require_pymupdf():
    """Import PyMuPDF, raising a helpful error when it is not installed."""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyMuPDF is required to parse PDFs: pip install 'pymupdf>=1.24' (extra: specintel[pdf]). "
            "Without it only the synthetic/text fixtures are usable."
        ) from exc


def open_pdf(path: str | Path):
    pymupdf = require_pymupdf()
    return pymupdf.open(str(path))


def page_text_lines(page: Any, page_no: int) -> list[RawLine]:
    """Extract text lines with font metrics from a PyMuPDF page."""
    out: list[RawLine] = []
    payload = page.get_text("dict")
    for bno, block in enumerate(payload.get("blocks", [])):
        if block.get("type") != 0:      # image block
            continue
        for lno, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            sizes = [float(s.get("size", 0.0)) for s in spans] or [0.0]
            fonts = " ".join(str(s.get("font", "")) for s in spans)
            x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))
            out.append(RawLine(page=page_no, text=text, x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                               size=max(sizes), bold=("bold" in fonts.lower() or "black" in fonts.lower()),
                               block_no=bno, line_no=lno))
    return out


def group_blocks(lines: Sequence[RawLine], max_gap: float = 6.0) -> list[RawBlock]:
    """Group lines into blocks using PyMuPDF's block ids plus a gap check."""
    blocks: dict[tuple[int, int], RawBlock] = {}
    for line in lines:
        key = (line.page, line.block_no)
        block = blocks.get(key)
        if block is None:
            block = RawBlock(page=line.page, bbox=list(line.bbox))
            blocks[key] = block
        block.lines.append(line)
        b = block.bbox or list(line.bbox)
        block.bbox = [min(b[0], line.x0), min(b[1], line.y0), max(b[2], line.x1), max(b[3], line.y1)]
    grouped = [b for b in blocks.values() if b.lines]
    grouped.sort(key=lambda b: (b.page, b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
    return grouped


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify_block(block: RawBlock, options: ParseOptions, page_height: float) -> str:
    """Decide the element type of a text block."""
    text = block.text
    bbox = block.bbox or [0, 0, 0, 0]
    if bbox[1] < options.header_zone and (BOILERPLATE_RE.search(text) or len(text) < 90):
        return "header"
    if bbox[3] > page_height - options.footer_zone and (BOILERPLATE_RE.search(text) or len(text) < 60):
        return "footer"
    if BOILERPLATE_RE.search(text) and len(text) < 200:
        return "footer"
    caption = CAPTION_RE.match(text)
    if caption:
        return "table_caption" if caption.group("kind").lower() == "table" else "figure_caption"
    if NOTE_RE.match(text):
        return "note"
    if SECTION_RE.match(text) and (block.bold or block.size >= options.min_heading_size + 1.2) and len(text) < 130:
        return "section_heading"
    if block.bold and len(text) < 90 and not text.endswith(SENTENCE_END):
        return "section_heading"
    if LIST_ITEM_RE.match(text):
        return "list_item"
    return "paragraph"


def heading_level(block: RawBlock) -> int:
    """Infer the heading depth from the section number and font size."""
    text = block.text
    match = SECTION_RE.match(text)
    number = match.group("number") if match else ""
    if number and "." in number:
        return min(3, number.count(".") + 1)
    if match and re.fullmatch(r"[A-Z](\.\d+)*", number):
        return min(3, number.count(".") + 1)
    return 1


def parse_heading(block: RawBlock) -> tuple[str, str]:
    """Return ``(section_number, section_title)`` for a heading block."""
    match = SECTION_RE.match(block.text)
    if match:
        return match.group("number"), match.group("title").strip()
    return "", block.text.strip()


def update_section_stack(stack: list[tuple[str, str]], number: str, title: str, level: int) -> None:
    """Maintain the section hierarchy so every element can be cited by section."""
    if not number:
        return
    del stack[level - 1:]
    while len(stack) < level - 1:
        stack.append(("", ""))
    stack.append((number, title))


def section_path(stack: Sequence[tuple[str, str]]) -> list[str]:
    return [f"{n} {t}".strip() for n, t in stack if n]


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
class PymupdfPlainParser:
    """Naive text-block parser: the plain-extraction baseline of Task 2."""

    name = "pymupdf-plain"

    def parse(self, path: str | Path, options: ParseOptions) -> list[Element]:
        doc = open_pdf(path)
        elements: list[Element] = []
        order = 0
        for pno in range(doc.page_count):
            page = doc[pno]
            for block in group_blocks(page_text_lines(page, pno + 1)):
                order += 1
                elements.append(Element(
                    element_id=f"{options.doc_id}:plain:{pno + 1}:{order}",
                    doc_id=options.doc_id, revision=options.revision, element_type="paragraph",
                    page=pno + 1, text=block.text, bbox=block.bbox, order=order,
                    extractor=self.name, meta={"parser": self.name},
                ))
        doc.close()
        return elements


class PymupdfParser:
    """Layout-aware parser: typing, section hierarchy, captions, order."""

    name = "pymupdf"

    def __init__(self, options: ParseOptions | None = None) -> None:
        self.options = options

    def parse(self, path: str | Path, options: ParseOptions | None = None) -> list[Element]:
        opts = options or self.options
        if opts is None:
            raise ValueError("ParseOptions are required")
        doc = open_pdf(path)
        elements: list[Element] = []
        stack: list[tuple[str, str]] = []
        current_section, current_title = "", ""
        order = 0
        for pno in range(doc.page_count):
            page = doc[pno]
            page_height = float(page.rect.height)
            blocks = group_blocks(page_text_lines(page, pno + 1))
            for block in blocks:
                etype = classify_block(block, opts, page_height)
                order += 1
                if etype == "section_heading":
                    number, title = parse_heading(block)
                    level = heading_level(block)
                    update_section_stack(stack, number, title, level)
                    current_section, current_title = (number or current_section), (title or current_title)
                element = Element(
                    element_id=f"{opts.doc_id}:{etype}:{pno + 1}:{order}",
                    doc_id=opts.doc_id, revision=opts.revision, element_type=etype, page=pno + 1,
                    text=block.text, markdown=block.text, bbox=block.bbox, order=order,
                    section_number=current_section, section_title=current_title,
                    section_path=section_path(stack), extractor=opts.extractor,
                    meta={"parser": self.name, "level": heading_level(block) if etype == "section_heading" else None,
                          "font_size": round(block.size, 2), "bold": block.bold},
                )
                if etype in ("table_caption", "figure_caption"):
                    cap = CAPTION_RE.match(block.text)
                    if cap:
                        element.caption = block.text
                        element.meta["caption_number"] = cap.group("number")
                        element.meta["caption_kind"] = cap.group("kind").lower()
                elements.append(element)
        doc.close()
        return elements

    # convenience: parse every PDF of one document in order ----------------- #
    def parse_document(self, paths: Iterable[str | Path], options: ParseOptions) -> list[Element]:
        out: list[Element] = []
        for p in paths:
            out.extend(self.parse(p, options))
        return out


def available_parsers() -> dict[str, str]:
    """Registry of parser backends; ``auto`` picks the best installed one."""
    registry = {"pymupdf": "PymupdfParser (layout aware)",
                "pymupdf-plain": "PymupdfPlainParser (naive baseline)"}
    try:
        require_pymupdf()
    except RuntimeError:
        return {"fallback": "text fixtures only (PyMuPDF not installed)"}
    return registry


def get_parser(name: str = "auto") -> Any:
    """Return a parser instance for ``name`` (``auto`` → layout-aware parser)."""
    if name in ("auto", "pymupdf", "layout"):
        return PymupdfParser()
    if name in ("pymupdf-plain", "plain", "naive"):
        return PymupdfPlainParser()
    raise ValueError(f"unknown parser '{name}'; available: {sorted(available_parsers())}")


# --------------------------------------------------------------------------- #
# self-check helpers (used by tests and the benchmark)
# --------------------------------------------------------------------------- #
def reading_order_ok(elements: Sequence[Element]) -> float:
    """Share of consecutive elements that keep a non-decreasing (page, y) order."""
    pairs = list(zip(elements, elements[1:]))
    if not pairs:
        return 1.0
    good = 0
    for a, b in pairs:
        ay = (a.bbox or [0, 0, 0, 0])[1]
        by = (b.bbox or [0, 0, 0, 0])[1]
        if (b.page, by) >= (a.page, ay):
            good += 1
    return good / len(pairs)


def element_type_counts(elements: Iterable[Element]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for el in elements:
        counts[el.element_type] = counts.get(el.element_type, 0) + 1
    return dict(sorted(counts.items()))


def validate_elements(elements: Iterable[Element]) -> list[str]:
    """Return a list of contract violations (empty means the stream is valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for el in elements:
        if el.element_type not in ELEMENT_TYPES:
            problems.append(f"{el.element_id}: unknown element_type '{el.element_type}'")
        if el.element_id in seen:
            problems.append(f"{el.element_id}: duplicate element_id")
        seen.add(el.element_id)
        if not el.doc_id or not el.revision:
            problems.append(f"{el.element_id}: missing doc_id/revision")
        if el.page < 1:
            problems.append(f"{el.element_id}: page must be >= 1")
        if el.bbox and (len(el.bbox) != 4 or el.bbox[2] < el.bbox[0] or el.bbox[3] < el.bbox[1]):
            problems.append(f"{el.element_id}: malformed bbox {el.bbox}")
    return problems


__all__ = [
    "ParseOptions", "RawLine", "RawBlock", "PymupdfParser", "PymupdfPlainParser",
    "get_parser", "available_parsers", "classify_block", "heading_level", "parse_heading",
    "validate_elements", "element_type_counts", "reading_order_ok", "open_pdf", "page_text_lines",
    "group_blocks", "_join_lines", "new_id",
]
