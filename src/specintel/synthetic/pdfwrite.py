"""A tiny, dependency-free PDF writer (test/demo utility).

Why this exists
---------------
The assignment corpus is licensed (USB-IF / WPC) and cannot be committed.  The
pipeline must still be runnable from a clean clone, so ``specintel.synthetic``
generates a stand-in corpus of *real PDFs* — vector text, vector diagrams and
raster images — using this writer.  Everything downstream (PyMuPDF parsing,
figure cropping, table stitching, OCR adapters) then runs on genuine PDF files
rather than on hand-written JSON.

Supported operations: text with the standard Helvetica/Courier base fonts,
rectangles, lines, and Flate-encoded raster images.  Coordinates are given
top-left origin like PyMuPDF and converted internally to PDF's bottom-left
origin.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

PAGE_W = 612.0
PAGE_H = 792.0
MARGIN_L = 72.0
MARGIN_R = 72.0
MARGIN_T = 72.0
MARGIN_B = 72.0
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

FONTS = {"regular": "F1", "bold": "F2", "mono": "F3"}
FONT_SIZES = {"regular": 9.0, "bold": 9.0, "mono": 8.0}
# Real Helvetica advance widths (units/1000 em, from the Adobe AFM metrics).
# The first version of this writer used a single average advance, which made
# wrapped lines a character or two wider than the estimate: the text then ran
# past the column rule it was supposed to fit inside, and a grid-based table
# reader assigned the overflow to the neighbouring cell ("Configuratio" + "n"
# instead of "Configuration"), corrupting the table.  Measuring per character
# and leaving a deliberate 4% safety margin keeps every line inside its column.
_HELVETICA = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015,
    "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333, "{": 334,
    "|": 260, "}": 334, "~": 584, "\u00b0": 400, "\u00b5": 556, "\u03a9": 776,
    "\u2013": 556, "\u2014": 1000, "\u2018": 222, "\u2019": 222, "\u201c": 333,
    "\u201d": 333, "\u2192": 584, "\u2264": 549, "\u2265": 549, "\u00d7": 584,
    "\u00a9": 737, "\u00a7": 556, "\u2022": 350, "\u2026": 1000,
}
_HELVETICA_BOLD = dict(_HELVETICA)
_HELVETICA_BOLD.update({
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722, "'": 238,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    ":": 333, ";": 333, "<": 584, "=": 584, ">": 584, "?": 611, "@": 975,
    "[": 333, "\\": 278, "]": 333, "^": 584, "_": 556, "`": 333, "{": 389,
    "|": 280, "}": 389, "~": 584,
})
for _c in "abcdefghijklmnopqrstuvwxyz":
    _HELVETICA_BOLD[_c] = {
        "a": 556, "b": 611, "c": 556, "d": 611, "e": 556, "f": 333, "g": 611, "h": 611,
        "i": 278, "j": 278, "k": 556, "l": 278, "m": 889, "n": 611, "o": 611, "p": 611,
        "q": 611, "r": 389, "s": 556, "t": 333, "u": 611, "v": 556, "w": 778, "x": 556,
        "y": 556, "z": 500}[_c]
for _c, _w in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                  [722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
                   667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611]):
    _HELVETICA_BOLD[_c] = _w
for _c in "0123456789":
    _HELVETICA[_c] = 556
    _HELVETICA_BOLD[_c] = 556
for _c, _w in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                  [667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
                   667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611]):
    _HELVETICA[_c] = _w

_WIDTH_TABLES = {"regular": _HELVETICA, "bold": _HELVETICA_BOLD}
SAFETY = 1.04                     # keep lines inside their column rule


def _esc(text: str) -> str:
    return (text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
                .replace("\r", "").replace("\n", " "))


def char_width(ch: str, font: str = "regular") -> float:
    """Advance width of one character in units/1000 em."""
    table = _WIDTH_TABLES.get(font, _HELVETICA)
    return table.get(ch, _HELVETICA.get(ch, 556))


def text_width(text: str, size: float, font: str = "regular") -> float:
    """Width in points of ``text`` when drawn at ``size`` in the given font."""
    return sum(char_width(ch, font) for ch in str(text)) * size / 1000.0 * SAFETY


def wrap(text: str, size: float = 9.0, font: str = "regular", width: float = TEXT_W) -> list[str]:
    """Greedy word wrap using the average-advance estimate."""
    out: list[str] = []
    for para in str(text).split("\n"):
        words, line = para.split(), ""
        for w in words:
            candidate = f"{line} {w}".strip()
            if text_width(candidate, size, font) <= width or not line:
                line = candidate
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


@dataclass
class ImageAsset:
    """A raster image emitted as a PDF image XObject."""

    name: str
    width: int
    height: int
    rgb: bytes          # len == width*height*3

    @staticmethod
    def from_pattern(width: int, height: int, seed: int = 7, kind: str = "boxes") -> "ImageAsset":
        """Deterministic synthetic raster (stand-in for scanned Tech 2 figures)."""
        px = bytearray()
        for y in range(height):
            for x in range(width):
                if kind == "boxes":
                    edge = ((x // 24) + (y // 16) + seed) % 2
                    on = (x % 24 < 2) or (y % 16 < 2)
                    v = 40 if on else (230 if edge else 120)
                else:  # gradient + noise
                    v = (x * 3 + y * 2 + seed * 11) % 255
                px += bytes((v, v, v))
        return ImageAsset(name=f"Im{abs(hash((width, height, seed, kind))) % 10000}",
                          width=width, height=height, rgb=bytes(px))


@dataclass
class Page:
    """One PDF page: a sequence of drawing operations in top-left coordinates."""

    ops: list[str] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    text_runs: list[tuple[float, float, str, str, float]] = field(default_factory=list)  # x,y,text,font,size

    # -- text ------------------------------------------------------------- #
    def text(self, x: float, y: float, s: str, *, font: str = "regular", size: float | None = None) -> "Page":
        size = size or FONT_SIZES.get(font, 9.0)
        pdf_y = PAGE_H - y - size
        self.ops.append(f"BT /{FONTS[font]} {size} Tf 1 0 0 1 {x:.2f} {pdf_y:.2f} Tm ({_esc(s)}) Tj ET")
        self.text_runs.append((x, y, s, font, size))
        return self

    def paragraph(self, x: float, y: float, s: str, *, font: str = "regular", size: float | None = None,
                  width: float = TEXT_W, leading: float = 11.5) -> float:
        """Write a wrapped paragraph; returns the y coordinate after the last line."""
        size = size or FONT_SIZES.get(font, 9.0)
        for i, line in enumerate(wrap(s, size, font, width)):
            self.text(x, y + i * leading, line, font=font, size=size)
        return y + max(1, len(wrap(s, size, font, width))) * leading

    # -- vector ----------------------------------------------------------- #
    def rect(self, x: float, y: float, w: float, h: float, *, fill: float | None = None,
             stroke: float = 0.0, line_width: float = 0.7) -> "Page":
        pdf_y = PAGE_H - y - h
        if fill is not None:
            self.ops.append(f"{fill:.2f} g {x:.2f} {pdf_y:.2f} {w:.2f} {h:.2f} re f")
        self.ops.append(f"{stroke:.2f} G {line_width:.2f} w {x:.2f} {pdf_y:.2f} {w:.2f} {h:.2f} re S")
        return self

    def line(self, x0: float, y0: float, x1: float, y1: float, *, gray: float = 0.0,
             line_width: float = 0.7, dash: bool = False) -> "Page":
        d = " [3 2] 0 d" if dash else ""
        self.ops.append(
            f"{gray:.2f} G {line_width:.2f} w{d} {x0:.2f} {PAGE_H - y0:.2f} m {x1:.2f} {PAGE_H - y1:.2f} l S"
        )
        if dash:
            self.ops.append("[] 0 d")
        return self

    def arrow(self, x0: float, y0: float, x1: float, y1: float, *, label: str = "",
              label_dy: float = -7.0) -> "Page":
        self.line(x0, y0, x1, y1)
        import math

        ang = math.atan2(y1 - y0, x1 - x0)
        for delta in (2.6, -2.6):
            hx = x1 - 8 * math.cos(ang + delta)
            hy = y1 - 8 * math.sin(ang + delta)
            self.line(x1, y1, hx, hy, line_width=0.7)
        if label:
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            self.text(mid_x + 3, mid_y + label_dy, label, size=7.0)
        return self

    def image(self, x: float, y: float, w: float, h: float, asset: ImageAsset) -> "Page":
        if asset not in self.images:
            self.images.append(asset)
        self.ops.append(
            f"q {w:.2f} 0 0 {h:.2f} {x:.2f} {PAGE_H - y - h:.2f} cm /{asset.name} Do Q"
        )
        return self


class PdfWriter:
    """Assembles pages into a PDF 1.4 file."""

    def __init__(self, title: str = "untitled", author: str = "specintelligence-synthetic") -> None:
        self.title = title
        self.author = author
        self.pages: list[Page] = []

    def add_page(self, page: Page | None = None) -> Page:
        page = page or Page()
        self.pages.append(page)
        return page

    # -- serialisation ---------------------------------------------------- #
    def to_bytes(self) -> bytes:
        objects: list[bytes] = []          # index i -> object number i+1

        def add(obj: bytes) -> int:
            objects.append(obj)
            return len(objects)

        font_ids = {k: add(f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} /Encoding /WinAnsiEncoding >>".encode())
                    for k, base in (("F1", "Helvetica"), ("F2", "Helvetica-Bold"), ("F3", "Courier"))}

        # images are shared across pages: allocate XObjects first
        unique_images: dict[str, ImageAsset] = {}
        for page in self.pages:
            for asset in page.images:
                unique_images.setdefault(asset.name, asset)
        image_ids: dict[str, int] = {}
        for name, asset in unique_images.items():
            comp = zlib.compress(asset.rgb, 6)
            head = (f"<< /Type /XObject /Subtype /Image /Width {asset.width} /Height {asset.height} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(comp)} >>")
            img_obj_num = len(objects) + 1
            objects.append(f"{head}\nstream\n".encode() + comp + b"\nendstream")
            image_ids[name] = img_obj_num

        pages_obj_num = add(b"placeholder")          # reserve, patched later
        page_obj_nums: list[int] = []
        for page in self.pages:
            content = "\n".join(page.ops).encode("cp1252", "replace")
            comp = zlib.compress(content, 6)
            content_num = len(objects) + 1
            objects.append(
                f"<< /Filter /FlateDecode /Length {len(comp)} >>\nstream\n".encode() + comp + b"\nendstream"
            )
            xobjects = " ".join(f"/{n} {image_ids[n]} 0 R" for n in image_ids if _page_has(page, n))
            resources = (
                "<< /Font << " + " ".join(f"/{k} {v} 0 R" for k, v in font_ids.items()) + " >>"
                + (f" /XObject << {xobjects} >>" if xobjects else "") + " >>"
            )
            page_num = len(objects) + 1
            objects.append(
                f"<< /Type /Page /Parent {pages_obj_num} 0 R /MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                f"/Resources {resources} /Contents {content_num} 0 R >>".encode()
            )
            page_obj_nums.append(page_num)

        kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
        objects[pages_obj_num - 1] = (
            f"<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>".encode()
        )
        catalog_num = len(objects) + 1
        objects.append(f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode())
        info_num = len(objects) + 1
        objects.append(f"<< /Title ({_esc(self.title)}) /Author ({_esc(self.author)}) "
                       f"/Producer (specintel.synthetic.pdfwrite) >>".encode())

        # -- write file with xref ----------------------------------------- #
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = []
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_num} 0 R /Info {info_num} 0 R >>\n"
                f"startxref\n{xref_at}\n%%EOF\n").encode()
        return bytes(out)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return p


def _page_has(page: Page, image_name: str) -> bool:
    return any(asset.name == image_name for asset in page.images)


# --------------------------------------------------------------------------- #
# layout helpers used by the synthetic corpus and by figure tests
# --------------------------------------------------------------------------- #
def draw_boxes_diagram(page: Page, x: float, y: float, w: float, h: float,
                       nodes: Sequence[tuple[str, float, float, float, float]],
                       edges: Iterable[tuple[int, int, str]] = (),
                       title: str = "") -> None:
    """Draw a state-machine style diagram inside ``(x, y, w, h)``.

    ``nodes`` are ``(label, rel_x, rel_y, rel_w, rel_h)`` in [0,1] coordinates
    of the box.  ``edges`` are ``(src_index, dst_index, label)``.
    """
    page.rect(x, y, w, h, fill=0.98, stroke=0.6)
    if title:
        page.text(x + 4, y + 4, title, font="bold", size=8.0)
    centres: list[tuple[float, float]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for label, rx, ry, rw, rh in nodes:
        bx, by = x + rx * w, y + ry * h
        bw, bh = rw * w, rh * h
        page.rect(bx, by, bw, bh, fill=1.0, stroke=0.0, line_width=1.0)
        # labels are written word-per-line, mimicking real vector diagrams where
        # state names break across lines when extracted
        lines = label.split()
        for i, line in enumerate(lines[:3]):
            page.text(bx + 4, by + 6 + i * 9, line, size=7.5)
        centres.append((bx + bw / 2, by + bh / 2))
        boxes.append((bx, by, bw, bh))
    for src, dst, label in edges:
        x0, y0 = centres[src]
        x1, y1 = centres[dst]
        page.arrow(x0, y0, x1, y1, label=label)
