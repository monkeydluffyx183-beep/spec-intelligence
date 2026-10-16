"""WS1 — OCR adapters, including Unlimited-OCR, and the block-type mapping.

The assignment makes Unlimited-OCR the OCR candidate for the text path and for
the capstone's OCR-then-text baseline, and requires a benchmark against the
layout parsers (Task 2, Step 2).  This module is the adapter layer:

``UnlimitedOCRBackend``
    Talks to a local vLLM / SGLang server that serves ``baidu/Unlimited-OCR``
    over an OpenAI-compatible API.  It supports **single-page** and
    **multi-page (one-shot)** passes: several page images are sent in one
    request and the model returns one document, which is the behaviour that
    Reference Sliding Window Attention makes possible.  The multi-page window
    size is configurable so the long-horizon table (1 / 5 / 10 / 20+ pages per
    pass) can be measured.
``DeepSeekOCRBackend``
    The optional further reference system (single page per request).
``MockOCRBackend``
    Deterministic stand-in that emits the *same block/provenance structure*
    using the embedded text layer.  It exists so the benchmark, the element
    mapping and the pipeline are runnable without a GPU: it is honest about
    what a text-layer reader cannot do (raster figures come back empty), and
    the benchmark report marks it as a stand-in rather than a model result.

Every backend returns :class:`OCRResult`; ``map_blocks_to_elements`` turns the
block types and bounding boxes into the shared element contract, keeping the
page-level provenance the assignment asks for.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.models import Element, new_id
from specintel.common.resources import gpu_memory_mb, rss_mb

# Layout type vocabulary of the model, mapped onto our element types.
LAYOUT_TYPE_MAP = {
    "title": "title",
    "section_header": "section_heading",
    "section_heading": "section_heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list": "list_item",
    "list_item": "list_item",
    "table": "table",
    "table_caption": "table_caption",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "image": "figure",
    "note": "note",
    "code": "code",
    "header": "header",
    "footer": "footer",
    "caption": "figure_caption",
    "toc": "toc",
}

MARKER_RE = re.compile(r"<\|(?P<key>layout|type|bbox|page)\s*:\s*(?P<value>[^|>]+)\|>", re.IGNORECASE)
CONTINUED_RE = re.compile(r"\(?\s*continued\s*\)?", re.IGNORECASE)


@dataclass
class OCRBlock:
    """One block returned by an OCR model, with provenance."""

    page: int
    text: str
    layout_type: str = "text"
    bbox: list[float] | None = None
    confidence: float = 1.0

    @property
    def element_type(self) -> str:
        return LAYOUT_TYPE_MAP.get(self.layout_type.lower(), "paragraph")


@dataclass
class OCRPass:
    """One model call and what it produced (used for the long-horizon table)."""

    pages: list[int]
    duration_s: float
    blocks: int
    chars: int
    peak_rss_mb: float = 0.0
    peak_gpu_mb: float | None = None
    note: str = ""


@dataclass
class OCRResult:
    """Everything one backend produced for a document or a page range."""

    backend: str
    blocks: list[OCRBlock] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    passes: list[OCRPass] = field(default_factory=list)
    text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(p.duration_s for p in self.passes)

    @property
    def pages_per_second(self) -> float:
        total_pages = sum(len(p.pages) for p in self.passes)
        return total_pages / self.duration_s if self.duration_s > 0 else 0.0

    def to_elements(self, doc_id: str, revision: str) -> list[Element]:
        return map_blocks_to_elements(self.blocks, doc_id=doc_id, revision=revision, extractor=self.backend)

    def plain_text(self, page: int | None = None) -> str:
        return "\n".join(b.text for b in self.blocks if page is None or b.page == page)


# --------------------------------------------------------------------------- #
# marker parsing
# --------------------------------------------------------------------------- #
def parse_marked_output(raw: str, *, start_page: int = 1) -> list[OCRBlock]:
    """Parse the layout markers Unlimited-OCR emits into typed blocks.

    The model marks each block with its layout type and bounding box; keeping
    those markers (instead of stripping them) is what preserves page-level
    provenance, which the assignment calls out explicitly.

    Two accepted shapes:

    * one marker line per block:  ``<|layout:table|><|bbox:72,96,540,240|> ...content...``
    * inline markers inside a paragraph of text, where a marker starts a block
      and the next marker ends it.
    """
    blocks: list[OCRBlock] = []
    if not raw:
        return blocks
    page = start_page
    # split into marker-delimited segments while tracking the current metadata
    tokens = MARKER_RE.split(raw)
    meta: dict[str, str] = {}
    buffer: list[str] = []
    order_meta: dict[str, str] = {}

    def flush() -> None:
        nonlocal meta, buffer, page
        text = "".join(buffer).strip()
        if text:
            blocks.append(OCRBlock(page=page, text=text,
                                   layout_type=meta.get("layout") or meta.get("type") or "text",
                                   bbox=_parse_bbox(meta.get("bbox"))))
        buffer = []

    idx = 0
    while idx < len(tokens):
        chunk = tokens[idx]
        if idx % 3 == 0:                       # plain text between markers
            buffer.append(chunk)
            idx += 1
            continue
        key, value = tokens[idx], tokens[idx + 1] if idx + 1 < len(tokens) else ""
        if key.lower() == "page":
            flush()
            try:
                page = int(value.strip())
            except ValueError:
                pass
        else:
            flush()
            order_meta[key.lower()] = value.strip()
            meta = dict(order_meta)
        idx += 2
    flush()
    return blocks


def _parse_bbox(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    parts = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(parts) < 4:
        return None
    x0, y0, x1, y1 = (float(p) for p in parts[:4])
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def strip_markers(raw: str) -> str:
    """Plain text with layout markers removed (the OCR-then-text baseline)."""
    return MARKER_RE.sub("", raw)


# --------------------------------------------------------------------------- #
# mapping to the element contract
# --------------------------------------------------------------------------- #
def map_blocks_to_elements(blocks: Sequence[OCRBlock], *, doc_id: str, revision: str,
                           extractor: str) -> list[Element]:
    """Map OCR blocks to typed elements, keeping page and bbox provenance."""
    elements: list[Element] = []
    section_number, section_title = "", ""
    for i, block in enumerate(blocks, start=1):
        etype = block.element_type
        text = block.text.strip()
        if not text:
            continue
        if etype in ("section_heading", "title"):
            section_number, section_title = _split_section(text, section_title)
        elements.append(Element(
            element_id=f"{doc_id}:{extractor}:{block.page}:{i}",
            doc_id=doc_id, revision=revision, element_type=etype, page=block.page, text=text,
            markdown=text, bbox=block.bbox, order=i, section_number=section_number,
            section_title=section_title, extractor=extractor,
            meta={"ocr_layout_type": block.layout_type, "ocr_confidence": block.confidence},
        ))
    return elements


def _split_section(text: str, fallback_title: str) -> tuple[str, str]:
    match = re.match(r"^(?P<number>(?:\d+(?:\.\d+)*)|[A-Z](?:\.\d+)*)\s+(?P<title>.+)$", text)
    if match:
        return match.group("number"), match.group("title").strip()
    return "", text or fallback_title


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class OCRBackend:
    """Interface: turn page images (or a PDF) into typed blocks."""

    name = "base"
    needs_gpu = False

    def parse_pages(self, images: Sequence[str | Path], *, doc_id: str, start_page: int = 1,
                    multi_page: bool = False, multi_page_window: int = 8) -> OCRResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "needs_gpu": self.needs_gpu}


class MockOCRBackend(OCRBackend):
    """Deterministic stand-in used when no GPU is available.

    It emits the same block/provenance structure as the real model by reading
    the PDF text layer of the rendered pages.  Consequences, reported honestly
    by the benchmark: text accuracy on born-digital pages is high, but content
    that exists only as pixels (raster Tech 2 figures) comes back empty, and
    the multi-page pass cannot show the real model's memory behaviour.
    """

    name = "mock-ocr"
    needs_gpu = False

    def __init__(self, pdf_paths: Sequence[str | Path] | None = None) -> None:
        self.pdf_paths = [Path(p) for p in (pdf_paths or [])]

    def parse_pages(self, images: Sequence[str | Path], *, doc_id: str, start_page: int = 1,
                    multi_page: bool = False, multi_page_window: int = 8) -> OCRResult:
        from specintel.ingest.parser import open_pdf, page_text_lines

        result = OCRResult(backend=self.name)
        result.notes.append("deterministic OCR stand-in: reads the embedded text layer; "
                           "run baidu/Unlimited-OCR on a GPU for real OCR")
        pdf = None
        for path in self.pdf_paths:
            if doc_id in Path(path).name or len(self.pdf_paths) == 1:
                pdf = path
                break
        pdf = pdf or (self.pdf_paths[0] if self.pdf_paths else None)
        if pdf is None:
            result.notes.append("no PDF available for the stand-in backend")
            return result
        doc = open_pdf(pdf)
        pages = [int(p) for p in images] if images and all(str(p).isdigit() for p in images) else list(
            range(start_page, start_page + doc.page_count))
        window = max(1, multi_page_window if multi_page else 1)
        for i in range(0, len(pages), window):
            batch = pages[i:i + window]
            t0 = time.perf_counter()
            chars = 0
            for pno in batch:
                if not (1 <= pno <= doc.page_count):
                    continue
                for line in page_text_lines(doc[pno - 1], pno):
                    block = OCRBlock(page=pno, text=line.text,
                                     layout_type="text", bbox=line.bbox, confidence=1.0)
                    result.blocks.append(block)
                    chars += len(block.text)
            result.passes.append(OCRPass(pages=list(batch), duration_s=time.perf_counter() - t0,
                                         blocks=sum(1 for b in result.blocks if b.page in batch), chars=chars,
                                         peak_rss_mb=rss_mb(), peak_gpu_mb=gpu_memory_mb(),
                                         note=f"window={window}"))
        doc.close()
        result.pages = list(pages)
        result.text = result.plain_text()
        return result


class UnlimitedOCRBackend(OCRBackend):
    """Adapter for ``baidu/Unlimited-OCR`` served locally (vLLM / SGLang).

    The model is about 3B parameters, built on DeepSeek-OCR and needs an NVIDIA
    GPU.  Two call shapes are supported:

    * single page — one image per request;
    * multi-page one-shot — ``multi_page_window`` consecutive page images in a
      single request, which is where the model's constant-memory long-horizon
      behaviour shows up.

    Never point this at a public demo endpoint with licensed spec pages; run it
    on your own hardware.
    """

    name = "unlimited-ocr"
    needs_gpu = True

    PROMPT = (
        "Parse the attached specification page(s) into a single document. For every block emit a marker line "
        "in the form <|layout:TYPE|><|bbox:x0,y0,x1,y1|><|page:N|> followed by the block content on the "
        "following lines. Use the layout types: title, section_header, text, list, table, table_caption, "
        "figure, figure_caption, note, header, footer. Keep tables as Markdown and keep every cell."
    )

    def __init__(self, endpoint: str = "http://localhost:8000", model: str = "baidu/Unlimited-OCR",
                 timeout_s: int = 300, max_tokens: int = 8192, dpi: int = 200) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.dpi = dpi

    def parse_pages(self, images: Sequence[str | Path], *, doc_id: str, start_page: int = 1,
                    multi_page: bool = False, multi_page_window: int = 8) -> OCRResult:
        result = OCRResult(backend=self.name)
        window = max(1, multi_page_window if multi_page else 1)
        images = list(images)
        for i in range(0, len(images), window):
            batch = images[i:i + window]
            t0 = time.perf_counter()
            try:
                raw = self._call(batch, start_page + i)
            except Exception as exc:                       # network/model failure is data, not a crash
                result.notes.append(f"pass failed for pages {i + 1}-{i + len(batch)}: {exc}")
                continue
            blocks = parse_marked_output(raw, start_page=start_page + i)
            result.blocks.extend(blocks)
            result.passes.append(OCRPass(pages=list(range(start_page + i, start_page + i + len(batch))),
                                         duration_s=time.perf_counter() - t0, blocks=len(blocks),
                                         chars=len(raw), peak_rss_mb=rss_mb(), peak_gpu_mb=gpu_memory_mb(),
                                         note=f"window={len(batch)}"))
        result.pages = list(range(start_page, start_page + len(images)))
        result.text = strip_markers(result.plain_text())
        return result

    def _call(self, batch: Sequence[str | Path], page_offset: int) -> str:
        import requests

        content: list[dict[str, Any]] = [{"type": "text", "text": self.PROMPT}]
        for path in batch:
            p = Path(path)
            if not p.exists():
                continue
            encoded = base64.b64encode(p.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0.0, "max_tokens": self.max_tokens}
        resp = requests.post(f"{self.endpoint}/v1/chat/completions", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class DeepSeekOCRBackend(UnlimitedOCRBackend):
    """Optional reference system: the base OCR model, one page per request."""

    name = "deepseek-ocr"

    def __init__(self, endpoint: str = "http://localhost:8000", model: str = "deepseek-ai/DeepSeek-OCR",
                 **kwargs: Any) -> None:
        super().__init__(endpoint=endpoint, model=model, **kwargs)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def probe(endpoint: str, timeout_s: float = 1.5) -> bool:
    try:
        import requests

        requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=timeout_s)
        return True
    except Exception:
        return False


def get_ocr_backend(config: Any | None = None, backend: str = "auto", *,
                    pdf_paths: Sequence[str | Path] | None = None) -> OCRBackend:
    """Resolve an OCR backend; ``auto`` prefers a reachable local server."""
    endpoint = "http://localhost:8000"
    model = "baidu/Unlimited-OCR"
    if config is not None:
        endpoint = str(config.get("ingest.ocr.endpoint", endpoint))
        model = str(config.get("ingest.ocr.model", model))
        if backend == "auto":
            backend = str(config.get("ingest.ocr.backend", "auto"))
    if backend in ("mock", "none"):
        return MockOCRBackend(pdf_paths)
    if backend == "auto":
        if probe(endpoint):
            return UnlimitedOCRBackend(endpoint=endpoint, model=model)
        return MockOCRBackend(pdf_paths)
    if backend == "unlimited-ocr":
        return UnlimitedOCRBackend(endpoint=endpoint, model=model)
    if backend == "deepseek-ocr":
        return DeepSeekOCRBackend(endpoint=endpoint)
    raise ValueError(f"unknown OCR backend '{backend}'")


__all__ = [
    "OCRBlock", "OCRPass", "OCRResult", "OCRBackend", "MockOCRBackend", "UnlimitedOCRBackend",
    "DeepSeekOCRBackend", "get_ocr_backend", "parse_marked_output", "map_blocks_to_elements",
    "strip_markers", "LAYOUT_TYPE_MAP", "new_id", "json",
]
