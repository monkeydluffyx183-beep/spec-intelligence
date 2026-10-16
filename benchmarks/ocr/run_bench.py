#!/usr/bin/env python3
"""WS1 — the OCR / document-AI benchmark (Task 2, Step 4).

Compares the text-recovery backends on exactly the same crops and reports:

* **character error rate** and **word error rate** against the ground-truth text
  recorded when the page was written (``manifest.json``);
* **equations / numbers**: the share of numeric values recovered exactly, which is
  what matters for a specification — a missed decimal point is a wrong timer;
* **latency** and **memory** per page;
* **availability**: whether the backend can run on this machine at all.

Backends
--------
``pymupdf-text``
    The vector text layer.  Instant and exact on USB-PD, Type-C and USB4 PDFs, and
    useless on a scanned page or a figure whose labels were rasterised.  This is
    the baseline every other backend must beat to be worth its cost.
``pymupdf-raster+ocr``
    Rasterise the crop at the configured DPI and run Tesseract.  Runs on CPU;
    degrades on small fonts and on the vector figures whose labels are scattered.
``rapidocr``
    Optional ONNX detector + recogniser, better on dense figure labels.  Skipped
    when the package or its models are not installed.
``unlimited-ocr``
    ``baidu/Unlimited-OCR`` (~3 B parameters, DeepSeek-OCR architecture, MIT) served
    through vLLM on an NVIDIA GPU.  This is the backend the assignment names for
    figure text; the harness probes its endpoint and reports it as unavailable
    rather than pretending, because this repository has no GPU and the model is
    never downloaded automatically.  ``--unlimited-ocr-endpoint`` points the
    harness at a machine that has one.
``noop``
    Returns nothing.  It is a control: it turns "the figure has no text" into a
    measured zero so the content-type breakdown can show what a missing OCR pass
    costs instead of hiding it.

Usage::

    PYTHONPATH=src python3 benchmarks/ocr/run_bench.py --docs pd,typec --limit 6
    PYTHONPATH=src python3 benchmarks/ocr/run_bench.py --unlimited-ocr-endpoint http://gpu:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from specintel.common.config import load_config                                    # noqa: E402
from specintel.common.logging import get_logger, log_event, setup_from_config      # noqa: E402
from specintel.common.resources import hardware_profile, rss_mb                    # noqa: E402

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance, iterative (rows are the shorter sequence)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(min(previous[j] + 1,                       # deletion
                               current[j - 1] + 1,                    # insertion
                               previous[j - 1] + (item_a != item_b)))  # substitution
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference, hypothesis = reference.strip(), hypothesis.strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = WORD_RE.findall(reference.lower())
    hyp_words = WORD_RE.findall(hypothesis.lower())
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def word_coverage(reference: str, hypothesis: str) -> tuple[float, float]:
    """Order-insensitive content recall and spurious-word rate.

    Character error rate is order-sensitive, and text read out of a vector figure
    arrives in a different order than the page reads (labels interleave, columns
    come back column by column).  That is a real defect for a reader, but it is
    not the same defect as *missing* or *inventing* words, so both are measured:
    ``recall`` is the share of the reference's words the backend produced and
    ``spurious`` is the share of its output that the reference does not contain.
    """
    ref_words = [w.lower() for w in WORD_RE.findall(reference)]
    hyp_words = [w.lower() for w in WORD_RE.findall(hypothesis)]
    if not ref_words or not hyp_words:
        return (0.0 if ref_words else 1.0), (1.0 if hyp_words else 0.0)
    ref_set, hyp_set = set(ref_words), set(hyp_words)
    recall = sum(1 for w in ref_set if w in hyp_set) / len(ref_set)
    spurious = sum(1 for w in hyp_set if w not in ref_set) / len(hyp_set)
    return recall, spurious


def number_accuracy(reference: str, hypothesis: str) -> float:
    """Share of the reference's numbers that come back exactly.

    Decimal points, unit-less digits and ranges are the substance of a
    specification; a backend that loses them is not usable here even with a low
    character error rate.
    """
    ref_numbers = NUMBER_RE.findall(reference)
    if not ref_numbers:
        return float("nan")
    hyp = set(NUMBER_RE.findall(hypothesis))
    return sum(1 for n in ref_numbers if n in hyp) / len(ref_numbers)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    """One benchmarked region: a figure crop or a full page."""

    crop_id: str
    doc_id: str
    page: int
    kind: str                       # figure | page
    image_path: str
    reference_text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    backend: str
    text: str
    latency_s: float
    rss_mb: float
    skipped: str = ""               # why the backend could not run, if it could not


def _pymupdf_text(crop: Crop, config: Any, **_: Any) -> Prediction:
    import pymupdf

    start = time.perf_counter()
    with pymupdf.open(str(crop.meta["pdf"])) as doc:
        page = doc[max(0, crop.page - 1)]
        bbox = crop.meta.get("bbox")
        if crop.kind == "figure" and bbox:
            text = page.get_text("text", clip=pymupdf.Rect(*bbox))
        else:
            text = page.get_text("text")
    return Prediction("pymupdf-text", text, time.perf_counter() - start, rss_mb())


def _pymupdf_raster_ocr(crop: Crop, config: Any, dpi: int = 220, **_: Any) -> Prediction:
    """Rasterise the crop, then OCR it with Tesseract through pytesseract."""
    import pymupdf

    try:
        import pytesseract
        from PIL import Image
        import io
    except Exception as exc:                     # pragma: no cover - optional dependency
        return Prediction("pymupdf-raster+ocr", "", 0.0, rss_mb(), skipped=f"not installed: {exc}")
    if not str(crop.image_path) or not Path(crop.image_path).exists():
        return Prediction("pymupdf-raster+ocr", "", 0.0, rss_mb(), skipped="no crop image")
    start = time.perf_counter()
    image = Image.open(crop.image_path)
    text = pytesseract.image_to_string(image)
    del Image, io, pymupdf
    return Prediction("pymupdf-raster+ocr", text, time.perf_counter() - start, rss_mb())


def _rapidocr(crop: Crop, config: Any, **_: Any) -> Prediction:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:                     # pragma: no cover - optional dependency
        return Prediction("rapidocr", "", 0.0, rss_mb(), skipped=f"not installed: {exc}")
    if not Path(crop.image_path).exists():
        return Prediction("rapidocr", "", 0.0, rss_mb(), skipped="no crop image")
    start = time.perf_counter()
    engine = RapidOCR()
    result, _ = engine(crop.image_path)
    text = "\n".join(line[1] for line in (result or []))
    return Prediction("rapidocr", text, time.perf_counter() - start, rss_mb())


def _unlimited_ocr(crop: Crop, config: Any, endpoint: str = "", **_: Any) -> Prediction:
    """``baidu/Unlimited-OCR`` over a vLLM OpenAI-compatible endpoint.

    The model is a ~3 B-parameter vision-language model and needs a GPU; this
    harness never downloads or starts it.  Point ``--unlimited-ocr-endpoint`` (or
    ``SPECINTEL_UNLIMITED_OCR_ENDPOINT``) at a running server and the same
    comparisons run against it.  Anything else is reported as unavailable: an
    honest "not measured on this machine" beats a fabricated number.
    """
    endpoint = endpoint or os.environ.get("SPECINTEL_UNLIMITED_OCR_ENDPOINT", "")
    if not endpoint:
        return Prediction("unlimited-ocr", "", 0.0, rss_mb(),
                          skipped="no endpoint configured (needs an NVIDIA GPU; "
                                  "see README § Unlimited-OCR)")
    if not Path(crop.image_path).exists():
        return Prediction("unlimited-ocr", "", 0.0, rss_mb(), skipped="no crop image")
    try:
        import base64
        import urllib.request

        payload = json.dumps({
            "model": os.environ.get("SPECINTEL_UNLIMITED_OCR_MODEL", "Unlimited-OCR"),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "OCR this specification figure. Return the text only."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(
                    Path(crop.image_path).read_bytes()).decode()}}]}],
            "temperature": 0.0, "max_tokens": 1024,
        }).encode()
        request = urllib.request.Request(f"{endpoint.rstrip('/')}/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode())
        latency = time.perf_counter() - start
        text = body["choices"][0]["message"]["content"]
        return Prediction("unlimited-ocr", text, latency, rss_mb())
    except Exception as exc:                      # pragma: no cover - network/GPU path
        return Prediction("unlimited-ocr", "", 0.0, rss_mb(), skipped=f"endpoint failed: {exc}")


def _noop(crop: Crop, config: Any, **_: Any) -> Prediction:
    """Control backend: recovers nothing, so a missing OCR pass is scored, not hidden."""
    return Prediction("noop", "", 0.0, rss_mb())


BACKENDS: dict[str, Callable[..., Prediction]] = {
    "pymupdf-text": _pymupdf_text,
    "pymupdf-raster+ocr": _pymupdf_raster_ocr,
    "rapidocr": _rapidocr,
    "unlimited-ocr": _unlimited_ocr,
    "noop": _noop,
}


def backends_available(config: Any) -> dict[str, bool]:
    """Which backends could run here: true when their dependency is present.

    Used by the test suite and by the report so "unavailable" is a fact about the
    machine, not an omission.
    """
    available = {"pymupdf-text": False, "pymupdf-raster+ocr": False, "rapidocr": False,
                 "unlimited-ocr": False, "noop": True}
    try:
        import pymupdf                                       # noqa: F401
        available["pymupdf-text"] = True
    except Exception:                                        # pragma: no cover
        pass
    for name, module in (("pymupdf-raster+ocr", "pytesseract"), ("rapidocr", "rapidocr_onnxruntime")):
        try:
            __import__(module)
            available[name] = True
        except Exception:                                    # pragma: no cover
            available[name] = False
    if os.environ.get("SPECINTEL_UNLIMITED_OCR_ENDPOINT"):
        available["unlimited-ocr"] = True
    else:
        try:                                                 # a local vLLM on the default port
            import socket

            with socket.create_connection(("127.0.0.1", 8000), timeout=0.2):
                available["unlimited-ocr"] = True
        except Exception:
            available["unlimited-ocr"] = False
    return available


# --------------------------------------------------------------------------- #
# building the crop set
# --------------------------------------------------------------------------- #
def build_crops(config: Any, docs: Sequence[str], *, limit: int | None = None,
                kinds: Sequence[str] = ("figure", "page")) -> list[Crop]:
    """Crops are built from the *ingested* artefacts, so the benchmark measures
    the regions the pipeline actually feeds to a vision backend."""
    from specintel.ingest.pipeline import load_figures

    manifest_path = config.path("paths.synthetic") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    elements = manifest.get("elements", [])
    pdfs: dict[str, str] = {}
    for doc_id in docs:
        pdf_dir = config.path("paths.synthetic") / doc_id
        found = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        if found:
            pdfs[doc_id] = str(found[0])
    crops: list[Crop] = []
    for doc_id in docs:
        if "figure" in kinds:
            for figure in load_figures(config, doc_id):
                if not getattr(figure, "image_path", ""):
                    continue
                bbox = getattr(figure, "bbox", None)
                reference = _figure_reference(elements, doc_id, str(figure.number))
                crops.append(Crop(crop_id=f"{doc_id}:figure:{figure.number}", doc_id=doc_id,
                                  page=int(getattr(figure, "page", 0) or 0), kind="figure",
                                  image_path=str(figure.image_path), reference_text=reference,
                                  meta={"pdf": pdfs.get(doc_id, ""), "bbox": bbox,
                                        "raster": bool(getattr(figure, "has_raster", False))}))
        if "page" in kinds and pdfs.get(doc_id):
            pages_dir = config.path("paths.processed") / doc_id / "pages"
            images = sorted(pages_dir.glob("*.png")) if pages_dir.exists() else []
            by_page = {_page_of(p): p for p in images}
            for page_no, image in sorted(by_page.items()):
                reference = _page_reference(elements, doc_id, page_no)
                if not reference:
                    continue
                crops.append(Crop(crop_id=f"{doc_id}:page:{page_no}", doc_id=doc_id, page=page_no,
                                  kind="page", image_path=str(image), reference_text=reference,
                                  meta={"pdf": pdfs[doc_id], "bbox": None}))
    if limit:
        crops = crops[:limit]
    return crops


def _page_reference(elements: Sequence[dict[str, Any]], doc_id: str, page: int) -> str:
    """Everything the page is supposed to carry, in manifest order.

    Headings, notes, list items and the cells of any table on the page are part
    of the reference: a text layer that loses a table cell or a heading has lost
    real content, and measuring only paragraphs would hide that.
    """
    parts: list[str] = []
    for element in elements:
        if element.get("doc_id") != doc_id or int(element.get("page") or 0) != page:
            continue
        kind = element.get("type")
        if kind in ("paragraph", "note", "heading", "section_heading"):
            parts.append(str(element.get("text", "")))
        elif kind == "bullets":
            parts.extend(str(i) for i in (element.get("items") or []))
        elif kind == "table":
            parts.extend(str(c) for row in (element.get("headers") or []) for c in row)
            parts.extend(str(c) for row in (element.get("rows") or []) for c in row)
        elif kind == "figure":
            parts.extend(str(n) for n in (element.get("nodes") or []))
            parts.extend(str(e[2]) for e in (element.get("edges") or [])
                         if isinstance(e, (list, tuple)) and len(e) > 2)
    return "\n".join(p for p in parts if p.strip())


def _page_of(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.png$)", path.name)
    return int(match.group(1)) if match else 0


def _figure_reference(elements: Sequence[dict[str, Any]], doc_id: str, number: str) -> str:
    """The text the figure is supposed to carry, from the generation manifest."""
    for element in elements:
        if (element.get("doc_id") == doc_id and element.get("type") == "figure"
                and str(element.get("number")) == number):
            labels = element.get("labels") or []
            if labels:
                return "\n".join(str(x) for x in labels)
            nodes = element.get("nodes") or []
            edges = element.get("edges") or []
            return "\n".join([str(n) for n in nodes]
                             + [str(e[2]) for e in edges if isinstance(e, (list, tuple)) and len(e) > 2])
    return ""


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_benchmark(config: Any, *, docs: Sequence[str], backend_names: Sequence[str],
                  limit: int | None, dpi: int, endpoint: str) -> dict[str, Any]:
    log = get_logger("ws1.bench", config)
    crops = build_crops(config, docs, limit=limit)
    available = backends_available(config)
    rows: list[dict[str, Any]] = []
    log_event(log, "ocr.bench.start", crops=len(crops), backends=",".join(backend_names),
              hardware=hardware_profile())
    for crop in crops:
        for name in backend_names:
            if not available.get(name, False) and name != "noop":
                rows.append({"crop_id": crop.crop_id, "doc_id": crop.doc_id, "kind": crop.kind,
                             "page": crop.page, "backend": name, "available": False,
                             "skipped": available.get(name, False) or "dependency missing",
                             "cer": None, "wer": None, "number_accuracy": None,
                             "latency_s": None, "rss_mb": None, "chars": 0})
                continue
            prediction = BACKENDS[name](crop, config, dpi=dpi, endpoint=endpoint)
            reference = crop.reference_text
            if prediction.skipped:
                rows.append({"crop_id": crop.crop_id, "doc_id": crop.doc_id, "kind": crop.kind,
                             "page": crop.page, "backend": name, "available": False,
                             "skipped": prediction.skipped, "cer": None, "wer": None,
                             "number_accuracy": None, "latency_s": None, "rss_mb": None, "chars": 0})
                continue
            recall, spurious = word_coverage(reference, prediction.text) if reference else (None, None)
            row = {
                "crop_id": crop.crop_id, "doc_id": crop.doc_id, "kind": crop.kind, "page": crop.page,
                "backend": name, "available": not prediction.skipped,
                "skipped": prediction.skipped, "chars": len(prediction.text),
                "word_recall": None if recall is None else round(recall, 4),
                "spurious_share": None if spurious is None else round(spurious, 4),
                "cer": None if not reference else round(character_error_rate(reference, prediction.text), 4),
                "wer": None if not reference else round(word_error_rate(reference, prediction.text), 4),
                "number_accuracy": (None if not reference
                                    else _none_if_nan(number_accuracy(reference, prediction.text))),
                "latency_s": round(prediction.latency_s, 4), "rss_mb": round(prediction.rss_mb, 1),
            }
            rows.append(row)
            log_event(log, "ocr.bench.crop", crop=crop.crop_id, backend=name,
                      cer=row["cer"], latency_s=row["latency_s"], chars=row["chars"])
    return {"crops": len(crops), "docs": list(docs), "rows": rows,
            "availability": available, "hardware": hardware_profile(),
            "dpi": dpi, "endpoint": endpoint}


def _none_if_nan(value: float) -> float | None:
    return None if value != value else value


def summarise(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Average each backend overall, per crop kind and per document."""
    rows = payload["rows"]
    summary: list[dict[str, Any]] = []
    for backend in sorted({r["backend"] for r in rows}):
        group = [r for r in rows if r["backend"] == backend]
        usable = [r for r in group if r["cer"] is not None]
        summary.append({
            "backend": backend,
            "available": bool(group and group[0]["available"]),
            "skipped": group[0]["skipped"] if group else "",
            "crops": len(group),
            "scored": len(usable),
            "cer": _mean([r["cer"] for r in usable]),
            "wer": _mean([r["wer"] for r in usable]),
            "word_recall": _mean([r["word_recall"] for r in usable]),
            "spurious_share": _mean([r["spurious_share"] for r in usable]),
            "number_accuracy": _mean([r["number_accuracy"] for r in usable
                                      if r["number_accuracy"] is not None]),
            "latency_s": _mean([r["latency_s"] for r in usable]),
            "rss_mb": _mean([r["rss_mb"] for r in usable]),
            "chars": _mean([r["chars"] for r in usable]) if usable else 0,
            "by_kind": {kind: _mean([r["cer"] for r in usable if r["kind"] == kind])
                        for kind in sorted({r["kind"] for r in usable})},
            "by_document": {doc: _mean([r["cer"] for r in usable if r["doc_id"] == doc])
                            for doc in sorted({r["doc_id"] for r in usable})},
        })
    return summary


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [v for v in values if v is not None and v == v]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def write_report(payload: dict[str, Any], summary: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, **payload}, indent=2),
                                          encoding="utf-8")
    lines = ["# OCR / document-AI benchmark (WS1, Task 2 Step 4)", "",
             f"Crops benchmarked: **{payload['crops']}** in {len(payload['docs'])} document(s) "
             f"({', '.join(payload['docs'])}), rasterised at {payload['dpi']} dpi.", "",
             "Lower is better for CER/WER; higher is better for number accuracy.", "",
             "| backend | available | crops | CER | WER | word recall | spurious | number acc. | "
             "s/crop | RSS MB | note |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        lines.append("| {backend} | {available} | {scored}/{crops} | {cer} | {wer} | {recall} | "
                     "{spurious} | {num} | {latency} | {rss} | {note} |".format(
                         backend=row["backend"], available="yes" if row["available"] else "**no**",
                         scored=row["scored"], crops=row["crops"], cer=_fmt(row["cer"]),
                         wer=_fmt(row["wer"]), recall=_fmt(row["word_recall"]),
                         spurious=_fmt(row["spurious_share"]), num=_fmt(row["number_accuracy"]),
                         latency=_fmt(row["latency_s"]), rss=_fmt(row["rss_mb"]),
                         note=row["skipped"] or ""))
    lines += ["", "## By content type (CER)", "",
              "| backend | " + " | ".join(sorted({r["kind"] for r in payload["rows"]})) + " |",
              "| --- | " + " | ".join("---" for _ in sorted({r["kind"] for r in payload["rows"]})) + " |"]
    for row in summary:
        kinds = sorted({r["kind"] for r in payload["rows"]})
        lines.append(f"| {row['backend']} | " + " | ".join(_fmt(row["by_kind"].get(k)) for k in kinds) + " |")
    lines += ["", "## By document (CER)", "",
              "| backend | " + " | ".join(payload["docs"]) + " |",
              "| --- | " + " | ".join("---" for _ in payload["docs"]) + " |"]
    for row in summary:
        lines.append(f"| {row['backend']} | "
                     + " | ".join(_fmt(row["by_document"].get(d)) for d in payload["docs"]) + " |")
    lines += ["", "## Read this table honestly", "",
              "* CER and WER compare the recovered text with the reference as one string, so they "
              "punish reading order as much as missing text. **word recall** and **spurious** split "
              "those two failures apart: recall is how much of the page came back, spurious is how "
              "much of the output is not on the page at all.",
              "* `pymupdf-text` reads the vector text layer. It is exact and free where a text "
              "layer exists (USB-PD, Type-C, USB4 as published) and returns nothing on a scan or "
              "a rasterised figure — which is why it is the baseline, not the answer. Its "
              "**spurious** share is the running headers, footers and page numbers, which are on "
              "the page but not part of the content being reconstructed; that is also why the "
              "pipeline strips them before indexing.",
              "* `noop` is a control. Every figure it scores is a text-recovery failure the "
              "pipeline would have to cover another way.",
              "* `unlimited-ocr` needs an NVIDIA GPU and a served model. When the row says **no**, "
              "the machine could not run it and no number is reported: see "
              "`docs/OCR_BENCHMARK.md` for the protocol to run it and what it is expected to fix "
              "(dense figure labels, where raster+OCR is weakest).",
              ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "report.md"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="pd,typec,usb4,tech2")
    parser.add_argument("--backends", default="pymupdf-text,pymupdf-raster+ocr,rapidocr,"
                                              "unlimited-ocr,noop")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of crops")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--unlimited-ocr-endpoint", default="")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "ocr" / "results"))
    args = parser.parse_args(argv)

    config = load_config()
    setup_from_config(config)
    docs = [d.strip() for d in args.docs.split(",") if d.strip()]
    payload = run_benchmark(config, docs=docs,
                            backend_names=[b.strip() for b in args.backends.split(",") if b.strip()],
                            limit=args.limit, dpi=args.dpi, endpoint=args.unlimited_ocr_endpoint)
    summary = summarise(payload)
    report = write_report(payload, summary, Path(args.out))
    print()
    for row in summary:
        print(f"  {row['backend']:<20} available={str(row['available']):<5} "
              f"CER={_fmt(row['cer']):<7} WER={_fmt(row['wer']):<7} "
              f"numbers={_fmt(row['number_accuracy']):<7} s/crop={_fmt(row['latency_s'])}")
    print(f"\n  report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
