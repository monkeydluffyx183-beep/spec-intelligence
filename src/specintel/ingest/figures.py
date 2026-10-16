"""WS1 — figures: detection, cropping, captioning and optional VLM structuring.

Specification figures matter twice: the assignment's figure questions can only
be answered from the diagram, and the vision path (WS2) retrieves the page
image.  This module:

1. finds figure regions by clustering vector drawings and raster images, so
   both the vector state machines of PD / Type-C and the scanned-looking
   figures of Tech 2 are found by the same code;
2. crops them to PNG at a configurable DPI, keeping the caption and any "see
   Figure ..." references that point at them;
3. asks a VLM (or the heuristic fallback) for (a) a retrieval description and
   (b) structured JSON when the figure type allows it:

       state machine → {"states": [...], "transitions": [{"from","to","trigger","actions"}]}
       timing        → {"signals": [...], "events": [...], "timers": [...]}

   The heuristic backend reads the labels that are *inside* the crop, which is
   exactly the information a text-only pipeline has, and reports what it could
   not determine, so the accuracy of the structured output can be measured
   honestly against the labels recorded in the corpus manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.models import Figure, new_id
from specintel.ingest.parser import CAPTION_RE, open_pdf, page_text_lines

STATE_NAME_RE = re.compile(r"\b(?:PE|PRL|PE_SNK|PE_SRC)_[A-Za-z_]{2,}\b|\b[A-Z][A-Za-z]+\.(?:SRC|SNK|DBGACC|AUDIOACC)\b")
TIMER_RE = re.compile(r"\bt[A-Z][A-Za-z0-9]{2,}\b")
SIGNAL_RE = re.compile(r"\b(?:VBUS|VCONN|CC1|CC2|Rp|Rd|Ra|D\+|D-)\b")
EVENT_WORDS = ("request", "accept", "received", "sent", "elapsed", "detected", "valid", "fail")
STATEMENT_KEYWORDS = ("state machine", "transition", "state diagram")
TIMING_KEYWORDS = ("timing", "waveform", "sequence", "control loop")


@dataclass
class Region:
    """A candidate figure region on one page."""

    page: int
    bbox: list[float]
    kind: str = "vector"        # vector | raster | mixed
    area: float = 0.0


@dataclass
class DetectOptions:
    doc_id: str
    revision: str
    min_area_ratio: float = 0.02
    min_width: float = 120.0
    min_height: float = 45.0
    cluster_gap: float = 14.0
    crop_dpi: int = 200
    max_figures_per_page: int = 4
    image_dir: str | Path = "data/processed/figures"


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _rect_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _expand(bbox: Sequence[float], gap: float) -> list[float]:
    return [bbox[0] - gap, bbox[1] - gap, bbox[2] + gap, bbox[3] + gap]


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def cluster_regions(rects: Sequence[tuple[list[float], str]], gap: float = 14.0) -> list[Region]:
    """Union overlapping/nearby rectangles into figure regions."""
    clusters: list[dict[str, Any]] = []
    for bbox, kind in rects:
        placed = False
        for cluster in clusters:
            if _overlaps(_expand(bbox, gap), cluster["bbox"]):
                cluster["bbox"] = _union(cluster["bbox"], bbox)
                cluster["members"].append(bbox)
                cluster["kinds"].add(kind)
                placed = True
                break
        if not placed:
            clusters.append({"bbox": list(bbox), "members": [bbox], "kinds": {kind}})
    out: list[Region] = []
    for cluster in clusters:
        bbox = cluster["bbox"]
        kinds = cluster["kinds"]
        out.append(Region(page=0, bbox=bbox, kind="mixed" if len(kinds) > 1 else next(iter(kinds)),
                          area=_rect_area(bbox)))
    return out


def detect_figures(path: str | Path, options: DetectOptions, *, table_boxes: Sequence[tuple[int, list[float]]] = ()
                   ) -> list[Region]:
    """Detect figure regions in a PDF, skipping table grids and page furniture."""
    doc = open_pdf(path)
    regions: list[Region] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        page_area = float(page.rect.width * page.rect.height)
        rects: list[tuple[list[float], str]] = []
        try:
            for drawing in page.get_drawings():
                bbox = [float(v) for v in drawing.get("rect", (0, 0, 0, 0))]
                if bbox[2] - bbox[0] < 3 and bbox[3] - bbox[1] < 3:
                    continue
                rects.append((bbox, "vector"))
        except Exception:
            pass
        try:
            for info in page.get_image_info():
                bbox = [float(v) for v in info.get("bbox", (0, 0, 0, 0))]
                rects.append((bbox, "raster"))
        except Exception:
            pass
        for region in cluster_regions(rects, options.cluster_gap):
            region.page = pno + 1
            if region.area < options.min_area_ratio * page_area:
                continue
            w = region.bbox[2] - region.bbox[0]
            h = region.bbox[3] - region.bbox[1]
            if w < options.min_width or h < options.min_height:
                continue
            if any(tpage == region.page and _overlap_ratio(region.bbox, tbox) > 0.35
                   for tpage, tbox in table_boxes):
                continue
            regions.append(region)
        # keep the largest few per page
        page_regions = [r for r in regions if r.page == pno + 1]
        if len(page_regions) > options.max_figures_per_page:
            keep = sorted(page_regions, key=lambda r: r.area, reverse=True)[:options.max_figures_per_page]
            regions = [r for r in regions if r.page != pno + 1] + keep
    doc.close()
    return sorted(regions, key=lambda r: (r.page, r.bbox[1], r.bbox[0]))


def _overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _rect_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return inter / max(1.0, _rect_area(a))


# --------------------------------------------------------------------------- #
# captions and references
# --------------------------------------------------------------------------- #
def find_caption(page_lines: Sequence[Any], bbox: Sequence[float]) -> tuple[str, str]:
    """Find the ``Figure N. caption`` nearest below (or above) a region."""
    best: tuple[float, str] | None = None
    for line in page_lines:
        match = CAPTION_RE.match(line.text)
        if not match or match.group("kind").lower() != "figure":
            continue
        if line.y0 >= bbox[3] - 4:
            distance = line.y0 - bbox[3]
        elif line.y1 <= bbox[1] + 4:
            distance = bbox[1] - line.y1
        else:
            continue
        if 0 <= distance <= 90 and (best is None or distance < best[0]):
            best = (distance, line.text)
    if best is None:
        return "", ""
    match = CAPTION_RE.match(best[1])
    if not match:
        return "", best[1]
    return match.group("number"), match.group("text").strip()


def find_references(elements: Iterable[Any], number: str, doc_id: str) -> list[str]:
    """Collect "see Figure N" style mentions of a figure from the text path."""
    if not number:
        return []
    pattern = re.compile(rf"\b(?:Figure|Fig\.?)\s+{re.escape(number)}\b", re.IGNORECASE)
    refs: list[str] = []
    for el in elements:
        if el.doc_id != doc_id or not el.text:
            continue
        if pattern.search(el.text):
            refs.append(f"{el.doc_id}:{el.element_type}:p{el.page}:{el.text[:160]}")
    return refs[:20]


# --------------------------------------------------------------------------- #
# cropping
# --------------------------------------------------------------------------- #
def crop_region(path: str | Path, region: Region, out_dir: str | Path, doc_id: str, dpi: int = 200) -> str | None:
    """Render a figure region to PNG and return the path."""
    pymupdf = open_pdf  # alias keeps the import local to this module
    doc = pymupdf(path)
    try:
        page = doc[region.page - 1]
        pad = 4.0
        clip = pymupdf_clip(page, region.bbox, pad)
        zoom = dpi / 72.0
        matrix = pymupdf_matrix(zoom)
        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_id}_p{region.page:04d}_{int(region.bbox[1])}_{int(region.bbox[0])}.png"
        pix.save(str(out_path))
        return str(out_path)
    except Exception:
        return None
    finally:
        doc.close()


def pymupdf_clip(page: Any, bbox: Sequence[float], pad: float = 4.0):
    import pymupdf

    rect = pymupdf.Rect(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    return rect & page.rect


def pymupdf_matrix(zoom: float):
    import pymupdf

    return pymupdf.Matrix(zoom, zoom)


# --------------------------------------------------------------------------- #
# structuring (VLM or heuristic)
# --------------------------------------------------------------------------- #
@dataclass
class StructureResult:
    kind: str
    description: str
    structure: dict[str, Any]
    confidence: float
    notes: list[str] = field(default_factory=list)
    backend: str = "heuristic"


def classify_figure(text_inside: str, caption: str, has_raster: bool) -> str:
    """Guess the figure type from its caption, its labels and its rendering."""
    blob = f"{caption} {text_inside}".lower()
    if any(k in blob for k in STATEMENT_KEYWORDS) or STATE_NAME_RE.search(text_inside):
        return "state_machine"
    if any(k in blob for k in TIMING_KEYWORDS) or re.search(r"\bt[A-Z][A-Za-z]+\b", text_inside):
        return "timing"
    if has_raster and not text_inside.strip():
        return "photo"
    if any(k in blob for k in ("router", "adapter", "block", "phase", "architecture", "flow", "diagram")):
        return "block"
    if has_raster:
        return "photo"
    return "chart"


def structure_heuristic(text_inside: str, caption: str, has_raster: bool = False,
                        kind: str | None = None) -> StructureResult:
    """Deterministic structuring from the labels inside the crop.

    This is the offline fallback: it reports what can be read from the labels
    and says explicitly what it could not determine, which is what the accuracy
    check of Task 2, Step 4 measures.
    """
    kind = kind or classify_figure(text_inside, caption, has_raster)
    states = sorted(set(STATE_NAME_RE.findall(text_inside)))
    timers = sorted(set(TIMER_RE.findall(text_inside) + TIMER_RE.findall(caption)))
    signals = sorted(set(SIGNAL_RE.findall(text_inside) + SIGNAL_RE.findall(caption)))
    notes: list[str] = []
    structure: dict[str, Any] = {}

    if kind == "state_machine":
        transitions: list[dict[str, Any]] = []
        lines = [l.strip() for l in text_inside.splitlines() if l.strip()]
        for i in range(len(lines) - 1):
            src, dst = lines[i], lines[i + 1]
            if STATE_NAME_RE.fullmatch(src.strip()) and STATE_NAME_RE.fullmatch(dst.strip()):
                transitions.append({"from": src.strip(), "to": dst.strip(), "trigger": "", "actions": []})
        structure = {"states": states, "transitions": transitions}
        if not transitions:
            notes.append("no transition found from labels alone: the arrows of the diagram carry no text")
        if len(states) < 2:
            notes.append("fewer than two state labels recovered from the crop")
    elif kind == "timing":
        events = [w for w in EVENT_WORDS if w in text_inside.lower()]
        # the labels drawn inside a timing diagram ("EPR Mode", "PS_RDY") name
        # the messages the waveform carries; keep them, the accuracy check of
        # Task 2 Step 4 matches them against the labels of the drawn figure
        drawn = [l.strip() for l in text_inside.splitlines() if l.strip()]
        structure = {"signals": signals, "events": events, "timers": timers,
                     "labels": [d for d in drawn if len(d) > 2][:20]}
        notes.append("timing relationships are estimated from label order only")
    else:
        structure = {"labels": [l.strip() for l in text_inside.splitlines() if l.strip()][:20]}

    description = _describe(kind, caption, states, timers, signals, structure)
    confidence = 0.55 if kind == "state_machine" and structure.get("transitions") else 0.35
    return StructureResult(kind=kind, description=description, structure=structure, confidence=confidence,
                           notes=notes, backend="heuristic")


def _describe(kind: str, caption: str, states: Sequence[str], timers: Sequence[str],
              signals: Sequence[str], structure: dict[str, Any]) -> str:
    parts = [f"{kind} figure", caption or "untitled"]
    labels = [str(l) for l in structure.get("labels", []) if str(l).strip()]
    if labels and kind not in ("state_machine", "timing"):
        parts.append("labels: " + ", ".join(labels[:16]))
    if states:
        parts.append("states/entities: " + ", ".join(states[:12]))
    if timers:
        parts.append("timers: " + ", ".join(timers[:8]))
    if signals:
        parts.append("signals: " + ", ".join(signals[:8]))
    if kind == "state_machine" and structure.get("transitions"):
        parts.append("transitions: " + "; ".join(f"{t['from']} -> {t['to']}" for t in structure["transitions"][:8]))
    return ". ".join(parts) + "."


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def detect_and_crop(path: str | Path, options: DetectOptions, *,
                    table_boxes: Sequence[tuple[int, list[float]]] = (),
                    elements: Sequence[Any] = (),
                    vlm: Any | None = None) -> list[Figure]:
    """Detect, crop, caption and structure every figure of one PDF."""
    doc = open_pdf(path)
    page_lines = {pno + 1: page_text_lines(doc[pno], pno + 1) for pno in range(doc.page_count)}
    doc.close()

    figures: list[Figure] = []
    for region in detect_figures(path, options, table_boxes=table_boxes):
        lines = page_lines.get(region.page, [])
        inside = "\n".join(l.text for l in lines
                           if l.x0 >= region.bbox[0] - 6 and l.x1 <= region.bbox[2] + 6
                           and l.y0 >= region.bbox[1] - 6 and l.y1 <= region.bbox[3] + 6)
        number, caption = find_caption(lines, region.bbox)
        image_path = crop_region(path, region, options.image_dir, options.doc_id, options.crop_dpi)
        has_raster = region.kind in ("raster", "mixed")

        if vlm is not None:
            try:
                result = vlm.structure_figure(image_path, caption=caption, labels=inside, has_raster=has_raster)
            except Exception as exc:                      # a failed VLM call must not stop ingestion
                from specintel.ingest.vlm import HeuristicVLM

                result = HeuristicVLM().structure_figure(image_path, caption=caption, labels=inside,
                                                        has_raster=has_raster)
                result.notes.append(f"VLM backend failed, used heuristic: {exc}")
        else:
            from specintel.ingest.vlm import HeuristicVLM

            result = HeuristicVLM().structure_figure(image_path, caption=caption, labels=inside,
                                                     has_raster=has_raster)

        figures.append(Figure(
            figure_id=f"{options.doc_id}:fig:{number or new_id('fig')}",
            doc_id=options.doc_id, revision=options.revision, number=number or "",
            caption=caption, page=region.page, image_path=image_path, bbox=region.bbox,
            kind=result.kind, description=result.description, structure=result.structure,
            references=find_references(elements, number, options.doc_id),
            extractor="pymupdf", hallucination_notes=result.notes,
        ))
    return figures


def figures_to_json(figures: Sequence[Figure]) -> dict[str, Any]:
    return {"figures": [f.to_dict() for f in figures]}


def structured_figure_accuracy(gold_structure: dict[str, Any], pred_structure: dict[str, Any] | None) -> dict[str, float]:
    """Compare structured figure JSON with the labels recorded for the page.

    Measures state recall (are the diagram's states named?), transition recall
    (are the transitions recovered?) and the trigger/action accuracy that the
    assignment asks to report for ten figures.
    """
    pred = pred_structure or {}
    gold_states = {s.lower() for s in gold_structure.get("states", [])}
    pred_states = {s.lower() for s in pred.get("states", [])}
    gold_tr = {(t.get("from", "").lower(), t.get("to", "").lower()) for t in gold_structure.get("transitions", [])}
    pred_tr = {(t.get("from", "").lower(), t.get("to", "").lower()) for t in pred.get("transitions", [])}
    gold_signals = {s.lower() for s in gold_structure.get("signals", [])}
    pred_signals = {s.lower() for s in pred.get("signals", [])}
    return {
        "state_recall": len(gold_states & pred_states) / len(gold_states) if gold_states else 0.0,
        "transition_recall": len(gold_tr & pred_tr) / len(gold_tr) if gold_tr else 0.0,
        "signal_recall": len(gold_signals & pred_signals) / len(gold_signals) if gold_signals else 0.0,
        "hallucination_rate": (len(pred_states - gold_states) / len(pred_states)) if pred_states else 0.0,
    }
