"""WS1 — ingestion pipeline: PDFs in, typed elements + tables + figures out.

Returns from any phase failed on a document are logged and skipped rather than
crashing the run, so one broken page never blocks the other six documents.

Usage
-----
    from specintel.ingest.pipeline import ingest_document, ingest_corpus
    result = ingest_document(config, "pd")                 # real PDFs if present,
    result = ingest_corpus(config)                         # synthetic otherwise
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.logging import get_logger, log_event, log_exception, timed
from specintel.common.models import Element, Figure, PageImage, Table, new_id
from specintel.common.resources import Stats, hardware_profile, rss_mb, stopwatch
from specintel.ingest.figures import DetectOptions, detect_and_crop, figures_to_json
from specintel.ingest.parser import (ParseOptions, available_parsers, element_type_counts, get_parser,
                                     open_pdf, validate_elements)
from specintel.ingest.tables import (attach_sections, detect_tables, stitch_tables, table_markdown_bundle,
                                     tables_to_json)


@dataclass
class IngestResult:
    """Everything WS1 produced for one document."""

    doc_id: str
    revision: str
    pages: int = 0
    source: str = "synthetic"                 # synthetic | raw
    files: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    page_images: list[PageImage] = field(default_factory=list)
    out_dir: Path | None = None
    problems: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- derived ----------------------------------------------------------- #
    @property
    def counts(self) -> dict[str, int]:
        return element_type_counts(self.elements)

    @property
    def stitched_tables(self) -> list[Table]:
        return [t for t in self.tables if t.stitched]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "revision": self.revision, "pages": self.pages, "source": self.source,
            "files": self.files, "counts": self.counts, "tables": len(self.tables),
            "stitched_tables": len(self.stitched_tables), "figures": len(self.figures),
            "page_images": len(self.page_images), "problems": self.problems, "stats": self.stats,
        }


# --------------------------------------------------------------------------- #
# source resolution
# --------------------------------------------------------------------------- #
def resolve_documents(config: Any) -> list[dict[str, Any]]:
    """Return the documents to ingest: real PDFs when present, else synthetic.

    ``data/raw/<doc_id>/*.pdf`` wins.  When that folder is missing the pipeline
    falls back to the generated stand-in corpus (``paths.synthetic``) so a clean
    clone can run ingest → build-graph → eval without the licensed files.
    """
    raw_root = config.path("paths.raw")
    synth_root = config.path("paths.synthetic")
    out: list[dict[str, Any]] = []
    for doc in config.documents():
        doc_id = doc["doc_id"]
        raw_dir = raw_root / doc_id
        files = sorted(str(p) for p in raw_dir.glob("*.pdf")) if raw_dir.exists() else []
        if files:
            out.append({**doc, "files": files, "source": "raw"})
            continue
        synth_dir = synth_root / doc_id
        synth_files = sorted(str(p) for p in synth_dir.glob("*.pdf")) if synth_dir.exists() else []
        out.append({**doc, "files": synth_files, "source": "synthetic"})
    return out


def doc_revision(config: Any, doc_id: str) -> str:
    return str(config.doc_meta(doc_id).get("revision", "unknown"))


# --------------------------------------------------------------------------- #
# page images (vision path input)
# --------------------------------------------------------------------------- #
def render_page_images(path: str | Path, *, doc_id: str, revision: str, out_dir: str | Path,
                       dpi: int = 160, pages: Sequence[int] | None = None) -> list[PageImage]:
    """Render pages to PNG for the vision path (WS2) and the OCR benchmark."""
    import pymupdf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(path))
    images: list[PageImage] = []
    try:
        selected = list(pages) if pages is not None else list(range(1, doc.page_count + 1))
        core_name = Path(path).stem.replace(" ", "_")
        for pno in selected:
            if not (1 <= pno <= doc.page_count):
                continue
            page = doc[pno - 1]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            out_path = out_dir / f"{doc_id}_{core_name}_p{pno:04d}.png"
            pix.save(str(out_path))
            images.append(PageImage(doc_id=doc_id, revision=revision, page=pno, image_path=str(out_path),
                                    dpi=dpi, width=pix.width, height=pix.height))
    finally:
        doc.close()
    return images


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def ingest_document(config: Any, doc_id: str, *, parser_name: str = "auto", ocr: Any | None = None,
                    vlm: Any | None = None, files: Sequence[str] | None = None,
                    source: str | None = None, render_pages: bool = True,
                    keep_images: bool = True) -> IngestResult:
    """Ingest one document: parse, tables, figures, page images, write artefacts."""
    log = get_logger("ws1.ingest", config)
    meta = config.doc_meta(doc_id)
    revision = str(meta.get("revision", "unknown"))
    if files is None or source is None:
        resolved = {d["doc_id"]: d for d in resolve_documents(config)}.get(doc_id)
        if resolved is None:
            raise KeyError(f"unknown doc_id '{doc_id}'")
        files = list(resolved["files"]) if files is None else files
        source = resolved["source"] if source is None else source
    files = list(files)

    processed = config.path("paths.processed") / doc_id
    processed.mkdir(parents=True, exist_ok=True)
    result = IngestResult(doc_id=doc_id, revision=revision, source=source or "synthetic",
                          files=[str(f) for f in files], out_dir=processed)
    options = ParseOptions(doc_id=doc_id, revision=revision,
                           extractor=str(config.get("ingest.parser", "pymupdf")))
    parser = get_parser(parser_name)
    detect_options = DetectOptions(
        doc_id=doc_id, revision=revision,
        min_area_ratio=float(config.get("ingest.figures.min_area_ratio", 0.02)),
        crop_dpi=int(config.get("ingest.figures.crop_dpi", 200)),
        image_dir=processed / "figures",
    )

    log_event(log, "ingest.start", doc=doc_id, revision=revision, files=len(files), source=source,
              parser=getattr(parser, "name", parser_name))

    part_boxes: list[tuple[int, list[float]]] = []
    parse_stats = Stats(name="parse")
    table_stats = Stats(name="tables")
    figure_stats = Stats(name="figures")
    page_stats = Stats(name="page_render")

    for path in files:
        if not Path(path).exists():
            result.problems.append(f"missing file {path}")
            log_event(log, "ingest.missing", level=30, doc=doc_id, file=path)
            continue
        try:
            with timed(log, "parse.file", doc=doc_id, file=Path(path).name) as t:
                with stopwatch(parse_stats) as elapsed:
                    elements = parser.parse(path, options)
                t["elements"] = len(elements)
            result.elements.extend(elements)
            result.problems.extend(validate_elements(elements))
            counts = element_type_counts(elements)
            log_event(log, "parse.counts", doc=doc_id, file=Path(path).name,
                      counts=json.dumps(counts, separators=(",", ":")), chars=sum(len(e.text) for e in elements))
        except Exception as exc:                     # never let one file block the rest
            log_exception(log, "parse.failed", exc, doc=doc_id, file=str(path))
            result.problems.append(f"parse failed for {path}: {exc}")
            continue

        try:
            with timed(log, "tables.file", doc=doc_id, file=Path(path).name) as t:
                with stopwatch(table_stats) as elapsed:
                    parts = detect_tables(path, doc_id=doc_id, revision=revision)
                    # every part of a multi-page table is excluded from figure
                    # detection, not only the page where the table started
                    part_boxes.extend((part.page, part.bbox) for part in parts)
                    tables = stitch_tables(parts, doc_id=doc_id, revision=revision,
                                           repeat_header_rows=int(config.get("ingest.tables.repeat_header_rows", 1)))
                    attach_sections(tables, elements)
                t["parts"] = len(parts)
                t["tables"] = len(tables)
                t["stitched"] = sum(1 for tb in tables if tb.stitched)
            result.tables.extend(tables)
            for table in tables:
                if table.stitched:
                    log_event(log, "tables.stitched", doc=doc_id, table=table.number, pages=table.pages,
                              rows=len(table.rows), cols=table.n_cols)
        except Exception as exc:
            log_exception(log, "tables.failed", exc, doc=doc_id, file=str(path))
            result.problems.append(f"tables failed for {path}: {exc}")

        file_table_boxes = [box for box in part_boxes
                            if box[0] <= (max((e.page for e in elements), default=0))]
        try:
            with timed(log, "figures.file", doc=doc_id, file=Path(path).name) as t:
                with stopwatch(figure_stats) as elapsed:
                    figures = detect_and_crop(path, detect_options, table_boxes=file_table_boxes,
                                              elements=elements, vlm=vlm)
                t["figures"] = len(figures)
                t["structured"] = sum(1 for f in figures if f.structure)
                t["raster"] = sum(1 for f in figures if f.image_path and f.kind == "photo")
            result.figures.extend(figures)
            for figure in figures:
                log_event(log, "figure.structured", doc=doc_id, figure=figure.number or figure.figure_id,
                          page=figure.page, kind=figure.kind, states=len((figure.structure or {}).get("states", [])),
                          transitions=len((figure.structure or {}).get("transitions", [])),
                          notes=len(figure.hallucination_notes))
        except Exception as exc:
            log_exception(log, "figures.failed", exc, doc=doc_id, file=str(path))
            result.problems.append(f"figures failed for {path}: {exc}")

        if render_pages:
            try:
                dpi = int(config.get("index.page_index.dpi", 160))
                with timed(log, "pages.render", doc=doc_id, file=Path(path).name, dpi=dpi) as t:
                    with stopwatch(page_stats) as elapsed:
                        images = render_page_images(path, doc_id=doc_id, revision=revision,
                                                    out_dir=processed / "pages", dpi=dpi)
                    t["pages"] = len(images)
                if not keep_images:
                    for img in images:
                        Path(img.image_path).unlink(missing_ok=True)
                result.page_images.extend(images)
            except Exception as exc:
                log_exception(log, "pages.failed", exc, doc=doc_id, file=str(path))
                result.problems.append(f"page render failed for {path}: {exc}")

    # element ids quoted on tables and figures so the agent can fetch them back
    _link_assets(result, part_boxes)

    result.pages = max((e.page for e in result.elements), default=0) or max(
        (pi.page for pi in result.page_images), default=0)
    result.stats = {
        "parse": parse_stats.as_dict(), "tables": table_stats.as_dict(),
        "figures": figure_stats.as_dict(), "page_render": page_stats.as_dict(),
    }

    # -- artefacts --------------------------------------------------------- #
    _write_artefacts(result, processed)
    if result.problems:
        log_event(log, "ingest.problems", level=30, doc=doc_id, n=len(result.problems),
                  sample=result.problems[:3])
    log_event(log, "ingest.done", doc=doc_id, elements=len(result.elements), tables=len(result.tables),
              stitched=len(result.stitched_tables), figures=len(result.figures),
              page_images=len(result.page_images), pages=result.pages,
              counts=json.dumps(result.counts, separators=(",", ":")), rss_mb=rss_mb())
    return result


def _link_assets(result: IngestResult, table_parts: Sequence[tuple[int, list[float]]] = ()) -> None:
    """Attach table_id / figure_id to the caption elements that belong to them.

    The link is what lets the agent answer "in which table is tSenderResponse
    defined" with a citation that names the table, and lets WS2 build atomic
    table/figure chunks from the same identifiers.

    ``table_parts`` carries the detected table regions (page, box) so that raw
    grid text the text path picked up inside a table is retyped to ``table``:
    a PDF table's body is *table content*, not prose, and leaving it as prose
    lets the agent quote column-merged fragments as if they were sentences.
    """
    by_page: dict[tuple[str, int], list[Element]] = {}
    for el in result.elements:
        by_page.setdefault((el.doc_id, el.page), []).append(el)

    def _matches(el: Element, kind: str, number: str) -> bool:
        if el.meta.get("caption_kind") != kind:
            return False
        if not number:
            return True
        return str(el.meta.get("caption_number", "")) == number

    for table in result.tables:
        for page in table.pages:
            for el in by_page.get((table.doc_id, page), []):
                if _matches(el, "table", table.number):
                    el.table_id = table.table_id
                    if el.element_id not in table.element_ids:
                        table.element_ids.append(el.element_id)
    # raw grid text that the text path picked up inside a table region is
    # retyped to "table" and linked to the stitched table
    retyped = 0
    for el in result.elements:
        if el.element_type in ("header", "footer", "toc") or el.page < 1 or not el.bbox:
            continue
        for page, bbox in table_parts:
            if page != el.page or not _inside(el.bbox, bbox):
                continue
            if el.element_type != "table":
                el.meta["retyped_from"] = el.element_type
                el.element_type = "table"
                retyped += 1
            if el.table_id is None:
                for table in result.tables:
                    if el.page in table.pages and getattr(table, "bbox", None) and _inside(el.bbox, table.bbox):
                        el.table_id = table.table_id
                        if el.element_id not in table.element_ids:
                            table.element_ids.append(el.element_id)
                        break
            break
    if retyped:
        result.stats["table_text_elements_retyped"] = {"count": retyped}

    for figure in result.figures:
        for page in {figure.page}:
            for el in by_page.get((figure.doc_id, page), []):
                if _matches(el, "figure", figure.number):
                    el.figure_id = figure.figure_id


def _inside(inner: Sequence[float], outer: Sequence[float], pad: float = 3.0) -> bool:
    """True when ``inner`` sits inside ``outer`` (with a small tolerance)."""
    if not inner or not outer:
        return False
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return (outer[0] - pad) <= cx <= (outer[2] + pad) and (outer[1] - pad) <= cy <= (outer[3] + pad)


def _write_artefacts(result: IngestResult, out_dir: Path) -> None:
    """Write the element JSONL, table JSON/Markdown, figure JSON and stats."""
    element_path = out_dir / "elements.jsonl"
    Element.write_jsonl(str(element_path), result.elements)
    (out_dir / "tables.json").write_text(json.dumps(tables_to_json(result.tables), indent=1, default=str),
                                         encoding="utf-8")
    (out_dir / "tables.md").write_text(table_markdown_bundle(result.tables), encoding="utf-8")
    (out_dir / "figures.json").write_text(json.dumps(figures_to_json(result.figures), indent=1, default=str),
                                          encoding="utf-8")
    if result.page_images:
        (out_dir / "pages.jsonl").write_text(
            "\n".join(pi.to_json() for pi in result.page_images) + "\n", encoding="utf-8")
    (out_dir / "ingest_stats.json").write_text(json.dumps(result.to_dict(), indent=1, default=str),
                                               encoding="utf-8")


def ingest_corpus(config: Any, *, doc_ids: Sequence[str] | None = None, parser_name: str = "auto",
                  vlm: Any | None = None, ocr: Any | None = None, render_pages: bool = True,
                  keep_images: bool = True) -> dict[str, IngestResult]:
    """Ingest every configured document (real PDFs when present, else synthetic)."""
    log = get_logger("ws1.ingest", config)
    resolved = {d["doc_id"]: d for d in resolve_documents(config)}
    ids = list(doc_ids) if doc_ids else list(resolved)
    results: dict[str, IngestResult] = {}
    log_event(log, "ingest.corpus.start", documents=len(ids), hardware=hardware_profile())
    for doc_id in ids:
        info = resolved.get(doc_id)
        if not info or not info["files"]:
            log_event(log, "ingest.skipped", level=30, doc=doc_id,
                      reason="no PDFs found in data/raw/<doc_id> and no synthetic stand-in")
            continue
        results[doc_id] = ingest_document(config, doc_id, parser_name=parser_name, vlm=vlm, ocr=ocr,
                                          files=info["files"], source=info["source"],
                                          render_pages=render_pages, keep_images=keep_images)
    totals = {
        "documents": len(results),
        "elements": sum(len(r.elements) for r in results.values()),
        "tables": sum(len(r.tables) for r in results.values()),
        "figures": sum(len(r.figures) for r in results.values()),
        "pages": sum(r.pages for r in results.values()),
    }
    log_event(log, "ingest.corpus.done", **totals)
    return results


def load_elements(config: Any, doc_ids: Sequence[str] | None = None) -> list[Element]:
    """Load the ingested element streams back from ``data/processed``."""
    processed = config.path("paths.processed")
    elements: list[Element] = []
    for doc in config.documents():
        if doc_ids and doc["doc_id"] not in doc_ids:
            continue
        path = processed / doc["doc_id"] / "elements.jsonl"
        if path.exists():
            elements.extend(Element.read_jsonl(str(path)))
    return elements


def load_tables(config: Any, doc_id: str) -> list[Table]:
    """Tables of one document, as saved by the ingestion run.

    A missing file is reported rather than swallowed: an unknown document id is
    a caller error, and returning an empty list would quietly turn every table
    metric into zero.
    """
    path = config.path("paths.processed") / doc_id / "tables.json"
    if not path.exists():
        log_event(get_logger("ws1.ingest", config), "tables.missing", level=30, doc=doc_id,
                  path=str(path), hint="run `specqa ingest --docs all` first")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Table.from_dict(t) for t in data.get("tables", [])]


def load_figures(config: Any, doc_id: str) -> list[Figure]:
    """Figures of one document; a missing file is reported, not hidden."""
    path = config.path("paths.processed") / doc_id / "figures.json"
    if not path.exists():
        log_event(get_logger("ws1.ingest", config), "figures.missing", level=30, doc=doc_id,
                  path=str(path), hint="run `specqa ingest --docs all` first")
        return []
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Figure.from_dict(f) for f in data.get("figures", [])]


def load_page_images(config: Any, doc_id: str) -> list[PageImage]:
    path = config.path("paths.processed") / doc_id / "pages.jsonl"
    if not path.exists():
        return []
    out: list[PageImage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(PageImage.from_dict(json.loads(line)))
    return out
