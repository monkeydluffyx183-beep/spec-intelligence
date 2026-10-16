"""WS2 — index build: elements → chunks → BM25 + dense + page-image indexes.

One command builds everything and writes a build report with timings, memory
and index sizes, because the assignment asks for the time/space behaviour of
each stage (index build, per-page embedding, query latency):

    python scripts/build_index.py                 # every document
    python scripts/build_index.py --docs pd,typec
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from specintel.common.logging import get_logger, log_event, log_exception, timed
from specintel.common.models import Chunk, Element, Figure, Table
from specintel.common.resources import rss_mb
from specintel.index.chunker import ChunkOptions, chunk_stats, chunk_corpus
from specintel.index.embeddings import get_embedder
from specintel.index.page_index import MaxSimIndex, get_page_embedder
from specintel.index.reranker import get_reranker
from specintel.index.search import SearchIndex


@dataclass
class BuildReport:
    documents: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"documents": self.documents, "totals": self.totals, "timings": self.timings}


def build_indexes(config: Any, *, doc_ids: Sequence[str] | None = None, elements: Sequence[Element] | None = None,
                  tables_by_doc: dict[str, Sequence[Table]] | None = None,
                  figures_by_doc: dict[str, Sequence[Figure]] | None = None,
                  with_pages: bool = True) -> tuple[SearchIndex, BuildReport]:
    """Build and persist the text, dense and page indexes for the corpus."""
    log = get_logger("ws2.index", config)
    index_dir = config.path("paths.index")
    index_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()

    if elements is None:
        from specintel.ingest.pipeline import load_elements, load_figures, load_tables

        elements = load_elements(config, doc_ids=doc_ids)
        tables_by_doc = {d["doc_id"]: load_tables(config, d["doc_id"]) for d in config.documents()
                         if not doc_ids or d["doc_id"] in doc_ids}
        figures_by_doc = {d["doc_id"]: load_figures(config, d["doc_id"]) for d in config.documents()
                          if not doc_ids or d["doc_id"] in doc_ids}
    tables_by_doc = tables_by_doc or {}
    figures_by_doc = figures_by_doc or {}

    options = ChunkOptions(
        target_tokens=int(config.get("ingest.chunk.target_tokens", 320)),
        overlap_tokens=int(config.get("ingest.chunk.overlap_tokens", 60)),
    )
    t0 = time.perf_counter()
    chunks = chunk_corpus(list(elements), tables_by_doc=tables_by_doc, figures_by_doc=figures_by_doc,
                          options=options)
    report.timings["chunk_s"] = round(time.perf_counter() - t0, 3)
    log_event(log, "chunk.done", **chunk_stats(chunks))

    embedder = get_embedder(config)
    reranker = get_reranker(config)
    t0 = time.perf_counter()
    index = SearchIndex.build(
        chunks, embedder=embedder, reranker=reranker,
        k1=float(config.get("index.bm25.k1", 1.5)), b=float(config.get("index.bm25.b", 0.75)),
        identifier_boost=float(config.get("index.bm25.identifier_boost", 3.0)), elements=list(elements),
    )
    report.timings["text_index_s"] = round(time.perf_counter() - t0, 3)
    log_event(log, "index.text", chunks=len(chunks), bm25_terms=len(index.bm25.postings) if index.bm25 else 0,
              dense_dims=index.dense.dim if index.dense else 0, embedder=embedder.name,
              reranker=reranker.name, duration_s=report.timings["text_index_s"], rss_mb=rss_mb())

    if with_pages:
        index.pages, report.timings["page_index_s"] = _build_page_index(config, elements, index)
        log_event(log, "index.pages", **(index.pages.storage_report() if index.pages else {}))

    # per-document accounting
    for doc in config.documents():
        doc_id = doc["doc_id"]
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]
        if not doc_chunks:
            continue
        report.documents[doc_id] = {
            "chunks": len(doc_chunks), **chunk_stats(doc_chunks),
            "tables": len(tables_by_doc.get(doc_id, [])), "figures": len(figures_by_doc.get(doc_id, [])),
        }
    report.totals = {
        "chunks": len(chunks), "bm25_terms": len(index.bm25.postings) if index.bm25 else 0,
        "dense_vectors": len(index.dense) if index.dense else 0,
        "dense_dims": index.dense.dim if index.dense else 0,
        "pages_indexed": len(index.pages) if index.pages else 0,
        "index_size_mb": round(_dir_size_mb(index_dir), 2),
        "embedder": embedder.name, "page_embedder": getattr(index.pages, "embedder", None).name
        if index.pages else None,
    }

    # persist
    _persist(config, index, chunks, report)
    log_event(log, "index.saved", directory=str(index_dir), **report.totals)
    return index, report


def _build_page_index(config: Any, elements: Sequence[Element], index: SearchIndex
                      ) -> tuple[MaxSimIndex | None, float]:
    """Embed the rendered pages with the page-image model (vision path)."""
    embedder = get_page_embedder(config)
    page_index = MaxSimIndex(embedder=embedder,
                             max_patches=int(config.get("index.page_index.max_patches_per_page", 256)))
    t0 = time.perf_counter()
    doc_ids = {d["doc_id"] for d in config.documents()}
    processed = config.path("paths.processed")
    from specintel.ingest.pipeline import load_page_images

    # the offline stand-in learns each page's text rectangles from the text path
    register = getattr(embedder, "register_page_text", None)
    if callable(register):
        by_page: dict[tuple[str, int], list[tuple[list[float], str]]] = {}
        sizes: dict[tuple[str, int], tuple[float, float]] = {}
        for el in elements:
            if el.bbox:
                by_page.setdefault((el.doc_id, el.page), []).append((el.bbox, el.text))
        for (doc_id, page), lines in by_page.items():
            width = max((b[2] for b, _ in lines), default=612.0)
            height = max((b[3] for b, _ in lines), default=792.0)
            sizes[(doc_id, page)] = (max(width * 1.25, 612.0), max(height * 1.1, 792.0))
            register(doc_id, page, lines, sizes[(doc_id, page)])
        for doc_id, page, size in _page_dims_from_processed(config, doc_ids):
            sizes.setdefault((doc_id, page), size)

    for doc in config.documents():
        doc_id = doc["doc_id"]
        images = load_page_images(config, doc_id)
        if not images:
            continue
        if not Path(images[0].image_path).exists():
            continue
        for image in images:
            try:
                page_index.add_pages([embedder.encode_page(image.image_path, doc_id=doc_id,
                                                           revision=image.revision, page=image.page)])
            except Exception as exc:                      # keep going: a page must not break the build
                log = get_logger("ws2.index", config)
                log_exception(log, "index.page.failed", exc, doc=doc_id, page=image.page)
    page_index.build_time_s = time.perf_counter() - t0
    return page_index, round(page_index.build_time_s, 3)


def _page_dims_from_processed(config: Any, doc_ids: set[str]) -> list[tuple[str, int, tuple[float, float]]]:
    from specintel.ingest.pipeline import load_page_images

    out = []
    for doc_id in doc_ids:
        for image in load_page_images(config, doc_id):
            if image.width and image.height:
                # page images were rendered at `dpi`; convert back to PDF points
                scale = 72.0 / max(image.dpi, 1)
                out.append((doc_id, image.page, (image.width * scale, image.height * scale)))
    return out


def _persist(config: Any, index: SearchIndex, chunks: Sequence[Chunk], report: BuildReport) -> None:
    index_dir = config.path("paths.index")
    index_dir.mkdir(parents=True, exist_ok=True)
    Chunk.write_jsonl(str(index_dir / "chunks.jsonl"), chunks)
    if index.bm25:
        index.bm25.save(index_dir / "bm25.json")
    if index.dense:
        index.dense.save(index_dir, "dense")
    if index.pages:
        index.pages.save(index_dir / "pages")
    (index_dir / "build_report.json").write_text(json.dumps(report.to_dict(), indent=1, default=str),
                                                 encoding="utf-8")


def load_index(config: Any, *, with_pages: bool = True) -> SearchIndex:
    """Load the persisted indexes (chunks, BM25, dense, pages) from disk."""
    index_dir = config.path("paths.index")
    chunks = Chunk.read_jsonl(str(index_dir / "chunks.jsonl")) if (index_dir / "chunks.jsonl").exists() else []
    from specintel.ingest.pipeline import load_elements

    index = SearchIndex(chunks=chunks, elements=load_elements(config), index_dir=index_dir,
                        embedder=get_embedder(config), reranker=get_reranker(config))
    if (index_dir / "bm25.json").exists():
        from specintel.index.bm25 import BM25Index

        index.bm25 = BM25Index.load(index_dir / "bm25.json")
    if (index_dir / "dense.json").exists():
        index.dense = VectorIndex.load(index_dir, "dense")
    if with_pages and (index_dir / "pages" / "pages.json").exists():
        payload = json.loads((index_dir / "pages" / "pages.json").read_text(encoding="utf-8"))
        embedder = get_page_embedder(config, backend=_backend_for_payload(config, payload))
        page_index = MaxSimIndex(embedder=embedder, dim=int(payload.get("dim", 128)))
        import numpy as np

        arrays = np.load(index_dir / "pages" / "pages.npz")
        from specintel.index.page_index import PageVectors

        for i, meta in enumerate(payload.get("pages", [])):
            vectors = arrays.get(f"page_{i}")
            if vectors is None:
                continue
            page_index.pages.append(PageVectors(doc_id=meta["doc_id"], revision=meta.get("revision", ""),
                                                page=int(meta["page"]), vectors=vectors,
                                                image_path=meta.get("image_path", "")))
        index.pages = page_index
    return index


def _backend_for_payload(config: Any, payload: dict[str, Any]) -> str:
    """Reload the same page embedder that produced the saved vectors."""
    name = str(payload.get("embedder", "hashing"))
    if name.startswith("hashing"):
        return "hashing"
    return str(config.get("index.page_index.backend", "auto"))


def _dir_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)


__all__ = ["build_indexes", "load_index", "BuildReport", "VectorIndex"]

from specintel.index.dense import VectorIndex  # noqa: E402  (kept for type-aware imports)
