#!/usr/bin/env python3
"""WS1 entry point: ingest the corpus into typed, page-anchored elements.

Usage
-----
    python scripts/ingest_corpus.py                      # every document in the registry
    python scripts/ingest_corpus.py --docs pd,typec      # a subset
    python scripts/ingest_corpus.py --no-page-images     # skip page rasterisation
    python scripts/ingest_corpus.py --stats              # print per-document counts only

``resolve_documents`` prefers the real documents dropped into ``data/raw/<doc_id>``
and falls back to the generated stand-in corpus in ``data/synthetic``.  The
licensed USB-IF / WPC PDFs are never committed (see README), so a clean clone
runs this script against the stand-in corpus and reports which source it used.
Everything downstream — index, knowledge graph, gold set, evaluation — reads the
artefacts written under ``paths.processed``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specintel.common.config import load_config  # noqa: E402
from specintel.common.logging import get_logger, log_event, setup_from_config  # noqa: E402
from specintel.common.resources import hardware_profile, rss_mb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default=None, help="comma separated doc ids (default: all)")
    parser.add_argument("--parser", default="auto", choices=["auto", "pymupdf", "pypdf", "plain"],
                        help="text/table parser backend")
    parser.add_argument("--ocr", default=None, choices=["auto", "none", "tesseract", "vlm"],
                        help="OCR backend for raster regions (default: config ingest.ocr.mode)")
    parser.add_argument("--no-page-images", action="store_true", help="skip page image rendering")
    parser.add_argument("--stats", action="store_true", help="also print the per-document element mix")
    args = parser.parse_args()

    config = load_config()
    setup_from_config(config)
    log = get_logger("ws1.ingest", config)

    doc_ids = [d.strip() for d in (args.docs or "").split(",") if d.strip()] or None
    from specintel.ingest.pipeline import ingest_corpus, resolve_documents

    resolved = {d["doc_id"]: d for d in resolve_documents(config)}
    sources = {doc_id: info.get("source", "unknown") for doc_id, info in resolved.items()}
    log_event(log, "ingest.start", docs=doc_ids or "all", parser=args.parser, ocr=args.ocr or "config",
              page_images=not args.no_page_images, sources=json.dumps(sources, separators=(",", ":")),
              rss_mb=rss_mb(), hardware=hardware_profile())

    t0 = time.perf_counter()
    results = ingest_corpus(config, doc_ids=doc_ids, parser_name=args.parser, ocr=args.ocr,
                            keep_images=not args.no_page_images)
    duration = time.perf_counter() - t0

    elements = sum(len(r.elements) for r in results.values())
    tables = sum(len(r.tables) for r in results.values())
    stitched = sum(len(r.stitched_tables) for r in results.values())
    figures = sum(len(r.figures) for r in results.values())
    ocr_figures = sum(1 for r in results.values() for f in r.figures if getattr(f, "ocr_text", ""))
    page_images = sum(len(r.page_images) for r in results.values())
    problems = [p for r in results.values() for p in r.problems]

    print(f"\nIngested {len(results)} documents in {duration:.2f} s")
    print(f"  elements        {elements}")
    print(f"  tables          {tables} (stitched across pages: {stitched})")
    print(f"  figures         {figures} ({ocr_figures} with OCR text)")
    print(f"  page images     {page_images}")
    print(f"  problems        {len(problems)}")
    for problem in problems[:5]:
        print(f"    ! {problem}")
    print("\n  per document")
    for doc_id, result in sorted(results.items()):
        print(f"    {doc_id:10s} source={sources.get(doc_id, '?'):9s} pages={result.pages:3d} "
              f"elements={len(result.elements):4d} tables={len(result.tables):3d} "
              f"figures={len(result.figures):2d} stitched={len(result.stitched_tables)}")
    if args.stats:
        for doc_id, result in sorted(results.items()):
            print(f"\n  {doc_id}: {json.dumps(result.counts, default=str)[:400]}")
    log_event(log, "ingest.script_done", documents=len(results), elements=elements, tables=tables,
              figures=figures, problems=len(problems), duration_s=round(duration, 3), rss_mb=rss_mb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
