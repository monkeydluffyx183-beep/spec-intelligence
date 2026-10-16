#!/usr/bin/env python3
"""Build the WS2 indexes (chunks, BM25, dense, page images) and report timings.

    python scripts/build_index.py                 # every document
    python scripts/build_index.py --docs pd,typec --no-pages
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specintel.common.config import load_config  # noqa: E402
from specintel.common.logging import get_logger, log_event, setup_from_config  # noqa: E402
from specintel.index.build import build_indexes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", default="", help="comma separated doc ids (default: all)")
    ap.add_argument("--no-pages", action="store_true", help="skip the vision path")
    args = ap.parse_args()
    config = load_config()
    setup_from_config(config)
    log = get_logger("ws2.index", config)
    docs = [d.strip() for d in args.docs.split(",") if d.strip()] or None
    index, report = build_indexes(config, doc_ids=docs, with_pages=not args.no_pages)
    log_event(log, "index.report", **report.totals)
    print("\nper document:")
    for doc_id, info in report.documents.items():
        print(f"  {doc_id:10s} chunks={info['chunks']:4d} mean_tokens={info['mean_tokens']:6.1f} "
              f"kinds={info['by_kind']}")
    print("\ntotals:", report.totals)
    print("timings:", report.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
