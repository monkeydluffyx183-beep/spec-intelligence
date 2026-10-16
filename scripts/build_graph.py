#!/usr/bin/env python3
"""Build the WS3 knowledge graph, validate it, export it and report statistics.

    python scripts/build_graph.py                    # rules + LLM pass (if a local model is up)
    python scripts/build_graph.py --no-llm --docs pd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specintel.common.config import load_config  # noqa: E402
from specintel.common.logging import setup_from_config, get_logger, log_event  # noqa: E402
from specintel.kg.build import build_graph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", default="", help="comma separated doc ids (default: all)")
    ap.add_argument("--no-llm", action="store_true", help="skip the schema-guided LLM pass")
    ap.add_argument("--sample", type=int, default=100, help="triples sampled for the precision check")
    args = ap.parse_args()
    config = load_config()
    setup_from_config(config)
    log = get_logger("ws3.graph", config)
    docs = [d.strip() for d in args.docs.split(",") if d.strip()] or None
    store, report = build_graph(config, doc_ids=docs, use_llm=not args.no_llm, sample_size=args.sample)
    log_event(log, "graph.report", nodes=report.nodes, edges=report.edges, merges=report.merges,
              rejected=report.rejected)
    print("\nper document:")
    for doc_id, info in report.documents.items():
        print(f"  {doc_id:10s} elements={info['elements']:4d} nodes={info['nodes']:4d} edges={info['edges']:4d}")
    print(f"\nnodes={report.nodes} edges={report.edges} merges={report.merges} rejected={report.rejected}")
    print("stats:", {k: v for k, v in report.stats.items() if k in ("orphan_nodes", "stale_references",
                                                                    "cross_document_edges")})
    print("llm pass:", report.llm_pass.get("backend"), report.llm_pass.get("notes", []))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
