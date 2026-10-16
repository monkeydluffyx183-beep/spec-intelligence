#!/usr/bin/env python3
"""Generate the license-free stand-in corpus (real PDFs + manifest + gold set).

Usage
-----
    python scripts/make_synthetic_corpus.py                 # data/synthetic + eval/gold.jsonl
    python scripts/make_synthetic_corpus.py --out data/synthetic

Run this when the licensed USB-IF / WPC PDFs are not available: the pipeline,
the OCR benchmark, the index, the knowledge graph and the evaluation all run
against the generated documents, which reproduce the structural traps of the
real corpus (multi-page tables, vector state diagrams, raster figures, per-book
numbering, stale cross-references).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specintel.common.config import load_config  # noqa: E402
from specintel.common.logging import get_logger, log_event, setup_from_config  # noqa: E402
from specintel.common.resources import hardware_profile, rss_mb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output directory (defaults to paths.synthetic)")
    parser.add_argument("--gold", default="eval/gold.jsonl", help="gold question file to write")
    parser.add_argument("--seed", default="eval/gold_seed.jsonl", help="Appendix A seed questions file")
    args = parser.parse_args()

    config = load_config()
    setup_from_config(config)
    log = get_logger("ws1.synthetic", config)
    out = Path(args.out) if args.out else config.path("paths.synthetic")

    from specintel.synthetic.corpus import build_corpus
    from specintel.synthetic.gold import build_seed_questions

    log_event(log, "synthetic.start", out=str(out), rss_mb=rss_mb(), hardware=hardware_profile())
    corpus = build_corpus(out, gold_path=config.root / args.gold, log=log)

    seed_path = config.root / args.seed
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    seeds = build_seed_questions()
    seed_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n", encoding="utf-8")

    log_event(log, "synthetic.done", documents=len(corpus.documents), elements=len(corpus.manifest),
              gold=len(corpus.gold), seed=len(seeds), rss_mb=rss_mb())
    print(f"\nGenerated {len(corpus.documents)} documents, {len(corpus.manifest)} manifest elements")
    for doc in corpus.documents:
        print(f"  {doc['doc_id']:10s} pages={doc['pages']:3d} elements={doc['elements']:4d}  "
              f"{Path(doc['files'][0]).name}")
    print(f"\n  gold set      -> {config.root / args.gold} ({len(corpus.gold)} questions)")
    print(f"  seed questions-> {seed_path} ({len(seeds)} questions from Appendix A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
