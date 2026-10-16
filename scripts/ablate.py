#!/usr/bin/env python3
"""Run one option-level ablation on top of the A-F configuration sweep.

The A-F sweep in ``specintel.eval.run_eval`` changes *architecture* (dense,
hybrid, vision, fused, knowledge graph, agent loop).  An ablation changes one
*option* while the architecture stays fixed, which is how the report answers
"which part of the pipeline is doing the work?" rather than "which architecture
wins?".

Ablations are declared in :data:`ABLATIONS` as ``(label, env, prompt)``; the
runner reloads the configuration for each one (``load_config_fresh``) so a
mutated option cannot leak into the next run, and every run writes to its own
output directory.  Results are collected into ``ablation.json`` next to the
main results.

Usage::

    PYTHONPATH=src python3 scripts/ablate.py                 # all ablations
    PYTHONPATH=src python3 scripts/ablate.py --only reranker  # one of them
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "eval" / "results"

#: label -> (environment overrides, what the ablation removes)
ABLATIONS: dict[str, tuple[dict[str, str], str]] = {
    "reranker": (
        {"SPECINTEL_ABLATE_RERANKER": "1"},
        "drop the lexical reranker from the hybrid retriever (config B)",
    ),
    "bm25": (
        {"SPECINTEL_ABLATE_BM25": "1"},
        "drop BM25 and keep dense retrieval only (config B)",
    ),
    "figure_text": (
        {"SPECINTEL_ABLATE_FIGURE_TEXT": "1"},
        "drop figure descriptions and OCR text from the text index (config E)",
    ),
    "table_rows": (
        {"SPECINTEL_ABLATE_TABLE_ROWS": "1"},
        "drop stitched table bodies from the text index (config E)",
    ),
    "kg_staleness": (
        {"SPECINTEL_ABLATE_STALE": "1"},
        "stop flagging cross-document edges that point at a superseded revision (config E)",
    ),
    "uniform_floor": (
        {"SPECINTEL_SCORE_FLOORS": "0.25,0.25,0.25,0.25"},
        "one retrieval floor for every retriever instead of a per-retriever floor (config D)",
    ),
    "single_pass": (
        {"SPECINTEL_ABLATE_MULTIPART": "1"},
        "answer compound questions in a single generation pass (config F)",
    ),
}

CONFIGS = {"reranker": "B", "bm25": "B", "figure_text": "E", "table_rows": "E",
           "kg_staleness": "E", "uniform_floor": "D", "single_pass": "F"}


def run_one(label: str, *, split: str, limit: int | None, python: str) -> dict[str, object]:
    env, description = ABLATIONS[label]
    out = RESULTS / f"ablation_{label}"
    out.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[label]
    cmd = [python, "-m", "specintel.eval.run_eval", "--configs", config, "--split", split,
           "--out", str(out)]
    if limit:
        cmd += ["--limit", str(limit)]
    child_env = dict(os.environ, PYTHONPATH=str(REPO / "src"), **env)
    print(f"  [{label}] {description}\n    {' '.join(cmd)} (env {env})")
    subprocess.run(cmd, env=child_env, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    overall = summary["aggregates"][config]["overall"]
    return {"label": label, "config": config, "description": description, "env": env,
            "overall": {k: (None if v != v else round(v, 4)) for k, v in overall.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=[], choices=sorted(ABLATIONS))
    ap.add_argument("--split", default="all", choices=["all", "dev", "held_out"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=str(RESULTS / "ablation.json"))
    args = ap.parse_args(argv)

    labels = args.only or list(ABLATIONS)
    print(f"Ablations: {len(labels)} run(s), split={args.split}")
    rows = [run_one(label, split=args.split, limit=args.limit, python=args.python)
            for label in labels]
    baseline = None
    if args.split == "all" and not args.limit:
        main_summary = RESULTS / "summary.json"
        if main_summary.exists():
            data = json.loads(main_summary.read_text(encoding="utf-8"))
            baseline = {name: data["aggregates"][name]["overall"] for name in "ABCDEF"}
    payload = {"ablations": rows, "baseline": baseline, "split": args.split}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {args.out}")
    for row in rows:
        o = row["overall"]
        print(f"  {row['label']:<14} {row['config']} correctness={o.get('correctness')} "
              f"citation={o.get('citation_accuracy')} recall@10={o.get('recall@10')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
