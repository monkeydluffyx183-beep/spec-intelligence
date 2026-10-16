"""WS4 — the ``specqa`` command line interface.

One entry point for the whole pipeline, so the system can be driven from a clean
clone without remembering module paths::

    specqa ask "What does PS_RDY signal?" --trace
    specqa ask "…" --docs pd,typec --json
    specqa build-index --docs pd,typec
    specqa build-graph
    specqa ingest --docs all
    specqa eval --configs A,B,C,D,E,F --split all
    specqa tools                       # list the agent's tools
    specqa graph --name tSenderResponse
    specqa serve --port 8000           # demo UI + HTTP API
    specqa doctor                      # what is built, what is missing

Every subcommand honours ``--config`` / ``SPECINTEL_CONFIG`` and writes the
structured log lines described in ``docs/INTERFACES.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from specintel.common.config import Config, load_config, load_config_fresh
from specintel.common.logging import get_logger, log_event, log_exception, setup_from_config
from specintel.common.resources import hardware_profile

DOCS = ("pd", "pd_cts", "scc_cts", "typec", "typec_fts", "usb4", "tech2")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_docs(value: str | None) -> list[str] | None:
    if not value or value == "all":
        return None
    return [d.strip() for d in value.split(",") if d.strip()]


def _print_trace(result: Any) -> None:
    """Print the run's trace: the plan, each tool call, the answer, the check."""
    trace = getattr(result, "trace", None)
    if trace is None:
        print("  (no trace recorded)")
        return
    steps = [s.to_dict() if hasattr(s, "to_dict") else s for s in (getattr(trace, "steps", []) or [])]
    print("\n  trace")
    plan = getattr(trace, "plan", None) or []
    if plan:
        print(f"    plan      : {' | '.join(str(p) for p in plan[:6])}")
    for step in steps:
        name = step.get("name") or step.get("action") or step.get("kind") or "?"
        detail = step.get("observation") or step.get("detail") or ""
        extra = []
        if step.get("n_hits") is not None:
            extra.append(f"{step['n_hits']} hits")
        if step.get("top_score") is not None:
            extra.append(f"top={float(step['top_score']):.3f}")
        if step.get("tokens"):
            extra.append(f"{step['tokens']} tok")
        if step.get("duration_s") is not None:
            extra.append(f"{float(step['duration_s']):.3f}s")
        print(f"    {str(name)[:22]:<22} {' '.join(extra):<28} {str(detail)[:110]}")
    usage = getattr(trace, "token_usage", {}) or {}
    print(f"    tokens    : {usage.get('total', usage.get('total_tokens', 0))}"
          f"   latency: {getattr(trace, 'latency_s', 0.0):.2f}s"
          f"   model: {getattr(trace, 'model', '')}"
          f"   tools: {getattr(result, 'tool_calls', 0)}")


def _print_answer(result: Any) -> None:
    status = "DECLINED" if getattr(result, "declined", False) else "ANSWERED"
    print(f"\n{status}")
    print(getattr(result, "answer", "").strip() or "(empty)")
    citations = getattr(result, "citations", []) or []
    if citations:
        print("\n  cited sources")
        for citation in citations:
            marker = citation.to_marker() if hasattr(citation, "to_marker") else str(citation)
            print(f"    {marker}")
    if getattr(result, "declined", False) and getattr(result, "decline_reason", ""):
        print(f"\n  reason: {result.decline_reason}")


def _answer_payload(result: Any) -> dict[str, Any]:
    trace = getattr(result, "trace", None)
    return {
        "question": getattr(result, "question", ""),
        "answer": getattr(result, "answer", ""),
        "declined": bool(getattr(result, "declined", False)),
        "decline_reason": getattr(result, "decline_reason", ""),
        "citations": [c.to_marker() if hasattr(c, "to_marker") else str(c)
                      for c in (getattr(result, "citations", []) or [])],
        "tool_calls": getattr(result, "tool_calls", 0),
        "latency_s": round(float(getattr(trace, "latency_s", 0.0) or 0.0), 3),
        "evidence": [
            {"doc_id": h.doc_id, "page": h.page, "section": h.section_number,
             "element_type": h.element_type, "score": round(float(h.score), 4),
             "table_id": h.table_id, "figure_id": h.figure_id}
            for h in (getattr(result, "evidence", []) or [])
        ],
        "trace": trace.to_dict() if hasattr(trace, "to_dict") else None,
    }


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_ask(config: Config, args: argparse.Namespace) -> int:
    from specintel.agent.loop import AgentConfig, build_agent

    agent_config = AgentConfig(
        max_steps=args.max_steps, budget_s=args.budget, top_k=args.top_k,
        verify_citations=not args.no_verify,
    )
    agent = build_agent(config, agent_config=agent_config)
    result = agent.answer(args.question, docs=_parse_docs(args.docs),
                          question_type=args.type)
    if args.json:
        print(json.dumps(_answer_payload(result), indent=2))
        return 0 if not getattr(result, "declined", False) else 3
    if args.quiet:
        print(getattr(result, "answer", "").strip())
        return 0
    _print_answer(result)
    if args.trace:
        _print_trace(result)
    return 0 if not getattr(result, "declined", False) else 3


def cmd_tools(config: Config, args: argparse.Namespace) -> int:
    from specintel.agent.prompts import TOOL_PROMPTS
    from specintel.agent.tools import TOOL_SPECS

    print(f"\n  {len(TOOL_PROMPTS)} tools\n")
    for spec in TOOL_PROMPTS:
        print(f"  {spec['name']}({spec['arguments']})")
        for line in _wrap(spec["description"], 96, indent=6):
            print(line)
        print()
    if args.verbose:
        print("  registry")
        for name, spec in sorted(TOOL_SPECS.items()):
            print(f"    {name:<16} {spec.get('signature', '') if isinstance(spec, dict) else spec}")
    return 0


def _wrap(text: str, width: int, indent: int) -> list[str]:
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(" " * indent + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(" " * indent + line)
    return out


def cmd_ingest(config: Config, args: argparse.Namespace) -> int:
    import subprocess

    cmd = [sys.executable, str(config.root / "scripts" / "ingest_corpus.py")]
    if args.docs:
        cmd += ["--docs", args.docs]
    if args.parser:
        cmd += ["--parser", args.parser]
    if args.ocr:
        cmd += ["--ocr", args.ocr]
    if args.stats:
        cmd += ["--stats"]
    return subprocess.call(cmd, cwd=config.root)


def cmd_build_index(config: Config, args: argparse.Namespace) -> int:
    import subprocess

    cmd = [sys.executable, str(config.root / "scripts" / "build_index.py")]
    if args.docs:
        cmd += ["--docs", args.docs]
    return subprocess.call(cmd, cwd=config.root)


def cmd_build_graph(config: Config, args: argparse.Namespace) -> int:
    import subprocess

    cmd = [sys.executable, str(config.root / "scripts" / "build_graph.py")]
    if args.docs:
        cmd += ["--docs", args.docs]
    if args.no_llm:
        cmd += ["--no-llm"]
    if args.sample:
        cmd += ["--sample", str(args.sample)]
    return subprocess.call(cmd, cwd=config.root)


def cmd_eval(config: Config, args: argparse.Namespace) -> int:
    from specintel.eval.run_eval import main as eval_main

    argv = ["--configs", args.configs, "--split", args.split, "--out", args.out]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.llm:
        argv += ["--llm", args.llm]
    return eval_main(argv)


def cmd_graph(config: Config, args: argparse.Namespace) -> int:
    from specintel.agent.tools import Toolkit

    toolkit = Toolkit.from_config(config)
    if args.stats:
        result = toolkit.call("graph_query", kind="stats")
    elif args.name:
        result = toolkit.call("graph_query", kind=args.kind, name=args.name, depth=args.depth)
    else:
        result = toolkit.call("graph_query", kind=args.kind, src=args.src, dst=args.dst)
    if args.json:
        print(json.dumps(result.to_dict() if hasattr(result, "to_dict") else result.graph,
                         indent=2, default=str))
    else:
        print(result.text or "(empty)")
    return 0 if result.graph else 2


def cmd_doctor(config: Config, args: argparse.Namespace) -> int:
    """Report what is built and what is missing, in the order it is built."""
    checks: list[tuple[str, bool, str]] = []

    def path_check(label: str, path: Path, hint: str) -> None:
        exists = path.exists()
        detail = hint if not exists else _describe_path(path)
        checks.append((label, exists, detail))

    processed = config.path("paths.processed")
    index_dir = config.path("paths.index")
    graph_dir = config.path("paths.graph")
    manifest = config.path("paths.synthetic") / "manifest.json"
    path_check("synthetic corpus", manifest, "python3 scripts/make_synthetic_corpus.py")
    path_check("ingested elements", processed, "specqa ingest --docs all")
    path_check("text index", index_dir / "chunks.jsonl", "specqa build-index")
    path_check("knowledge graph", graph_dir / "graph.sqlite", "specqa build-graph")
    path_check("gold set", config.root / "eval" / "gold.jsonl", "python3 scripts/make_synthetic_corpus.py")
    path_check("eval results", config.root / "eval" / "results" / "summary.json", "specqa eval")

    from specintel.common.llm import get_llm

    llm = get_llm(config, "auto")
    backend = f"{getattr(llm, 'name', 'none')} ({getattr(llm, 'backend', 'none')})"
    print("\n  spec-intelligence environment\n")
    for label, ok, detail in checks:
        mark = "ok  " if ok else "--  "
        print(f"    {mark}{label:<20} {detail}")
    profile = hardware_profile()
    gpu = profile.get("gpu") or "none"
    print(f"\n    hardware: {profile.get('platform')}, {profile.get('cpu_count')} cpu, "
          f"{profile.get('ram_gb')} GB RAM, gpu={gpu}, python {profile.get('python')}")
    if getattr(llm, "backend", "") == "extractive":
        print(f"    llm backend: {backend} — deterministic fallback, no Ollama reachable at "
              f"{config.get('agent.llm.endpoint', 'http://localhost:11434')}")
    else:
        print(f"    llm backend: {backend} (local open-weight model)")
    missing = [label for label, ok, _ in checks if not ok]
    if missing:
        print(f"\n  missing: {', '.join(missing)}\n  run the hint command for each, in order.")
        return 2
    print("\n  everything needed for `specqa ask` and `specqa eval` is present.\n")
    return 0


def _describe_path(path: Path) -> str:
    if path.is_dir():
        try:
            return f"{path} ({len(list(path.iterdir()))} entries)"
        except OSError:                                          # pragma: no cover - permissions
            return str(path)
    size = path.stat().st_size
    return f"{path} ({size / 1024:.0f} KB)"


def cmd_serve(config: Config, args: argparse.Namespace) -> int:
    from specintel.server.app import serve

    return serve(config, host=args.host, port=args.port)


def cmd_chunks(config: Config, args: argparse.Namespace) -> int:
    """Inspect the index without starting the agent (used while ingesting)."""
    from specintel.agent.tools import Toolkit

    toolkit = Toolkit.from_config(config)
    hits = toolkit.index.search_text(args.query, k=args.k, filters=_filters(args))
    print(f"\n  {len(hits)} hits for {args.query!r}\n")
    for hit in hits:
        print(f"  {hit.score:8.3f}  {hit.doc_id} §{hit.section_number or '—'} p.{hit.page} "
              f"[{hit.element_type}] {hit.table_id or hit.figure_id or ''}")
        for line in _wrap(str(hit.text).split("\n")[0], 92, 12):
            print(line)
    return 0


def _filters(args: argparse.Namespace) -> Any:
    from specintel.index.search import Filters

    return Filters(docs=tuple(_parse_docs(getattr(args, "docs", None)) or ()),
                   tables_only=bool(getattr(args, "tables_only", False)),
                   figures_only=bool(getattr(args, "figures_only", False)))


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specqa", description="Spec Intelligence — ask the USB specs a question.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="ask one question and print the cited answer")
    p.add_argument("question")
    p.add_argument("--docs", default=None, help="comma-separated document ids, or 'all'")
    p.add_argument("--trace", action="store_true", help="print the agent trace")
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    p.add_argument("--quiet", action="store_true", help="print only the answer text")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--budget", type=float, default=30.0, help="seconds")
    p.add_argument("--type", default=None, help="force the question type")
    p.add_argument("--no-verify", action="store_true", help="skip citation verification")
    p.set_defaults(func=cmd_ask, needs_index=True)

    p = sub.add_parser("tools", help="list the agent's tools")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("ingest", help="run the multimodal ingestion pipeline")
    p.add_argument("--docs", default=None)
    p.add_argument("--parser", default="auto")
    p.add_argument("--ocr", default="auto")
    p.add_argument("--stats", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("build-index", help="build the BM25 + dense + page index")
    p.add_argument("--docs", default=None)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("build-graph", help="build the knowledge graph")
    p.add_argument("--docs", default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--sample", type=int, default=100)
    p.set_defaults(func=cmd_build_graph)

    p = sub.add_parser("eval", help="run the evaluation harness")
    p.add_argument("--configs", default="A,B,C,D,E,F")
    p.add_argument("--split", default="all", choices=["all", "dev", "held_out"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="eval/results")
    p.add_argument("--llm", default="")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("graph", help="query the knowledge graph")
    p.add_argument("--name", default="")
    p.add_argument("--src", default="")
    p.add_argument("--dst", default="")
    p.add_argument("--kind", default="neighbors",
                   choices=["lookup", "neighbors", "path", "subgraph", "stats"])
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--stats", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("chunks", help="debug retrieval: show what the index returns")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--docs", default=None)
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    p.set_defaults(func=cmd_chunks, needs_index=True)

    p = sub.add_parser("doctor", help="report which artefacts are present")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="run the demo web UI and HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve, needs_index=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config_fresh(args.config) if getattr(args, "needs_index", False) else load_config(args.config)
    if args.log_level:
        config.data.setdefault("logging", {})["level"] = args.log_level
    setup_from_config(config)
    logger = get_logger("cli", config)
    started = time.perf_counter()
    try:
        code = args.func(config, args)
    except FileNotFoundError as exc:
        log_event(logger, "cli.missing_artifact", command=args.command, error=str(exc))
        print(f"\n  missing artefact: {exc}\n  run `specqa doctor` to see what to build first.\n",
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:                                    # pragma: no cover - interactive
        print("\n  interrupted\n", file=sys.stderr)
        return 130
    except Exception as exc:                                     # noqa: BLE001 - report, do not trace
        log_exception(logger, "cli.failed", exc, command=args.command)
        print(f"\n  {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 1
    log_event(logger, "cli.done", command=args.command, status="ok" if code == 0 else "error",
              duration_s=round(time.perf_counter() - started, 3))
    return code


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
