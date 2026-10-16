"""WS5 — the evaluation driver: gold set in, metrics and reports out.

Usage
-----
    python -m specintel.eval.run_eval --configs A,B,E,F --split dev
    python scripts/run_eval.py --configs F --split held_out --limit 20

What it does
------------
1. Loads the gold set (``eval/gold.jsonl``), the persisted index and the graph.
2. Runs every requested configuration over every question:

   =========  ==================================================================
   config     pipeline
   =========  ==================================================================
   A          naive dense RAG: dense retrieval, no rerank, extractive answer
   B          + hybrid BM25 + dense with the lexical reranker
   C          vision only: page-image retrieval (the capstone-04 path)
   D          B + C fused with reciprocal rank fusion
   E          D + the knowledge-graph tool
   F          E + the agent loop (planning, reflection, citation verification)
   =========  ==================================================================

3. Scores recall@5/@10, MRR, nDCG@5, answer correctness (key-point recall and
   token F1), faithfulness, citation accuracy and refusal accuracy, per question,
   per category and per content type — the Task 5 matrix.
4. Audits the knowledge graph (triple rules + annotated precision when the sample
   has been reviewed) and scores the structural accuracy of tables and figures
   against the generator manifest.
5. Writes ``eval/results/*``: per-config JSON, raw answers, agent traces, the
   ablation comparison, the error analysis and a Markdown report that quotes the
   target metrics next to the measured values, with latency percentiles and the
   hardware profile.

Every number in the report is produced by this file from files under ``data/``
and ``eval/`` — nothing is hand-entered, so the report can be regenerated from a
clean clone (after ``make_synthetic_corpus`` + ``ingest`` + ``build_index`` +
``build_graph``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.agent.loop import AgentConfig, SpecAgent, SYSTEM_PROMPT
from specintel.agent.tools import Toolkit, citation_from_hit
from specintel.agent.verifier import CitationVerifier
from specintel.common.llm import ExtractiveLLM, generate, get_llm
from specintel.common.logging import get_logger, log_event, log_exception, setup_from_config
from specintel.common.models import AgentTrace, SearchHit, TraceStep, new_id
from specintel.common.resources import hardware_profile, rss_mb
from specintel.eval import metrics
from specintel.eval.kg_precision import audit_triples, evaluate_edge_provenance, load_triple_sample
from specintel.eval.structure import evaluate_structure

#: target metrics from the assignment, quoted next to the measured values
TARGETS: dict[str, float] = {
    "recall@10": 0.80,
    "correctness": 0.75,
    "faithfulness": 0.85,
    "citation_accuracy": 0.80,
    "refusal_accuracy": 0.80,
    "kg_triple_precision": 0.80,
    "table_cell_accuracy": 0.80,
    "figure_structure_accuracy": 0.80,
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "A": "naive dense RAG (dense-only retrieval, no rerank)",
    "B": "hybrid BM25 + dense with lexical reranker",
    "C": "vision-only: page-image retrieval",
    "D": "B + C fused with reciprocal rank fusion",
    "E": "D + knowledge-graph tool",
    "F": "E + agent loop (planning, reflection, citation verification)",
}


@dataclass
class QuestionOutcome:
    """Everything measured for one question in one configuration."""

    qid: str
    config: str
    question: str
    category: str
    content_type: str
    answerable: bool
    split: str = "dev"
    answer: str = ""
    declined: bool = False
    decline_reason: str = ""
    citations: list[str] = field(default_factory=list)
    gold_citations: list[str] = field(default_factory=list)
    n_evidence: int = 0
    tool_calls: int = 0
    latency_s: float = 0.0
    tokens: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    evidence_locs: list[str] = field(default_factory=list)
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    trace: Any = field(default=None, repr=False)          # AgentTrace, written to traces_<config>.jsonl

    def to_dict(self) -> dict[str, Any]:
        return {"qid": self.qid, "config": self.config, "question": self.question,
                "category": self.category, "content_type": self.content_type, "answerable": self.answerable,
                "split": self.split, "answer": self.answer, "declined": self.declined,
                "decline_reason": self.decline_reason, "citations": self.citations,
                "gold_citations": self.gold_citations, "n_evidence": self.n_evidence,
                "tool_calls": self.tool_calls, "latency_s": round(self.latency_s, 4), "tokens": self.tokens,
                "scores": {k: (None if v != v else round(v, 4)) for k, v in self.scores.items()},
                "evidence": self.evidence_locs, "gold_evidence": self.gold_evidence}


class RetrievalOnlyAgent:
    """Configurations A–D: retrieval plus composition, no planning and no graph.

    The same composer, the same citation machinery and the same decline rule as
    the full agent are used, so the ablation isolates *retrieval* (A→B→C→D)
    rather than conflating it with answer handling.
    """

    #: BM25 and the lexical reranker produce scores in the 1-4 range, cosine
    #: similarity and reciprocal rank fusion in the 0-1 range.  A single
    #: absolute threshold would make the dense and fused configurations decline
    #: every question, which is a measurement artefact, not a finding.
    #: A dense hit's score is a cosine similarity, BM25 and the lexical reranker
    #: return scores in the 1-4 range, and RRF returns 1/(60+rank) sums.  The same
    #: absolute threshold cannot be used for all three, so each retriever carries
    #: its own floor.  Set SPECINTEL_SCORE_FLOORS="0.25,0.25,0.25,0.25" to hold the
    #: floor constant instead — that is the "one threshold for every retriever"
    #: ablation reported in the evaluation report.
    SCORE_FLOORS: dict[str, float] = {"dense": 0.05, "hybrid": 0.25, "vision": 0.05, "fused": 0.005}

    def __init__(self, toolkit: Toolkit, llm: Any, *, mode: str, top_k: int = 8,
                 config: Any | None = None, min_score: float | None = None) -> None:
        self.tools = toolkit
        self.llm = llm
        self.mode = mode                    # dense | hybrid | vision | fused
        self.top_k = top_k
        self.config = config
        self.min_score = self.SCORE_FLOORS.get(mode, 0.25) if min_score is None else min_score
        #: True when the floor was set from outside (a test or the
        #: ``uniform_floor`` ablation) rather than taken from the calibrated
        #: defaults.  An explicit floor is obeyed exactly; a default floor yields
        #: to the ranking, because an ablation that switches a *component* off may
        #: change the retriever's score scale and then the calibrated number
        #: measures the threshold instead of the component (see D-19).
        self.floor_is_explicit = min_score is not None
        # one option change, no code change: the same floor for every retriever
        override = getattr(config, "_score_floor_override", None)
        if override is not None and min_score is None:
            self.min_score = float(override)
            self.floor_is_explicit = True
        self.verifier = CitationVerifier(drop_unsupported=False)

    def retrieve(self, question: str) -> list[SearchHit]:
        if self.mode == "dense":
            return list(self.tools.index.search_text(question, k=self.top_k, hybrid=False, rerank=False))
        if self.mode == "hybrid":
            return list(self.tools.index.search_text(question, k=self.top_k, hybrid=True, rerank=True))
        if self.mode == "vision":
            raw = self.tools.index.search_pages(question, k=self.top_k)
            from specintel.agent.tools import _page_hit

            return [_page_hit(item) for item in raw]
        return list(self.tools.index.search_fused(question, k=self.top_k))

    def answer(self, question: str, *, docs: Sequence[str] | None = None,
               question_type: str | None = None) -> Any:
        started = time.perf_counter()
        trace = AgentTrace(run_id=new_id("run"), question=question, config_name=self.mode,
                           plan=[f"retrieval-only ({self.mode})"], model=getattr(self.llm, "name", "extractive"),
                           hardware=hardware_profile())
        try:
            hits = self.retrieve(question)
        except Exception as exc:                                # pragma: no cover - defensive
            log_exception(get_logger("ws5.eval", self.config), "eval.retrieve.failed", exc)
            hits = []
        trace.steps.append(TraceStep(step=1, kind="tool", name=f"retrieve:{self.mode}", n_hits=len(hits),
                                     top_score=max((h.score for h in hits), default=0.0) or None,
                                     observation=f"{len(hits)} hits from the {self.mode} retriever"))
        from specintel.agent.loop import AgentAnswer

        strong = [h for h in hits if h.score >= self.min_score]
        if not strong and hits and not self.floor_is_explicit:
            top = max(h.score for h in hits)
            if top > 0:
                strong = [h for h in hits if h.score >= 0.25 * top]
                trace.warnings.append(
                    f"calibrated floor {self.min_score} is above this retriever's score scale "
                    f"(best {top:.3f}); used the relative floor 0.25 × best")
        if not hits or not strong:
            answer = (f"INSUFFICIENT — the {self.mode} retriever returned no evidence above the threshold "
                      f"for this question. Declining rather than answering from model knowledge.")
            trace.answer, trace.declined, trace.decline_reason = answer, True, "no evidence above threshold"
            trace.latency_s = time.perf_counter() - started
            return AgentAnswer(question=question, answer=answer, declined=True,
                               decline_reason="no evidence above threshold", evidence=hits, trace=trace,
                               tool_calls=1)
        ranked = sorted(strong, key=lambda h: -h.score)[: max(self.top_k, 6)]
        citations = [citation_from_hit(h) for h in ranked]
        payload = [{"text": h.text, "citation": citation_from_hit(h).to_marker()} for h in ranked]
        prompt = (f"Question: {question}\n\n" + _evidence_blocks(ranked)
                  + "\n\nAnswer the question, citing each sentence.")
        response = generate(self.llm, prompt, evidence=payload, question=question, system=SYSTEM_PROMPT)
        text = (response.text or "").strip()
        if not text:
            answer = ("INSUFFICIENT — no sentence in the retrieved evidence answered the question. "
                      "Declining rather than answering from model knowledge.")
            trace.answer, trace.declined, trace.decline_reason = answer, True, "no sentence answered"
            trace.latency_s = time.perf_counter() - started
            return AgentAnswer(question=question, answer=answer, declined=True,
                               decline_reason="no sentence answered", evidence=hits, trace=trace, tool_calls=1)
        trace.answer, trace.citations = text, citations
        trace.token_usage = {"total": response.total_tokens}
        trace.latency_s = time.perf_counter() - started
        return AgentAnswer(question=question, answer=text, citations=citations, evidence=list(hits),
                           trace=trace, tool_calls=1)


def _evidence_blocks(hits: Sequence[SearchHit]) -> str:
    """Render evidence for a prompt, each block labelled with its citation marker."""
    blocks = [f"Evidence [{citation_from_hit(h).to_marker()}] ({h.element_type}, score {h.score:.3f})\n"
              f"{h.text[:1500]}" for h in hits]
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# running one configuration
# --------------------------------------------------------------------------- #
def build_runner(config: Any, name: str, *, toolkit: Toolkit, llm: Any, top_k: int) -> Any:
    """Return the runner for a configuration name."""
    if name in ("A", "B", "C", "D"):
        mode = {"A": "dense", "B": "hybrid", "C": "vision", "D": "fused"}[name]
        return RetrievalOnlyAgent(toolkit, llm, mode=mode, top_k=top_k, config=config)
    if name == "E":
        agent_config = AgentConfig(max_steps=6, top_k=top_k, reflect_rounds=0, verify_citations=False,
                                   config_name="E")
        return SpecAgent(toolkit, llm=llm, config=config, agent_config=agent_config)
    agent_config = AgentConfig(max_steps=10, top_k=top_k, reflect_rounds=1, verify_citations=True,
                               config_name="F")
    return SpecAgent(toolkit, llm=llm, config=config, agent_config=agent_config)


def run_config(config: Any, name: str, gold: Sequence[dict[str, Any]], *, toolkit: Toolkit, llm: Any,
               top_k: int = 8, limit: int | None = None) -> list[QuestionOutcome]:
    """Run one configuration over the gold questions."""
    log = get_logger("ws5.eval", config)
    runner = build_runner(config, name, toolkit=toolkit, llm=llm, top_k=top_k)
    outcomes: list[QuestionOutcome] = []
    questions = list(gold)[: limit] if limit else list(gold)
    log_event(log, "eval.config.start", config=name, questions=len(questions),
              description=CONFIG_DESCRIPTIONS.get(name, ""))
    for row in questions:
        question = row["question"]
        answerable = bool(row.get("answerable", True))
        evidence_gold = list(row.get("gold_evidence") or [])
        started = time.perf_counter()
        try:
            result = runner.answer(question, docs=row.get("docs") or None)
        except Exception as exc:                                # noqa: BLE001 - one failure must not stop the run
            log_exception(log, "eval.question.failed", exc, config=name, qid=row.get("qid"))
            result = None
        latency = time.perf_counter() - started
        hits = list(getattr(result, "evidence", []) or [])
        answer_text = getattr(result, "answer", "") or ""
        declined = bool(getattr(result, "declined", True))
        citations = [c.to_marker() for c in (getattr(result, "citations", []) or [])]

        scores: dict[str, float] = {
            "recall@5": metrics.recall_at_k(hits, evidence_gold, 5),
            "recall@10": metrics.recall_at_k(hits, evidence_gold, 10),
            "mrr": metrics.reciprocal_rank(hits, evidence_gold),
            "ndcg@5": metrics.ndcg_at_k(hits, evidence_gold, 5),
            # a decline has no claims to be faithful to; refusal accuracy scores it
            "faithfulness": (float("nan") if declined else metrics.faithfulness(answer_text, hits)),
        }
        gold_context = " ".join(str(e.get("quote", "")) for e in evidence_gold)
        # what the answer could have drawn on: the context the agent retrieved for
        # this question, so a true fact from a *retrieved* page is not scored as
        # an invention (invention is scored by `faithfulness` and by the strict
        # on-topic precision in `key_point_precision_strict`)
        used_context = " ".join(str(h.get("text", "") if isinstance(h, dict)
                                    else getattr(h, "text", "")) for h in hits)
        scores.update(metrics.answer_correctness(answer_text, row.get("gold_answer", ""),
                                                 decline_expected=not answerable,
                                                 gold_context=gold_context,
                                                 used_context=used_context))
        if answerable:
            scores.update(metrics.citation_scores(getattr(result, "citations", []) or [], evidence_gold))
        scores.update(metrics.refusal_scores(declined, answerable))

        outcome = QuestionOutcome(
            qid=str(row.get("qid", "")), config=name, question=question,
            category=str(row.get("category", "")), content_type=str(row.get("content_type", "prose")),
            answerable=answerable, split=str(row.get("split", "dev")), answer=answer_text, declined=declined,
            decline_reason=str(getattr(result, "decline_reason", "") or ""), citations=citations,
            gold_citations=list(row.get("gold_citations") or []), n_evidence=len(hits),
            tool_calls=int(getattr(result, "tool_calls", 0) or 0), latency_s=latency,
            tokens=int(sum((getattr(result, "trace", None).token_usage or {}).values())
                       if getattr(result, "trace", None) else 0),
            scores={k: v for k, v in scores.items()},
            evidence_locs=[f"{h.doc_id} §{h.section_number} p.{h.page}" for h in hits[:12]],
            gold_evidence=evidence_gold, trace=getattr(result, "trace", None))
        outcomes.append(outcome)
        log_event(log, "eval.question", config=name, qid=outcome.qid, declined=declined,
                  recall_at_10=round(scores["recall@10"], 3) if scores["recall@10"] == scores["recall@10"] else None,
                  correctness=round(scores.get("correctness", float("nan")), 3)
                  if scores.get("correctness") == scores.get("correctness") else None,
                  duration_s=round(latency, 3))
    log_event(log, "eval.config.done", config=name, questions=len(outcomes))
    return outcomes


# --------------------------------------------------------------------------- #
# aggregation and reporting
# --------------------------------------------------------------------------- #
def aggregate(outcomes: Sequence[QuestionOutcome]) -> dict[str, Any]:
    """Mean metrics overall, and grouped by category and by content type."""
    def mean_rows(rows: Sequence[QuestionOutcome]) -> dict[str, Any]:
        agg = metrics.Aggregate()
        for row in rows:
            agg.add_many(row.scores)
        summary = agg.summary()
        latencies = [r.latency_s for r in rows]
        summary["p50_latency_s"] = round(metrics.percentile(latencies, 0.5), 4)
        summary["p95_latency_s"] = round(metrics.percentile(latencies, 0.95), 4)
        summary["mean_tokens"] = round(sum(r.tokens for r in rows) / len(rows), 1) if rows else 0.0
        return summary

    def group(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[QuestionOutcome]] = {}
        for row in outcomes:
            groups.setdefault(str(getattr(row, key)), []).append(row)
        return {name: mean_rows(rows) for name, rows in sorted(groups.items())}

    return {"overall": mean_rows(outcomes), "by_category": group("category"), "by_content_type": group("content_type"),
            "n_questions": len(outcomes)}


def structure_and_kg(config: Any, ingested_tables: Sequence[Any], ingested_figures: Sequence[Any]) -> dict[str, Any]:
    """Audit the graph and the structural accuracy of tables and figures."""
    manifest = config.path("paths.synthetic") / "manifest.json"
    structure = evaluate_structure(manifest, ingested_tables, ingested_figures)
    triples = load_triple_sample(config.path("paths.graph") / "triple_sample.jsonl")
    audit = audit_triples(triples)
    provenance = evaluate_edge_provenance(config.path("paths.graph") / "edges.jsonl")
    return {"structure": structure.to_dict(), "graph": {**audit.to_dict(), "provenance": provenance}}


def error_analysis(outcomes: Sequence[QuestionOutcome], *, limit: int = 25) -> list[dict[str, Any]]:
    """Classify every failed question so the owner can pick it up.

    Categories are assigned from the measured quantities, in the order the
    pipeline would have to fix them: retrieval miss → wrong evidence → unsupported
    answer → citation miss → over-refusal → false answer.
    """
    findings: list[dict[str, Any]] = []
    for row in outcomes:
        scores = row.scores
        problems: list[str] = []
        if row.answerable:
            if scores.get("recall@10", 1.0) == 0.0:
                problems.append("retrieval_miss")
            if not row.declined and scores.get("correctness", 1.0) < 0.5:
                problems.append("answer_incomplete")
            if not row.declined and scores.get("faithfulness", 1.0) < 0.8:
                problems.append("unsupported_claim")
            if not row.declined and scores.get("citation_accuracy", 1.0) < 0.5:
                problems.append("citation_miss")
            if row.declined:
                problems.append("over_refusal")
        else:
            if not row.declined:
                problems.append("false_answer")
            elif row.decline_reason.startswith("no evidence") or "does not occur anywhere" in row.decline_reason:
                problems.append("declined_weak_reason")
        if problems:
            findings.append({"qid": row.qid, "config": row.config, "category": row.category,
                             "content_type": row.content_type, "problems": problems,
                             "question": row.question, "declined": row.declined,
                             "recall@10": scores.get("recall@10"), "correctness": scores.get("correctness"),
                             "faithfulness": scores.get("faithfulness"),
                             "citation_accuracy": scores.get("citation_accuracy"),
                             "evidence": row.evidence_locs[:5], "gold_evidence": row.gold_evidence[:3],
                             "answer": row.answer[:400]})
    findings.sort(key=lambda f: (-len(f["problems"]), f["qid"]))
    return findings[:limit]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "n/a" if value != value else f"{value:.3f}"
    return str(value)


def write_report(results_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the Markdown report quoted in the README."""
    lines: list[str] = []
    lines.append("# Evaluation report — spec-intelligence")
    lines.append("")
    lines.append(f"Generated {payload['generated_at']} by `python scripts/run_eval.py "
                 f"--configs {','.join(payload['configs'])} --split {payload['split']}`.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    hardware = payload.get("hardware", {})
    lines.append("| item | value |")
    lines.append("|---|---|")
    for key in sorted(hardware):
        lines.append(f"| {key} | {hardware[key]} |")
    lines.append(f"| llm backend | {payload.get('llm_backend')} |")
    lines.append(f"| questions | {payload.get('n_questions')} |")
    lines.append(f"| split | {payload.get('split')} |")
    lines.append("")
    lines.append("## Target metrics (config F, the full system)")
    lines.append("")
    lines.append("| metric | target | measured (dev split) | measured (held-out) | status |")
    lines.append("|---|---|---|---|---|")
    for key, target in TARGETS.items():
        dev = payload.get("targets_measured", {}).get(key)
        held = payload.get("targets_measured_held_out", {}).get(key)
        status = "—"
        if isinstance(dev, (int, float)) and dev == dev:
            status = "met" if dev >= target else "below"
        lines.append(f"| {key} | ≥ {target:.2f} | {_fmt(dev)} | {_fmt(held)} | {status} |")
    lines.append("")
    lines.append("## Ablations (config A → F)")
    lines.append("")
    lines.append("| config | pipeline | recall@10 | recall@5 | correctness | faithfulness | citation acc. | "
                 "refusal acc. | p50 s | p95 s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in payload["configs"]:
        summary = payload["aggregates"].get(name, {}).get("overall", {})
        lines.append(f"| {name} | {CONFIG_DESCRIPTIONS.get(name, '')} | {_fmt(summary.get('recall@10'))} | "
                     f"{_fmt(summary.get('recall@5'))} | {_fmt(summary.get('correctness'))} | "
                     f"{_fmt(summary.get('faithfulness'))} | {_fmt(summary.get('citation_accuracy'))} | "
                     f"{_fmt(summary.get('refusal_correct'))} | {_fmt(summary.get('p50_latency_s'))} | "
                     f"{_fmt(summary.get('p95_latency_s'))} |")
    lines.append("")
    lines.append("## Content type × approach (recall@10 / correctness)")
    lines.append("")
    content_types = sorted({ct for name in payload["configs"]
                            for ct in payload["aggregates"].get(name, {}).get("by_content_type", {})})
    header = "| content type | " + " | ".join(payload["configs"]) + " |"
    lines.append(header)
    lines.append("|---" * (len(payload["configs"]) + 1) + "|")
    for content_type in content_types:
        cells = []
        for name in payload["configs"]:
            block = payload["aggregates"].get(name, {}).get("by_content_type", {}).get(content_type, {})
            cells.append(f"{_fmt(block.get('recall@10'))} / {_fmt(block.get('correctness'))}")
        lines.append(f"| {content_type} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Category breakdown (config F)")
    lines.append("")
    lines.append("| category | n | recall@10 | correctness | faithfulness | citation acc. | refusal acc. |")
    lines.append("|---|---|---|---|---|---|---|")
    for category, block in sorted(payload["aggregates"].get(payload["configs"][-1], {})
                                  .get("by_category", {}).items()):
        n = payload["counts"].get(category, "")
        lines.append(f"| {category} | {n} | {_fmt(block.get('recall@10'))} | {_fmt(block.get('correctness'))} | "
                     f"{_fmt(block.get('faithfulness'))} | {_fmt(block.get('citation_accuracy'))} | "
                     f"{_fmt(block.get('refusal_correct'))} |")
    lines.append("")
    lines.append("## Structural accuracy and knowledge-graph audit")
    lines.append("")
    structure = payload.get("structure_kg", {}).get("structure", {})
    graph = payload.get("structure_kg", {}).get("graph", {})
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| table cell accuracy | {_fmt(structure.get('table_cell_accuracy'))} |")
    lines.append(f"| figure structure accuracy | {_fmt(structure.get('figure_structure_accuracy'))} |")
    lines.append(f"| KG rule precision (sample of {graph.get('sample_size')}) | {_fmt(graph.get('rule_precision'))} |")
    lines.append(f"| KG annotated precision | {_fmt(graph.get('annotated_precision'))} "
                 f"({graph.get('annotated_total')} annotated) |")
    lines.append(f"| edges with document provenance | {_fmt((graph.get('provenance') or {}).get('doc_provenance'))} |")
    lines.append(f"| edges with page provenance | {_fmt((graph.get('provenance') or {}).get('page_provenance'))} |")
    lines.append("")
    lines.append("Per-document structural accuracy:")
    lines.append("")
    lines.append("| document | tables | table cell acc. | figures | figure struct. acc. |")
    lines.append("|---|---|---|---|---|")
    for doc_id, block in (structure.get("per_document") or {}).items():
        lines.append(f"| {doc_id} | {block.get('n_tables')} | {_fmt(block.get('tables'))} | "
                     f"{block.get('n_figures')} | {_fmt(block.get('figures'))} |")
    lines.append("")
    lines.append("## Error analysis")
    lines.append("")
    lines.append("Failures from the last config, classified by the pipeline stage that must fix them. "
                 "The full list is in `eval/results/error_analysis.json`.")
    lines.append("")
    lines.append("| qid | problems | recall@10 | correctness | note |")
    lines.append("|---|---|---|---|---|")
    for finding in payload.get("error_analysis", [])[:20]:
        lines.append(f"| {finding['qid']} | {', '.join(finding['problems'])} | {_fmt(finding.get('recall@10'))} | "
                     f"{_fmt(finding.get('correctness'))} | {finding['answer'][:80].replace('|', '/')} |")
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Correctness is the mean of key-point recall (the numbers, identifiers and named facts "
                 "the gold answer contains) and content-token F1; wording is not scored, facts are.")
    lines.append("- Faithfulness is recomputed from the retrieved evidence by `specintel.eval.metrics`, "
                 "independently of the agent's own citation verifier.")
    lines.append("- Citation accuracy is the share of the answer's citations whose document and page occur in "
                 "the gold evidence for that question.")
    lines.append("- Refusal accuracy counts an honest decline on an unanswerable question and an answer on an "
                 "answerable one as correct; over-refusal and false answers are the two failure modes.")
    lines.append("- Recall is page-level (`doc_id`, `page`) with element-id matching as a fallback, because the "
                 "gold evidence is recorded as a page plus an element id.")
    lines.append("")
    path = results_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the spec-intelligence evaluation")
    parser.add_argument("--configs", default="A,B,C,D,E,F", help="comma separated config names")
    parser.add_argument("--split", default="all", choices=["all", "dev", "held_out"])
    parser.add_argument("--limit", type=int, default=None, help="limit the number of questions")
    parser.add_argument("--gold", default="eval/gold.jsonl")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--llm", default=None, help="llm backend: extractive | ollama | openai | auto")
    args = parser.parse_args(argv)

    from specintel.common.config import load_config

    config = load_config()
    setup_from_config(config)
    log = get_logger("ws5.eval", config)
    results_dir = config.root / args.out
    results_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = [json.loads(line) for line in (config.root / args.gold).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if args.split != "all":
        gold_rows = [row for row in gold_rows if str(row.get("split", "dev")) == args.split]
    counts: dict[str, int] = {}
    for row in gold_rows:
        counts[str(row.get("category", ""))] = counts.get(str(row.get("category", "")), 0) + 1

    toolkit = Toolkit.from_config(config)
    llm = get_llm(config, args.llm) if args.llm else ExtractiveLLM()
    floors_env = os.environ.get("SPECINTEL_SCORE_FLOORS", "")
    if floors_env:
        values = [float(v) for v in floors_env.split(",") if v.strip()]
        floors = dict(zip(("dense", "hybrid", "vision", "fused"), values))
        for mode, value in floors.items():
            config._score_floor_override = None
            config.__dict__.setdefault("_floor_map", {})[mode] = value
        # a runner reads its floor from the agent config; keep it simple and set
        # a single shared value, which is exactly what this ablation tests
        config._score_floor_override = values[0]
        log_event(get_logger("ws5.eval", config), "eval.score_floor_override",
                  floors=",".join(str(v) for v in values))
    # tuning hooks used by scripts/tune_composer.py; unset in normal runs
    import os as _os
    if _os.environ.get("SPECINTEL_TUNE_MAX_SENTENCES"):
        llm.max_sentences = int(_os.environ["SPECINTEL_TUNE_MAX_SENTENCES"])
    if _os.environ.get("SPECINTEL_TUNE_PER_ITEM"):
        llm.per_item_limit = int(_os.environ["SPECINTEL_TUNE_PER_ITEM"])
    if _os.environ.get("SPECINTEL_TUNE_FOCUS"):
        llm.focus_ratio = float(_os.environ["SPECINTEL_TUNE_FOCUS"])
    if _os.environ.get("SPECINTEL_TUNE_GATE") == "off":
        import specintel.common.llm as _llm
        _llm._is_relevant = lambda *a, **k: True
    llm_backend = getattr(llm, "name", "extractive")

    log_event(log, "eval.start", configs=args.configs, split=args.split, questions=len(gold_rows),
              llm=llm_backend, rss_mb=rss_mb(), hardware=hardware_profile())

    aggregates: dict[str, Any] = {}
    all_outcomes: dict[str, list[QuestionOutcome]] = {}
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        outcomes = run_config(config, name, gold_rows, toolkit=toolkit, llm=llm, top_k=args.top_k,
                              limit=args.limit)
        all_outcomes[name] = outcomes
        aggregates[name] = aggregate(outcomes)
        (results_dir / f"answers_{name}.jsonl").write_text(
            "\n".join(json.dumps(o.to_dict(), ensure_ascii=False) for o in outcomes) + "\n", encoding="utf-8")
        if name in ("E", "F"):
            (results_dir / f"traces_{name}.jsonl").write_text(
                "\n".join(json.dumps({"qid": o.qid, "split": o.split, "trace": o.trace.to_dict()},
                                      ensure_ascii=False)
                          for o in outcomes if getattr(o, "trace", None)) + "\n", encoding="utf-8")

    structure_kg = structure_and_kg(config, toolkit.tables, toolkit.figures)
    findings = error_analysis(all_outcomes[list(all_outcomes)[-1]])

    last_config = list(all_outcomes)[-1]
    measured = {
        "recall@10": aggregates[last_config]["overall"].get("recall@10"),
        "correctness": aggregates[last_config]["overall"].get("correctness"),
        "faithfulness": aggregates[last_config]["overall"].get("faithfulness"),
        "citation_accuracy": aggregates[last_config]["overall"].get("citation_accuracy"),
        "refusal_accuracy": aggregates[last_config]["overall"].get("refusal_correct"),
        "kg_triple_precision": (structure_kg["graph"].get("annotated_precision")
                                or structure_kg["graph"].get("rule_precision")),
        "table_cell_accuracy": structure_kg["structure"].get("table_cell_accuracy"),
        "figure_structure_accuracy": structure_kg["structure"].get("figure_structure_accuracy"),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configs": list(all_outcomes), "split": args.split, "n_questions": len(gold_rows),
        "counts": counts, "hardware": hardware_profile(), "llm_backend": llm_backend,
        "aggregates": aggregates, "structure_kg": structure_kg, "error_analysis": findings,
        "targets": TARGETS, "targets_measured": measured,
        "config_descriptions": CONFIG_DESCRIPTIONS,
        "gold": str(config.root / args.gold),
    }
    (results_dir / "summary.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    (results_dir / "matrix.json").write_text(json.dumps(
        {name: aggregates[name]["by_content_type"] for name in aggregates}, indent=1, default=str),
        encoding="utf-8")
    (results_dir / "error_analysis.json").write_text(json.dumps(findings, indent=1, default=str),
                                                     encoding="utf-8")
    report = write_report(results_dir, payload)
    log_event(log, "eval.done", configs=len(all_outcomes), questions=len(gold_rows), report=str(report))

    print(f"\nEvaluation complete — {len(gold_rows)} questions, configs {', '.join(all_outcomes)}")
    print(f"  report  -> {report}")
    print(f"  summary -> {results_dir / 'summary.json'}")
    print("\n  target vs measured (config F)")
    for key, target in TARGETS.items():
        value = measured.get(key)
        status = "n/a"
        if isinstance(value, (int, float)) and value == value:
            status = "met" if value >= target else "BELOW"
        print(f"    {key:28s} target {target:.2f}  measured {_fmt(value):>7s}  {status}")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
