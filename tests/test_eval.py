"""WS5 — the evaluation harness: metrics, structure scoring, triples, reporting.

The metric functions are tested directly, and one configuration is run
end-to-end over the fixture corpus with a two-question gold set, so the harness
itself is covered even though the real gold set and the seven-document corpus
live behind ``data/`` (which is never committed).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from specintel.common.models import Citation, SearchHit
from specintel.eval import metrics
from specintel.eval.kg_precision import audit_triples, parse_triple
from specintel.eval.structure import (
    StructureReport,
    evaluate_structure,
    normalise_cell,
    score_figures,
    score_tables,
)


def hit(doc_id="pd", page=3, element_ids=("pd:t6.8:v3",), section="6.6", text="tSenderResponse 24 ms"):
    return SearchHit(chunk_id=f"{doc_id}:c:{page}:{section}", score=1.0, doc_id=doc_id, page=page,
                     section_number=section, text=text, element_ids=list(element_ids))


GOLD = {"doc_id": "pd", "page": 3, "element_id": "pd:t6.8:v3", "quote": "tSenderResponse"}


# --------------------------------------------------------------------------- #
# retrieval metrics
# --------------------------------------------------------------------------- #
def test_recall_rewards_a_hit_on_the_gold_page():
    assert metrics.recall_at_k([hit()], [GOLD], 10) == 1.0


def test_recall_is_zero_when_the_page_was_never_retrieved():
    assert metrics.recall_at_k([hit(page=9, element_ids=("other",))], [GOLD], 10) == 0.0


def test_an_element_id_match_beats_a_page_match_even_on_another_page():
    assert metrics.hit_matches_evidence(hit(page=9, element_ids=("pd:t6.8:v3",)), GOLD)
    assert not metrics.hit_matches_evidence(hit(page=9, element_ids=("other",)), GOLD)


def test_recall_at_k_respects_k():
    hits = [hit(page=9, element_ids=("other",))] * 5 + [hit()]
    assert metrics.recall_at_k(hits, [GOLD], 5) == 0.0
    assert metrics.recall_at_k(hits, [GOLD], 10) == 1.0


def test_mrr_uses_the_rank_of_the_first_correct_hit():
    hits = [hit(page=9, element_ids=("a",)), hit(page=8, element_ids=("b",)), hit()]
    assert metrics.reciprocal_rank(hits, [GOLD]) == pytest.approx(1 / 3)


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_when_the_hit_is_buried():
    perfect = metrics.ndcg_at_k([hit()], [GOLD], 5)
    buried = metrics.ndcg_at_k([hit(page=9, element_ids=("a",)), hit(page=8, element_ids=("b",)), hit()],
                               [GOLD], 5)
    assert perfect == pytest.approx(1.0)
    assert 0.0 < buried < 1.0


def test_metrics_survive_an_empty_hit_list():
    assert metrics.recall_at_k([], [GOLD], 10) == 0.0
    assert metrics.reciprocal_rank([], [GOLD]) == 0.0
    assert math.isnan(metrics.ndcg_at_k([], [GOLD], 5)) or metrics.ndcg_at_k([], [GOLD], 5) == 0.0


# --------------------------------------------------------------------------- #
# answer metrics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prediction", [
    "tSenderResponse is at least 24 ms and at most 30 ms.",
    "The timer tSenderResponse has a minimum of 24 ms and a maximum of 30 ms.",
])
def test_a_correct_answer_scores_high(prediction):
    gold = "tSenderResponse is 24 ms minimum, 30 ms maximum."
    assert metrics.answer_correctness(prediction, gold)["correctness"] >= 0.75


def test_an_answer_with_the_wrong_numbers_scores_low():
    gold = "tSenderResponse is 24 ms minimum, 30 ms maximum."
    wrong = "tSenderResponse is 250 ms minimum and 300 ms maximum."
    assert metrics.answer_correctness(wrong, gold)["correctness"] < 0.6


def test_an_unanswerable_question_rewards_declining():
    """``refusal_correct`` is the refusal-accuracy figure the report quotes."""
    assert metrics.refusal_scores(declined=True, answerable=False)["refusal_correct"] == 1.0
    assert metrics.refusal_scores(declined=False, answerable=False)["refusal_correct"] == 0.0
    assert metrics.refusal_scores(declined=False, answerable=True)["refusal_correct"] == 1.0
    assert metrics.refusal_scores(declined=True, answerable=True)["refusal_correct"] == 0.0
    modes = metrics.refusal_scores(declined=False, answerable=False)
    assert modes["false_answer"] == 1.0 and modes["correct_decline"] == 0.0


def test_faithfulness_falls_when_the_answer_invents_content():
    evidence = [hit(text="tSenderResponse: minimum 24 ms, maximum 30 ms")]
    faithful = metrics.faithfulness("tSenderResponse is 24 ms to 30 ms.", evidence)
    invented = metrics.faithfulness(
        "tSenderResponse is 24 ms to 30 ms and the host must also enable the purple "
        "indicator lamp with a 45 ohm resistor and a secret handshake.", evidence)
    assert faithful > invented
    assert invented < 0.34


def test_citation_scores_check_document_page_and_element():
    citations = [Citation(doc_id="pd", revision="R3.2 V1.2", section_number="6.6", page=3,
                          region="Table 6.8", element_ids=["pd:t6.8:v3"])]
    scores = metrics.citation_scores(citations, [GOLD])
    assert scores["citation_accuracy"] == 1.0
    bad = [Citation(doc_id="usb4", revision="2.0", section_number="6", page=99, region="Table 6.8")]
    assert metrics.citation_scores(bad, [GOLD])["citation_accuracy"] == 0.0


def test_citation_accuracy_of_a_declined_answer_is_reported_as_zero():
    scores = metrics.citation_scores([], [GOLD])
    assert scores["citation_accuracy"] == 0.0 and scores["citation_recall"] == 0.0

def test_citation_accuracy_is_nan_when_there_is_no_gold_locator():
    assert math.isnan(metrics.citation_scores([], [])["citation_accuracy"])


# --------------------------------------------------------------------------- #
# structure scoring
# --------------------------------------------------------------------------- #
def test_normalise_cell_folds_whitespace_and_case():
    assert normalise_cell("  24  ms ") == normalise_cell("24 ms")
    assert normalise_cell("Min") == normalise_cell("min")


def table_element(doc_id="pd", number="6.8", page=3, caption="Timer definitions and limits",
                  headers=(("Timer", "Min", "Max", "Unit"),), rows=(("tSenderResponse", "24", "30", "ms"),)):
    """A manifest entry, in the shape ``synthetic/corpus.py`` writes it."""
    return {"type": "table", "doc_id": doc_id, "number": number, "page": page, "caption": caption,
            "headers": [list(h) for h in headers], "rows": [list(r) for r in rows]}


def figure_element(doc_id="pd", number="8-1", page=6, caption="Source policy engine states",
                   nodes=("PE_SRC_Startup", "PE_SRC_Send_Capabilities"),
                   edges=((0, 1, "VBUS valid"),), raster=False,
                   figure_kind="state_machine"):
    return {"type": "figure", "doc_id": doc_id, "number": number, "page": page, "caption": caption,
            "figure_kind": figure_kind, "nodes": list(nodes), "edges": [list(e) for e in edges],
            "raster": raster, "description": ""}


class FakeTable:
    def __init__(self, **kw):
        self.__dict__.update({"doc_id": "pd", "number": "6.8", "page": 3, "pages": [3],
                              "caption": "Timer definitions and limits", "stitched": False,
                              "headers": [["Timer", "Min", "Max", "Unit"]],
                              "rows": [["tSenderResponse", "24", "30", "ms"]],
                              "markdown": "| Timer | Min | Max | Unit |\n| tSenderResponse | 24 | 30 | ms |"})
        self.__dict__.update(kw)


class FakeFigure:
    def __init__(self, **kw):
        self.__dict__.update({"doc_id": "pd", "number": "8-1", "page": 6,
                              "caption": "Source policy engine states",
                              "kind": "state_machine", "raster": False,
                              "description": "PE_SRC_Startup to PE_SRC_Send_Capabilities.",
                              "structure": {"states": ["PE_SRC_Startup", "PE_SRC_Send_Capabilities"],
                                            "transitions": [{"from": "PE_SRC_Startup",
                                                             "to": "PE_SRC_Send_Capabilities",
                                                             "trigger": "VBUS valid"}]},
                              "structure_text": "", "ocr_text": "", "text": ""})
        self.__dict__.update(kw)


def test_table_cell_accuracy_scores_a_stitched_table():
    scores = score_tables([table_element()], [FakeTable()])
    assert len(scores) == 1 and scores[0].matched
    assert scores[0].accuracy == 1.0
    assert scores[0].to_dict()["body_cells"] == 4


def test_a_lost_cell_lowers_the_accuracy():
    ingested = FakeTable(rows=[["tSenderResponse", "24", "30"]], markdown="| tSenderResponse |")
    score = score_tables([table_element()], [ingested])[0]
    assert 0.0 < score.accuracy < 1.0


def test_a_table_the_ingester_never_found_is_reported_and_scores_zero():
    """A miss is scored as a miss, and the note says so."""
    score = score_tables([table_element(number="9.9")], [FakeTable()])[0]
    assert score.matched is False
    assert score.note == "table not detected by the ingester"
    assert score.accuracy == 0.0
    assert score.body_hits == 0


def test_multi_page_table_stitching_is_visible_in_the_score():
    plain = FakeTable(pages=[3], stitched=False)
    score = score_tables([table_element(page=3, headers=(("Timer",),), rows=(("tSenderResponse",),))],
                         [plain])[0]
    assert score.pages == [3] and score.stitched is False


def test_figure_labels_and_transitions_score():
    scores = score_figures([figure_element()], [FakeFigure()])
    assert len(scores) == 1
    assert scores[0].node_hits == 2 and scores[0].transitions == 1
    assert scores[0].accuracy == 1.0
    assert "trigger labels recovered: 1/1" in scores[0].note


def test_a_raster_figure_without_ocr_keeps_the_note_and_scores_zero_nodes():
    """The raster figure is scored, and the note says why it is incomplete."""
    ingested = FakeFigure(raster=True, description="", structure={}, ocr_text="")
    score = score_figures([figure_element(raster=True, figure_kind="timing")], [ingested])[0]
    assert score.raster is True
    assert score.node_hits == 0
    assert "OCR/VLM" in score.note


def test_structure_report_averages_and_counts_detection():
    report = StructureReport(manifest_path="manifest.json")
    report.tables = score_tables([table_element()], [FakeTable()])
    report.figures = score_figures([figure_element()], [FakeFigure()])
    payload = report.to_dict()
    assert payload["tables_detected"] == 1.0
    assert payload["table_cell_accuracy"] == 1.0
    assert payload["figure_structure_accuracy"] == 1.0
    assert payload["per_document"]["pd"]["n_tables"] == 1


def test_evaluate_structure_reads_a_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"elements": [table_element(), figure_element()]}), encoding="utf-8")
    report = evaluate_structure(manifest, [FakeTable()], [FakeFigure()])
    assert report.to_dict()["table_cell_accuracy"] == 1.0
    assert report.to_dict()["figure_structure_accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# knowledge-graph triples
# --------------------------------------------------------------------------- #
def test_parse_triple_accepts_the_export_shape():
    src, edge, dst = parse_triple({"src_id": "State:pe_src", "edge_type": "STARTS_TIMER",
                                   "dst_id": "Timer:tsenderresponse"})
    assert (src, edge, dst) == ("State:pe_src", "STARTS_TIMER", "Timer:tsenderresponse")


def provenance(doc_id="pd", page=4, section="6.6", method="rule"):
    return {"provenance": {"doc_id": doc_id, "page": page, "section": section, "method": method,
                           "confidence": 0.9}}


def test_audit_triples_measures_rule_precision():
    triples = [
        {"src_id": "State:pe_src", "dst_id": "Timer:t1", "edge_type": "STARTS_TIMER", **provenance()},
        {"src_id": "Timer:t1", "dst_id": "Table:tbl_1", "edge_type": "DEFINED_IN", **provenance()},
        # an edge type outside the ontology, and no provenance at all
        {"src_id": "State:pe_src", "dst_id": "Timer:t1", "edge_type": "MADE_UP_EDGE",
         **provenance()},
        {"src_id": "State:pe_src", "dst_id": "Timer:t9", "edge_type": "STARTS_TIMER",
         "provenance": {"doc_id": "pd", "page": 9}},
    ]
    audit = audit_triples(triples)
    assert audit.total == 4
    assert audit.rule_pass == 2, audit.failures
    assert audit.rule_precision == pytest.approx(0.5)
    assert any("not in ontology" in reason for reason in audit.failures)
    assert audit.failures["no section or element provenance"] == 1


def test_audit_counts_the_human_verdicts_when_they_are_present():
    triples = [
        {"src_id": "State:pe_src", "dst_id": "Timer:t1", "edge_type": "STARTS_TIMER",
         "verdict": "valid", **provenance()},
        {"src_id": "State:pe_src", "dst_id": "Timer:t2", "edge_type": "STARTS_TIMER",
         "verdict": "invalid", **provenance()},
    ]
    audit = audit_triples(triples)
    assert audit.annotated_total == 2 and audit.annotated_valid == 1
    assert audit.annotated_precision == pytest.approx(0.5)


def test_audit_triples_of_an_empty_graph_is_not_a_perfect_score():
    audit = audit_triples([])
    payload = audit.to_dict()
    assert payload["sample_size"] == 0
    assert payload["rule_precision"] is None, "an empty sample proves nothing"


# --------------------------------------------------------------------------- #
# the harness end to end (fixture corpus, two questions)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def eval_setup(ingested_tiny, tmp_path_factory):
    from specintel.index.build import build_indexes
    from specintel.kg.build import build_graph

    build_indexes(ingested_tiny, doc_ids=["pd", "typec"])
    build_graph(ingested_tiny, doc_ids=["pd", "typec"], use_llm=False)
    from specintel.agent.tools import Toolkit

    toolkit = Toolkit.from_config(ingested_tiny)
    toolkit.doc_ids = ["pd", "typec"]
    return ingested_tiny, toolkit


GOLD_QUESTIONS = [
    {"qid": "T1", "question": "What are the limits of tSenderResponse?",
     "category": "timer", "content_type": "bitfield_table", "author": "agent", "split": "dev",
     "answerable": True, "gold_answer": "tSenderResponse is 24 ms minimum and 30 ms maximum.",
     "gold_evidence": [{"doc_id": "pd", "page": 2, "quote": "tSenderResponse"}],
     "gold_citations": ["pd"], "verifier": "agent"},
    {"qid": "U1", "question": "What is the maximum number of DisplayPort tunnels on a Meteor Lake host?",
     "category": "unknown", "content_type": "prose", "author": "agent", "split": "held_out",
     "answerable": False, "gold_answer": "The corpus does not answer this question.",
     "gold_evidence": [], "gold_citations": [], "verifier": "agent"},
]


def test_run_config_produces_one_outcome_per_question(eval_setup):
    from specintel.common.llm import ExtractiveLLM
    from specintel.eval.run_eval import run_config

    config, toolkit = eval_setup
    outcomes = run_config(config, "F", GOLD_QUESTIONS, toolkit=toolkit, llm=ExtractiveLLM(), top_k=8)
    assert [o.qid for o in outcomes] == ["T1", "U1"]
    answered = outcomes[0]
    assert answered.scores["recall@10"] == 1.0
    assert not answered.declined
    assert answered.citations, "an answered question carries citations"
    assert answered.scores["refusal_correct"] == 1.0
    declined = outcomes[1]
    assert declined.declined and declined.scores["refusal_correct"] == 1.0


def test_aggregate_groups_by_category_and_content_type(eval_setup):
    from specintel.common.llm import ExtractiveLLM
    from specintel.eval.run_eval import aggregate, run_config

    config, toolkit = eval_setup
    outcomes = run_config(config, "B", GOLD_QUESTIONS, toolkit=toolkit, llm=ExtractiveLLM(), top_k=8)
    summary = aggregate(outcomes)
    assert {"overall", "by_category", "by_content_type"} <= set(summary)
    assert summary["overall"]["recall@10"] >= 0.5
    assert summary["overall"]["p50_latency_s"] >= 0.0
    assert set(summary["by_content_type"]) == {"bitfield_table", "prose"}


def test_aggregate_reports_a_superset_of_the_target_metrics(eval_setup):
    from specintel.common.llm import ExtractiveLLM
    from specintel.eval.run_eval import aggregate, run_config

    config, toolkit = eval_setup
    outcomes = run_config(config, "F", GOLD_QUESTIONS, toolkit=toolkit, llm=ExtractiveLLM(), top_k=8)
    overall = aggregate(outcomes)["overall"]
    for key in ("recall@10", "correctness", "faithfulness", "citation_accuracy", "refusal_correct"):
        assert key in overall, key


def test_error_analysis_classifies_what_goes_wrong(eval_setup):
    """Every failure must arrive with the stage that has to fix it."""
    from specintel.common.llm import ExtractiveLLM
    from specintel.eval.run_eval import QuestionOutcome, error_analysis

    weak = QuestionOutcome(qid="T9", config="A", question="What are the limits of tSenderResponse?",
                           category="T", content_type="bitfield_table", answerable=True, answer="", declined=True,
                           decline_reason="no evidence above threshold",
                           scores={"recall@10": 0.0, "correctness": 0.0, "faithfulness": float("nan"),
                                   "citation_accuracy": 0.0})
    answered_wrong = QuestionOutcome(qid="T8", config="A", question="What is tPSTransition?",
                                     category="T", content_type="prose", answerable=True,
                                     answer="tPSTransition is 5 ms.", declined=False,
                                     scores={"recall@10": 1.0, "correctness": 0.2, "faithfulness": 0.4,
                                             "citation_accuracy": 0.1})
    missed_decline = QuestionOutcome(qid="U1", config="A", question="Which Wi-Fi amendment?",
                                     category="U", content_type="prose", answerable=False,
                                     answer="802.11be.", declined=False,
                                     scores={"recall@10": 0.0, "correctness": 0.0,
                                             "faithfulness": 0.0, "citation_accuracy": 0.0})
    rows = {row["qid"]: row for row in error_analysis([weak, answered_wrong, missed_decline])}
    assert rows["T9"]["problems"] == ["retrieval_miss", "over_refusal"]
    assert {"answer_incomplete", "unsupported_claim", "citation_miss"} <= set(rows["T8"]["problems"])
    assert rows["U1"]["problems"] == ["false_answer"]
    assert {"qid", "problems", "question", "answer"} <= set(rows["T9"])


def test_gold_set_rows_validate_against_the_contract():
    from specintel.common.models import GoldQuestion

    for row in GOLD_QUESTIONS:
        question = GoldQuestion(qid=row["qid"], question=row["question"], category=row["category"],
                                gold_answer=row["gold_answer"], gold_evidence=row["gold_evidence"],
                                author=row["author"], verifier=row["verifier"], split=row["split"],
                                answerable=row["answerable"])
        assert question.qid and question.question


def test_committed_gold_set_is_well_formed_if_present():
    """``eval/gold.jsonl`` is committed (it holds no licensed text), so check it."""
    path = Path(__file__).resolve().parents[1] / "eval" / "gold.jsonl"
    if not path.exists():
        pytest.skip("gold set not present in this checkout")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 60, "the brief asks for at least 60 questions"
    authors: dict[str, int] = {}
    for row in rows:
        for key in ("qid", "question", "category", "gold_answer", "author"):
            assert str(row.get(key, "")).strip(), (row.get("qid"), key)
        assert row.get("gold_evidence") or row.get("answerable") is False
        authors[row["author"]] = authors.get(row["author"], 0) + 1
    assert min(authors.values()) >= 12, "every member contributes at least twelve questions"
    assert sum(1 for r in rows if r.get("split") == "held_out") >= 15
    assert all(r.get("verifier") for r in rows), "every gold row names a verifier"
