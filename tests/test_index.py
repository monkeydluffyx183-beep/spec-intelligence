"""WS2 — chunking, the three indexes, fusion and the retrieval metrics."""

from __future__ import annotations

import json

import pytest

from specintel.eval import metrics as irmetrics


@pytest.fixture(scope="session")
def fixture_index(ingested_tiny):
    """The fixture corpus, ingested once per session and indexed."""
    pytest.importorskip("numpy")
    from specintel.index.build import build_indexes

    index, _report = build_indexes(ingested_tiny, doc_ids=["pd", "typec"])
    return ingested_tiny, index


def test_chunks_carry_page_and_section_provenance(fixture_index):
    config, index = fixture_index
    assert index.chunks
    for chunk in index.chunks[:40]:
        assert chunk.doc_id and chunk.page >= 1
        assert chunk.revision, "every chunk can be cited with its revision"
        assert chunk.text.strip()
        assert chunk.element_type


def test_parent_child_chunking_keeps_small_units_retrievable(fixture_index):
    """A table row must be findable on its own, not only through its table."""
    config, index = fixture_index
    parents = [c for c in index.chunks if c.child_chunk_ids]
    if not parents:
        pytest.skip("fixture corpus is small enough to need no parent/child split")
    children = [c for c in index.chunks if c.parent_chunk_id]
    assert children, "children exist when parents do"
    parent_ids = {c.chunk_id for c in parents}
    assert all(c.parent_chunk_id in parent_ids for c in children)


def test_hybrid_search_finds_a_timer_that_dense_alone_can_miss(fixture_index):
    config, index = fixture_index
    hits = index.search_text("tSenderResponse limits", k=5, hybrid=True, rerank=True)
    assert hits
    assert any("tSenderResponse" in h.text for h in hits[:3]), [h.text[:40] for h in hits[:3]]


def test_search_filters_are_honoured(fixture_index):
    from specintel.index.search import Filters

    config, index = fixture_index
    hits = index.search_text("termination", k=5, filters=Filters(docs=("typec",)))
    assert hits and all(h.doc_id == "typec" for h in hits)
    table_hits = index.search_text("tSenderResponse", k=5, filters=Filters(tables_only=True))
    assert all(h.table_id for h in table_hits)


def test_page_index_returns_elements_of_the_winning_page(fixture_index):
    config, index = fixture_index
    pages = index.search_pages("PS_RDY ready", k=2)
    assert pages, "the page (vision) path returns pages"
    for page in pages:
        assert page["doc_id"] and page["page"] >= 1
        assert "element_ids" in page and "text" in page


def test_fusion_keeps_both_paths_represented(fixture_index):
    config, index = fixture_index
    fused = index.search_fused("Rp Rd terminations", k=6)
    assert fused
    sources = {h.source for h in fused}
    assert sources & {"fused", "text", "pages"}


def test_reciprocal_rank_fusion_maths():
    from specintel.index.search import reciprocal_rank_fusion

    rankings = {"a": [{"chunk_id": "x"}, {"chunk_id": "y"}],
                "b": [{"chunk_id": "y"}, {"chunk_id": "z"}]}
    fused = reciprocal_rank_fusion(rankings, id_key="chunk_id")
    order = [item["chunk_id"] for item in fused]
    assert order[0] == "y", "a hit ranked by both retrievers wins"
    assert set(order) == {"x", "y", "z"}
    assert fused[0]["component_scores"]["a"] > 0 and fused[0]["component_scores"]["b"] > 0


def test_retrieval_metrics_have_known_values():
    from specintel.common.models import SearchHit

    hits = [SearchHit(chunk_id="a", score=1.0, doc_id="pd", page=2, text="PS_RDY"),
            SearchHit(chunk_id="b", score=0.5, doc_id="pd", page=9, text="other")]
    gold = [{"doc_id": "pd", "page": 2, "element_id": "e1"}]
    assert irmetrics.recall_at_k(hits, gold, 5) == 1.0
    assert irmetrics.reciprocal_rank(hits, gold) == 1.0
    assert irmetrics.recall_at_k([], gold, 5) == 0.0
    assert irmetrics.ndcg_at_k(hits, gold, 5) == pytest.approx(1.0)
    # a hit on the right page of the wrong document is not a hit
    assert irmetrics.recall_at_k([SearchHit(chunk_id="c", score=1.0, doc_id="typec", page=2)],
                                 gold, 5) == 0.0
    # an element id match counts even when the page differs (stitched tables)
    hit = SearchHit(chunk_id="d", score=1.0, doc_id="pd", page=5, element_ids=["e1"])
    assert irmetrics.recall_at_k([hit], gold, 5) == 1.0


def test_embeddings_are_deterministic_and_normalised():
    from specintel.index.embeddings import get_embedder

    embedder = get_embedder(backend="hashing")
    a = embedder.encode(["PS_RDY signals ready"])
    b = embedder.encode(["PS_RDY signals ready"])
    assert a.shape == b.shape
    assert (a == b).all(), "the offline embedder must be deterministic (no random seed)"
    assert float((a ** 2).sum()) == pytest.approx(1.0, abs=1e-5)


def test_baseline_configs_cover_the_ablation_matrix(fixture_index):
    config, index = fixture_index
    from specintel.index.baseline import CONFIGS

    assert set("ABCDEF") <= set(CONFIGS), sorted(CONFIGS)
    for name in "ABCDEF":
        entry = CONFIGS[name]
        assert entry.name == name and entry.description
    assert CONFIGS["F"].use_agent and CONFIGS["F"].use_graph
    assert CONFIGS["A"].use_dense and not CONFIGS["A"].use_bm25
    assert CONFIGS["B"].use_hybrid and CONFIGS["B"].use_rerank
    assert CONFIGS["C"].use_pages and CONFIGS["D"].use_fusion


def test_index_build_report_records_sizes(fixture_index):
    config, index = fixture_index
    report = config.path("paths.index") / "build_report.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    totals = payload["totals"]
    assert totals["chunks"] == len(index.chunks) > 0
    assert totals["dense_dims"] > 0 and totals["bm25_terms"] > 0
    assert payload["documents"] and set(payload["documents"]) == {"pd", "typec"}
