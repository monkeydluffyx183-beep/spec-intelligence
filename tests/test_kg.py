"""WS3 — the knowledge graph: schema, rules, entity resolution, provenance."""

from __future__ import annotations

import pytest

from specintel.kg import schema as kg_schema


@pytest.fixture(scope="session")
def built_graph(ingested_tiny):
    """Build the graph over the fixture corpus, once per session."""
    from specintel.kg.build import build_graph

    return build_graph(ingested_tiny, doc_ids=["pd", "typec"], use_llm=False)


def test_schema_declares_the_ontology_the_tasks_need():
    """Task 3 names timers, parameters, states and test cases explicitly."""
    required = {"Timer", "Parameter", "Message", "TestCase", "Section", "Document",
                "Figure", "Table", "Field", "PowerProfile", "Role", "State"}
    assert required <= set(kg_schema.NODE_TYPES), sorted(required - set(kg_schema.NODE_TYPES))
    # "test cases linked to requirements, timers and states" (Task 3) needs these
    assert {"DEFINED_IN", "REFERENCES", "VERIFIES", "TESTS", "STARTS_TIMER", "STOPS_TIMER",
            "HAS_STATE", "TRANSITIONS_TO", "HAS_VALUE", "SUPERSEDES"} <= set(kg_schema.EDGE_TYPES)


def test_validation_rejects_edges_without_provenance():
    node = kg_schema.Node(node_id="timer:t", node_type="Timer", name="tSenderResponse", doc_id="pd")
    kg_schema.validate_node(node)
    types = {"timer:t": "Timer", "section:6.6": "Section", "a": "Timer", "b": "Section"}
    edge = kg_schema.Edge(edge_id="e", src_id="timer:t", dst_id="section:6.6", edge_type="DEFINED_IN",
                          doc_id="pd", revision="R3.2 V1.2", section_number="6.6", page=3,
                          element_id="pd:table:6.6:1")
    assert kg_schema.validate_edge(edge, types, require_provenance=True) == []
    thin = kg_schema.Edge(edge_id="e2", src_id="a", dst_id="b", edge_type="DEFINED_IN", doc_id="pd")
    issues = kg_schema.validate_edge(thin, types, require_provenance=True)
    assert issues, "an edge with no page, section or element is rejected"
    assert any("provenance" in i.reason for i in issues), [i.to_dict() for i in issues]


def test_every_edge_carries_document_section_page_and_method(built_graph):
    store, _report = built_graph
    edges = store.all_edges()
    assert edges, "the fixture corpus yields relations"
    for edge in edges:
        assert edge.doc_id, "an edge without a document cannot be cited"
        assert edge.page >= 1
        assert edge.section_number or edge.edge_type in ("PART_OF", "MENTIONS")
        assert edge.extraction_method in ("rule", "llm", "vlm")


def test_rule_extraction_finds_the_timer_of_the_fixture_corpus(built_graph):
    store, _report = built_graph
    assert store.lookup("tSenderResponse"), "the timer in Table 6.8 becomes a node"
    nodes, edges = store.neighbors(store.lookup("tSenderResponse")[0].node_id, depth=1)
    node_ids = {n.node_id for n in nodes}
    assert any("Table" in n_id for n_id in node_ids), \
        "the timer is linked to the table that defines it"
    assert any("Section" in n_id for n_id in node_ids), "…and to the section"
    assert any(e.edge_type == "DEFINED_IN" for e in edges)
    assert all(e.page >= 1 for e in edges), "every relation keeps its page"


def test_entity_resolution_merges_case_and_separator_variants():
    from specintel.kg.resolve import EntityResolver

    from specintel.kg.resolve import resolve_entities
    from specintel.kg.schema import Edge, Node

    nodes = [Node(node_id="timer:1", node_type="Timer", name="tSenderResponse", doc_id="pd"),
             Node(node_id="timer:2", node_type="Timer", name="tsenderresponse", doc_id="pd"),
             Node(node_id="timer:3", node_type="Timer", name="tPSTransition", doc_id="pd")]
    result = resolve_entities(nodes, [], similarity_threshold=0.86)
    assert result.merges, "case variants of one timer merge"
    survivors = {n.name for n in result.nodes}
    assert len([n for n in result.nodes if n.node_type == "Timer"]) == 2, survivors
    assert {"tSenderResponse", "tPSTransition"} <= survivors
    assert sum(result.merges[0].to_dict().get("count", 1) for _ in [0]) >= 1


def test_graph_store_reset_clears_previous_build(built_graph, ingested_tiny):
    """A rebuild must not accumulate: an earlier bug doubled every node."""
    from specintel.kg.build import build_graph

    store, _report = built_graph
    before = store.node_count()
    store.reset()
    assert store.node_count() == 0
    rebuilt_store, _ = build_graph(ingested_tiny, doc_ids=["pd", "typec"], use_llm=False)
    assert rebuilt_store.node_count() == before, "the same input must build the same graph"


def test_provenance_lookup_returns_a_citable_location(built_graph):
    store, _report = built_graph
    nodes = store.lookup("tSenderResponse")
    node = nodes[0]
    assert node.doc_id == "pd" and node.page >= 1
    assert node.revision


def test_graph_query_kinds_are_consistent(built_graph):
    from specintel.common.models import GraphQuery

    store, _report = built_graph
    stats = store.stats()
    assert stats.nodes > 0 and stats.edges > 0
    assert sum(stats.nodes_by_type.values()) == stats.nodes
    result = store.query(GraphQuery(kind="neighbors", name="tSenderResponse", depth=1))
    assert result.nodes
    missing = store.query(GraphQuery(kind="lookup", name="definitely-not-an-entity"))
    assert missing.nodes == [] and missing.notes


def test_documents_stale_references_are_flagged_not_followed(needs_corpus, config):
    """The assignment's trap: a test spec cites a base spec's older revision.

    The flag lives on the edge, and the report counts it; nothing deletes the
    edge, because a reader still needs to see what the document claims.
    """
    from specintel.kg.build import build_graph

    store, _report = build_graph(config, use_llm=False)
    stats = store.stats()
    assert stats.cross_document_edges > 0, "the corpus links documents to each other"
    stale = [e for e in store.all_edges() if e.stale]
    assert stale, "at least one cross-document reference points at a superseded revision"
    for edge in stale:
        assert edge.doc_id, "even a stale edge keeps its provenance"
