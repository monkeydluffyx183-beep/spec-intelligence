"""WS3 — graph build: elements + tables + figures → validated, stored knowledge graph.

Pipeline (Task 3, Steps 1–3):

1. **rules** per document (deterministic, precise, provenance-carrying);
2. **LLM/VLM pass** for prose relations, schema-constrained and validated;
3. **entity resolution** with a logged merge trail;
4. **schema validation** — everything rejected is written out with its reason;
5. **storage** in SQLite plus JSON / GraphML / Cypher exports and Mermaid
   subgraph visualisations;
6. **statistics** (node/edge counts by type, orphans, merges, stale references)
   and a triple-precision sample for manual checking.

Run it with:
    python scripts/build_graph.py                 # every document
    python scripts/build_graph.py --docs pd,typec --no-llm
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from specintel.common.logging import get_logger, log_event, log_exception
from specintel.common.models import Edge, Figure, Node, Table
from specintel.common.resources import rss_mb
from specintel.kg.llm_extractor import LLMExtractor, LLMExtractionReport
from specintel.kg.resolve import resolve_entities
from specintel.kg.rules import ExtractionContext, RuleExtractor
from specintel.kg.schema import KG_SCHEMA, schema_markdown, validate_graph
from specintel.kg.store import GraphStore, visualise_subgraph


@dataclass
class GraphBuildReport:
    documents: dict[str, Any] = field(default_factory=dict)
    nodes: int = 0
    edges: int = 0
    rejected: int = 0
    merges: int = 0
    duplicate_rate: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    llm_pass: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    sample_triples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"documents": self.documents, "nodes": self.nodes, "edges": self.edges,
                "rejected": self.rejected, "merges": self.merges,
                "duplicate_rate": round(self.duplicate_rate, 4), "timings": self.timings,
                "llm_pass": self.llm_pass, "stats": self.stats, "sample_triples": self.sample_triples}


def build_graph(config: Any, *, doc_ids: Sequence[str] | None = None, use_llm: bool = True,
                sample_size: int = 100, store_path: str | Path | None = None
                ) -> tuple[GraphStore, GraphBuildReport]:
    """Build the knowledge graph from the ingested artefacts."""
    log = get_logger("ws3.graph", config)
    report = GraphBuildReport()
    t_start = time.perf_counter()

    from specintel.ingest.pipeline import load_elements, load_figures, load_tables

    docs = [d for d in config.documents() if not doc_ids or d["doc_id"] in doc_ids]
    other_docs = {d["doc_id"]: d for d in config.documents()}

    all_nodes: list[Node] = []
    all_edges: list[Edge] = []
    extractor = RuleExtractor()
    llm_extractor = LLMExtractor(get_llm_for_extraction(config), max_elements=int(
        config.get("kg.extraction.max_elements_per_pass", 0) or 0)) if use_llm else None

    for doc in docs:
        doc_id = doc["doc_id"]
        elements = load_elements(config, doc_ids=[doc_id])
        if not elements:
            log_event(log, "graph.skipped", level=30, doc=doc_id, reason="no elements ingested")
            continue
        ctx = ExtractionContext(doc_id=doc_id, revision=str(doc.get("revision", "")),
                                revision_key=str(doc.get("revision_key", "0")), title=str(doc.get("title", "")),
                                elements=elements, tables=load_tables(config, doc_id),
                                figures=load_figures(config, doc_id), other_docs=other_docs)
        t0 = time.perf_counter()
        rule_result = extractor.extract(ctx)
        rule_time = time.perf_counter() - t0

        llm_result = None
        if llm_extractor is not None:
            t0 = time.perf_counter()
            try:
                llm_result, llm_report = llm_extractor.run(ctx, base=rule_result)
                report.llm_pass = llm_report.to_dict()
            except Exception as exc:
                log_exception(log, "graph.llm.failed", exc, doc=doc_id)
                report.llm_pass = {"backend": "error", "notes": [str(exc)]}
            rule_time += time.perf_counter() - t0

        nodes = list(rule_result.nodes)
        edges = list(rule_result.edges)
        if llm_result is not None:
            nodes.extend(n for n in llm_result.nodes if n.node_id not in {x.node_id for x in nodes})
            edges.extend(llm_result.edges)

        all_nodes.extend(nodes)
        all_edges.extend(edges)
        report.documents[doc_id] = {
            "elements": len(elements), "tables": len(ctx.tables), "figures": len(ctx.figures),
            "nodes": len(nodes), "edges": len(edges), "extract_time_s": round(rule_time, 3),
        }
        log_event(log, "graph.doc", doc=doc_id, elements=len(elements), nodes=len(nodes), edges=len(edges),
                  tables=len(ctx.tables), figures=len(ctx.figures), duration_s=round(rule_time, 3))

    report.timings["extract_s"] = round(time.perf_counter() - t_start, 3)

    # -- resolution -------------------------------------------------------- #
    t0 = time.perf_counter()
    resolution = resolve_entities(all_nodes, all_edges,
                                  similarity_threshold=float(config.get("kg.entity_resolution.similarity_threshold", 0.86)))
    report.merges = len(resolution.merges)
    report.duplicate_rate = resolution.duplicate_rate
    report.timings["resolve_s"] = round(time.perf_counter() - t0, 3)
    log_event(log, "graph.resolve", merges=len(resolution.merges), nodes_in=len(all_nodes),
              nodes_out=len(resolution.nodes), duplicate_rate=round(resolution.duplicate_rate, 4))
    for merge in resolution.merges[:10]:
        log_event(log, "graph.merge", canonical=merge.canonical_name, merged=merge.merged_name,
                  reason=merge.reason, score=round(merge.score, 3))

    # -- validation -------------------------------------------------------- #
    t0 = time.perf_counter()
    accepted_nodes, accepted_edges, issues = validate_graph(
        resolution.nodes, resolution.edges,
        require_provenance=bool(config.get("kg.validation.require_provenance", True)))
    report.rejected = len(issues)
    report.timings["validate_s"] = round(time.perf_counter() - t0, 3)
    if issues:
        log_event(log, "graph.rejected", level=30, n=len(issues),
                  sample=[i.to_dict() for i in issues[:5]])
    for issue in issues[:20]:
        log_event(log, "graph.reject.sample", level=30, **issue.to_dict())

    # -- storage ----------------------------------------------------------- #
    t0 = time.perf_counter()
    path = Path(store_path) if store_path else config.path("kg.store_path")
    store = GraphStore(path)
    store.reset()                     # a rebuild replaces the graph, never appends to it
    store.add_nodes(accepted_nodes)
    store.add_edges(accepted_edges)
    store.set_meta("merges", len(resolution.merges))
    store.set_meta("duplicate_rate", resolution.duplicate_rate)
    store.set_meta("documents", [d["doc_id"] for d in docs])
    store.set_meta("rejected", len(issues))
    report.timings["store_s"] = round(time.perf_counter() - t0, 3)

    stats = store.stats(merges=len(resolution.merges), duplicate_rate=resolution.duplicate_rate)
    report.nodes, report.edges = stats.nodes, stats.edges
    report.stats = stats.to_dict()

    # -- exports ----------------------------------------------------------- #
    graph_dir = config.path("paths.graph")
    graph_dir.mkdir(parents=True, exist_ok=True)
    store.export_json(graph_dir)
    store.export_graphml(graph_dir / "graph.graphml")
    store.export_cypher(graph_dir / "graph.cypher")
    (graph_dir / "rejections.jsonl").write_text(
        "\n".join(json.dumps(i.to_dict(), default=str) for i in issues), encoding="utf-8")
    (graph_dir / "stats.json").write_text(json.dumps(report.to_dict(), indent=1, default=str), encoding="utf-8")

    # -- visualisations ---------------------------------------------------- #
    _write_visualisations(config, store, graph_dir, log)

    report.sample_triples = sample_triples(store, sample_size)
    (graph_dir / "triple_sample.jsonl").write_text(
        "\n".join(json.dumps(t, default=str) for t in report.sample_triples), encoding="utf-8")

    # -- schema documentation --------------------------------------------- #
    docs_dir = config.root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "KG_SCHEMA.md").write_text(schema_markdown(), encoding="utf-8")

    report.timings["total_s"] = round(time.perf_counter() - t_start, 3)
    log_event(log, "graph.done", nodes=report.nodes, edges=report.edges, rejected=report.rejected,
              merges=report.merges, stale=stats.stale_references, cross_doc=stats.cross_document_edges,
              orphans=stats.orphan_nodes, rss_mb=rss_mb(), duration_s=report.timings["total_s"])
    return store, report


def get_llm_for_extraction(config: Any) -> Any:
    """Resolver kept separate so ``--no-llm`` runs never touch the network."""
    from specintel.common.llm import get_llm

    return get_llm(config)


def _write_visualisations(config: Any, store: GraphStore, graph_dir: Path, log: Any) -> None:
    """At least two subgraph visualisations, as the assignment requires."""
    targets = [
        ("Source Policy Engine state diagram", "PE_SRC_Send_Capabilities", 2,
         ("TRANSITIONS_TO", "STARTS_TIMER", "STOPS_TIMER", "HAS_STATE")),
        ("TEST.PD.PROT.SRC.3 and everything it verifies", "TEST.PD.PROT.SRC.3", 3,
         ("VERIFIES", "TESTS", "PART_OF", "DEPENDS_ON", "REFERENCES")),
        ("Shared capacity charging across documents", "Shared Capacity Charging", 2,
         ("REFERENCES", "DEFINED_IN", "VERIFIES")),
        ("USB4 tunnelling", "PCIe Adapter", 2, ("TUNNELS", "CONTAINS", "REFERENCES")),
    ]
    written = []
    for i, (title, seed, depth, edge_types) in enumerate(targets, start=1):
        try:
            path = visualise_subgraph(store, seed, graph_dir / f"subgraph_{i}.mmd", depth=depth,
                                      edge_types=edge_types, title=title)
            written.append(str(path))
        except Exception as exc:                      # visualisation must never break a build
            log_exception(log, "graph.visualise.failed", exc, seed=seed)
    if written:
        log_event(log, "graph.visualisations", files=written)


def sample_triples(store: GraphStore, size: int = 100, seed: int = 17) -> list[dict[str, Any]]:
    """Random triples with their provenance, for the hand-checked precision figure."""
    edges = store.all_edges()
    if not edges:
        return []
    rng = random.Random(seed)
    picked = rng.sample(edges, min(size, len(edges)))
    out: list[dict[str, Any]] = []
    for edge in picked:
        src = store.get_node(edge.src_id)
        dst = store.get_node(edge.dst_id)
        out.append({
            "triple": f"{src.name if src else edge.src_id} -[{edge.edge_type}]-> "
                      f"{dst.name if dst else edge.dst_id}",
            "src_type": src.node_type if src else "?",
            "dst_type": dst.node_type if dst else "?",
            "provenance": {"doc_id": edge.doc_id, "revision": edge.revision, "section": edge.section_number,
                           "page": edge.page, "element_id": edge.element_id,
                           "method": edge.extraction_method, "confidence": edge.confidence},
            "evidence": edge.evidence[:240],
            "stale": edge.stale,
            "verdict": "",          # a human fills correct/incorrect during the precision check
        })
    return out


def load_graph(config: Any, path: str | Path | None = None) -> GraphStore:
    """Open the persisted graph store."""
    return GraphStore(Path(path) if path else config.path("kg.store_path"))


def graph_query(config: Any, query: dict[str, Any], *, store: GraphStore | None = None) -> dict[str, Any]:
    """Run a query described as a plain dict (used by the CLI and the tests)."""
    from specintel.common.models import GraphQuery

    store = store or load_graph(config)
    request = GraphQuery(**{k: v for k, v in query.items() if k in GraphQuery.__dataclass_fields__})
    result = store.query(request)
    return result.to_dict()


__all__ = ["build_graph", "load_graph", "graph_query", "sample_triples", "GraphBuildReport", "KG_SCHEMA"]
