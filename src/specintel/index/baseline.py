"""WS2 — the naive RAG baseline (config A) and the hybrid/vision configs.

The baseline is deliberately simple — embed the question, take the top-k dense
chunks, generate an answer with citations — because it is the reference every
other workstream is measured against (Task 2, Step 7 and Task 5, Step 4).

The same module exposes the intermediate retrieval strategies so the ablation
runner can ask for config A…F without duplicating retrieval code:

===== ==========================================================================
A     naive dense RAG over plain extracted text
B     A + hybrid search (BM25 + dense) + reranker
C     vision path only (capstone 04 approach)
D     B + C fused, with structured tables and figures
E     D + knowledge-graph tool
F     E + agent loop with planning, reflection and citation verification
===== ==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from specintel.common.llm import ExtractiveLLM, generate, get_llm
from specintel.common.models import AgentTrace, Citation, SearchHit, TraceStep, new_id
from specintel.index.search import Filters, SearchIndex

PROMPT_TEMPLATE = (
    "Answer the engineering question using only the evidence below. Cite every factual sentence as "
    "[doc, revision, §section, p.page, region]. If the evidence does not contain the answer, say so and "
    "explain what was searched. Do not use any knowledge outside the evidence.\n\n"
    "Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
)


@dataclass
class BaselineConfig:
    """Which retrieval strategy a config uses (see the table above)."""

    name: str = "A"
    description: str = "naive dense RAG over plain extracted text"
    use_bm25: bool = False
    use_dense: bool = True
    use_hybrid: bool = False
    use_rerank: bool = False
    use_pages: bool = False
    use_fusion: bool = False
    use_graph: bool = False
    use_agent: bool = False


CONFIGS: dict[str, BaselineConfig] = {
    "A": BaselineConfig("A", "naive dense RAG over plain extracted text"),
    "B": BaselineConfig("B", "A + hybrid search (BM25 + dense) + reranker",
                        use_bm25=True, use_hybrid=True, use_rerank=True),
    "C": BaselineConfig("C", "vision path only (capstone 04 approach)", use_dense=False, use_pages=True),
    "D": BaselineConfig("D", "B + C fused, with structured tables and figures",
                        use_bm25=True, use_hybrid=True, use_rerank=True, use_pages=True, use_fusion=True),
    "E": BaselineConfig("E", "D + knowledge-graph tool",
                        use_bm25=True, use_hybrid=True, use_rerank=True, use_pages=True, use_fusion=True,
                        use_graph=True),
    "F": BaselineConfig("F", "E + agent loop with planning, reflection and citation verification",
                        use_bm25=True, use_hybrid=True, use_rerank=True, use_pages=True, use_fusion=True,
                        use_graph=True, use_agent=True),
}


def get_config(name: str) -> BaselineConfig:
    return CONFIGS.get(name.upper(), CONFIGS["A"])


@dataclass
class Baselines:
    """Runs the naive and intermediate strategies over one index."""

    index: SearchIndex
    llm: Any = None
    top_k: int = 8
    config: BaselineConfig = field(default_factory=lambda: CONFIGS["A"])

    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = ExtractiveLLM()

    # -- retrieval per config --------------------------------------------- #
    def retrieve(self, question: str, config: BaselineConfig | None = None, k: int | None = None
                 ) -> list[SearchHit]:
        cfg = config or self.config
        k = k or self.top_k
        if cfg.use_fusion and cfg.use_pages:
            return self.index.search_fused(question, k=k)
        if cfg.use_pages and not cfg.use_dense and not cfg.use_bm25:
            return _page_hits_to_hits(self.index.search_pages(question, k=k))
        hits = self.index.search_text(question, k=k, hybrid=cfg.use_hybrid or (cfg.use_dense and cfg.use_bm25),
                                      rerank=cfg.use_rerank)
        return hits

    # -- end-to-end naive RAG --------------------------------------------- #
    def answer(self, question: str, config: BaselineConfig | None = None, k: int | None = None
               ) -> tuple[str, list[Citation], list[SearchHit], AgentTrace]:
        cfg = config or self.config
        hits = self.retrieve(question, cfg, k=k)
        evidence = [{"text": h.text, "citation": _marker(h)} for h in hits]
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            evidence="\n\n".join(f"[{i + 1}] {h.locator} ({h.element_type})\n{h.text}"
                                 for i, h in enumerate(hits)) or "(no evidence retrieved)",
        )
        response = generate(self.llm, prompt, evidence=evidence, question=question)
        citations = [_citation(h) for h in hits if h.text][:6]
        trace = AgentTrace(
            run_id=new_id("run"), question=question, config_name=cfg.name, model=response.model,
            plan=[f"config {cfg.name}: {cfg.description}"],
            steps=[TraceStep(step=1, kind="tool", name="retrieve", args={"k": k or self.top_k},
                             n_hits=len(hits), top_score=hits[0].score if hits else None,
                             observation=f"top hit {hits[0].locator}" if hits else "no hits",
                             duration_s=response.latency_s, tokens=response.total_tokens,
                             decision="generate" if hits else "decline")],
            answer=response.text, citations=citations,
            declined=not response.text.strip(), decline_reason="" if response.text.strip() else "no evidence",
            confidence=hits[0].score if hits else 0.0,
            token_usage={"prompt": response.prompt_tokens, "completion": response.completion_tokens},
            latency_s=response.latency_s,
            warnings=[w for w in (response.error,) if w],
        )
        return response.text, citations, hits, trace


def _marker(hit: SearchHit) -> str:
    return _citation(hit).to_marker()


def _citation(hit: SearchHit) -> Citation:
    region = ""
    if hit.table_id:
        region = f"table {hit.meta.get('table_number') or hit.table_id}"
    elif hit.figure_id:
        region = f"figure {hit.meta.get('figure_number') or hit.figure_id}"
    elif hit.section_number:
        region = f"§{hit.section_number}"
    return Citation(doc_id=hit.doc_id, revision=str(hit.meta.get("revision", "")),
                    section_number=hit.section_number, page=hit.page, page_end=hit.page_end,
                    region=region, element_ids=list(hit.element_ids),
                    quote=hit.text[:240])


def _page_hits_to_hits(page_hits: Sequence[dict[str, Any]]) -> list[SearchHit]:
    """Vision-only config: page hits presented through the shared hit type."""
    out: list[SearchHit] = []
    for hit in page_hits:
        out.append(SearchHit(
            chunk_id=f"{hit['doc_id']}:p{hit['page']}", score=float(hit["score"]), doc_id=hit["doc_id"],
            page=int(hit["page"]), element_type="page_image", text=str(hit.get("text", ""))[:1200],
            source="pages", element_ids=list(hit.get("element_ids") or []),
            meta={"image_path": hit.get("image_path"), "sections": hit.get("sections", [])},
        ))
    return out
