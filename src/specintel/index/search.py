"""WS2 — the search contract exposed to the agent: ``search_text`` and ``search_pages``.

These two functions are what WS4 calls as tools (INTERFACES.md § tool
signatures).  They are the only entry points the agent needs, so the agent can
start against mocks and switch to the real indexes without code changes.

``search_text(query, k, filters)``
    hybrid retrieval: BM25 + dense, optionally reranked, filterable by document,
    revision and element type; returns :class:`~specintel.common.models.SearchHit`
    objects carrying page, section, element ids and table/figure ids.

``search_pages(query, k)``
    vision-path retrieval over page images by MaxSim, mapped back to the text
    elements on those pages.

``hybrid_search`` also returns the per-component contributions, which is what
the trace and the ablation report need.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from specintel.common.models import Chunk, SearchHit
from specintel.index.bm25 import BM25Index, build_bm25
from specintel.index.dense import VectorIndex
from specintel.index.embeddings import EmbeddingBackend, get_embedder
from specintel.index.page_index import MaxSimIndex
from specintel.index.reranker import (LexicalReranker, Reranker, get_reranker, page_key,
                                      reciprocal_rank_fusion)


@dataclass
class Filters:
    """Filters the agent can pass to ``search_text`` (all optional)."""

    docs: Sequence[str] = ()
    revisions: Sequence[str] = ()
    element_types: Sequence[str] = ()
    sections: Sequence[str] = ()
    pages: Sequence[int] = ()
    tables_only: bool = False
    figures_only: bool = False
    #: ablation switches: drop a content type from the text index entirely
    #: ("does the stitched table body actually help, or does the caption alone
    #: carry the answer?" - Table cell accuracy answers the first half of that).
    exclude_tables: bool = False
    exclude_figures: bool = False

    def matches(self, meta: dict[str, Any]) -> bool:
        if self.docs and str(meta.get("doc_id")) not in self.docs:
            return False
        if self.revisions and str(meta.get("revision")) not in self.revisions:
            return False
        if self.element_types and str(meta.get("element_type")) not in self.element_types:
            return False
        if self.sections and str(meta.get("section_number")) not in self.sections:
            return False
        if self.pages and int(meta.get("page") or 0) not in self.pages:
            return False
        if self.tables_only and not meta.get("table_id"):
            return False
        if self.figures_only and not meta.get("figure_id"):
            return False
        if self.exclude_tables and (meta.get("table_id") or meta.get("kind") == "table"):
            return False
        if self.exclude_figures and (meta.get("figure_id") or meta.get("kind") == "figure"):
            return False
        return True

    @staticmethod
    def from_kwargs(**kwargs: Any) -> "Filters":
        return Filters(
            docs=tuple(kwargs.get("docs") or ()),
            revisions=tuple(kwargs.get("revisions") or ()),
            element_types=tuple(kwargs.get("element_types") or ()),
            sections=tuple(kwargs.get("sections") or ()),
            pages=tuple(kwargs.get("pages") or ()),
            tables_only=bool(kwargs.get("tables_only")),
            figures_only=bool(kwargs.get("figures_only")),
        )


@dataclass
class SearchIndex:
    """Everything WS2 built for one corpus: chunks, BM25, dense and page indexes."""

    chunks: list[Chunk] = field(default_factory=list)
    bm25: BM25Index | None = None
    dense: VectorIndex | None = None
    pages: MaxSimIndex | None = None
    embedder: EmbeddingBackend | None = None
    reranker: Reranker | None = None
    elements: list[Any] = field(default_factory=list)
    index_dir: Path | None = None
    build_report: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------ #
    @staticmethod
    def build(chunks: Sequence[Chunk], *, embedder: EmbeddingBackend | None = None,
              reranker: Reranker | None = None, k1: float = 1.5, b: float = 0.75,
              identifier_boost: float = 3.0, elements: Sequence[Any] = (),
              verbose: bool = False) -> "SearchIndex":
        embedder = embedder or get_embedder()
        reranker = reranker or get_reranker()
        index = SearchIndex(chunks=list(chunks), embedder=embedder, reranker=reranker, elements=list(elements))
        index.bm25 = build_bm25(chunks, k1=k1, b=b, identifier_boost=identifier_boost)
        index.dense = VectorIndex().build([c.chunk_id for c in chunks], [c.text for c in chunks], embedder,
                                          meta=[_chunk_meta(c) for c in chunks])
        index.build_report = {
            "chunks": len(chunks), "bm25_terms": len(index.bm25.postings),
            "dense_dims": index.dense.dim, "embedder": embedder.name, "reranker": reranker.name,
        }
        return index

    # -- search ------------------------------------------------------------ #
    def _doc_filter(self, filters: Filters) -> Callable[[int], bool] | None:
        if not (filters.docs or filters.revisions or filters.element_types or filters.sections
                or filters.pages or filters.tables_only or filters.figures_only):
            return None
        metas = self.bm25.meta if self.bm25 else [c.meta for c in self.chunks]
        return lambda i: filters.matches(metas[i] if i < len(metas) else {})

    def search_text(self, query: str, k: int = 10, *, filters: Filters | None = None,
                    hybrid: bool = True, rerank: bool = True, bm25_k: int | None = None,
                    dense_k: int | None = None) -> list[SearchHit]:
        """Hybrid BM25 + dense retrieval, optionally reranked (Task 2, Step 7)."""
        filters = filters or Filters()
        # Ablation switches (scripts/ablate.py).  They are read from the
        # environment so the ablation runner can compare architectures without
        # forking the code; unset in normal runs, so nothing changes by default.
        # ("drop BM25 and keep dense": the dense ranking is what `hybrid` adds,
        # so the switch removes the BM25 ranking rather than clearing `hybrid`)
        if os.environ.get("SPECINTEL_ABLATE_RERANKER"):
            rerank = False
        if os.environ.get("SPECINTEL_ABLATE_TABLE_ROWS"):
            filters = replace(filters, exclude_tables=True)
        if os.environ.get("SPECINTEL_ABLATE_FIGURE_TEXT"):
            filters = replace(filters, exclude_figures=True)
        fetch = max(k * 4, 20)
        bm25_k = bm25_k or fetch
        dense_k = dense_k or fetch

        rankings: dict[str, list[dict[str, Any]]] = {}
        if self.bm25 is not None and not os.environ.get("SPECINTEL_ABLATE_BM25"):
            rankings["bm25"] = self.bm25.search(query, k=bm25_k, doc_filter=self._doc_filter(filters))
        if hybrid and self.dense is not None and self.embedder is not None:
            rankings["dense"] = self.dense.search(query, self.embedder, k=dense_k,
                                                  doc_filter=self._doc_filter(filters))
        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings, weights={"bm25": 1.0, "dense": 1.0}, id_key="chunk_id")
        by_id = {c.chunk_id: c for c in self.chunks}
        hits: list[SearchHit] = []
        for item in fused:
            chunk = by_id.get(item["chunk_id"])
            if chunk is None:
                continue
            hits.append(_to_hit(chunk, item["score"], item.get("component_scores", {}), source="text"))
        if rerank and self.reranker is not None and hits:
            payload = [{"chunk_id": h.chunk_id, "score": h.score, "text": h.text} for h in hits]
            reranked = self.reranker.rerank(query, payload)
            order = {r["chunk_id"]: (r.get("rerank_score", r["score"]), r.get("rerank_parts", {}))
                     for r in reranked}
            for hit in hits:
                score, parts = order.get(hit.chunk_id, (hit.score, {}))
                hit.meta["rerank_parts"] = parts
                hit.component_scores["rerank"] = score
                hit.score = score
            hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def search_pages(self, query: str, k: int = 10, *, docs: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Page-image retrieval by MaxSim, mapped back to that page's elements."""
        if self.pages is None:
            return []
        doc_filter = (lambda doc_id, page: doc_id in set(docs)) if docs else None
        raw = self.pages.search(query, k=k, doc_filter=doc_filter)
        by_page: dict[tuple[str, int], list[Any]] = {}
        for el in self.elements:
            by_page.setdefault((el.doc_id, el.page), []).append(el)
        for hit in raw:
            els = by_page.get((hit["doc_id"], hit["page"]), [])
            hit["element_ids"] = [e.element_id for e in els]
            hit["sections"] = sorted({e.section_number for e in els if e.section_number})
            hit["text"] = " ".join(e.text for e in els
                                   if e.element_type in ("paragraph", "note", "table"))[:1200]
        return raw

    def search_fused(self, query: str, k: int = 10, *, filters: Filters | None = None) -> list[SearchHit]:
        """Fuse the text path and the vision path with RRF (Task 2, Step 7)."""
        text_hits = self.search_text(query, k=max(k, 10), filters=filters)
        page_hits = self.search_pages(query, k=max(k, 10))
        rankings = {
            "text": [{"chunk_id": h.chunk_id, "score": h.score, "text": h.text} for h in text_hits],
            "pages": page_hits,
        }
        fused = reciprocal_rank_fusion(rankings, weights={"text": 1.0, "pages": 1.0},
                                       key_fn=lambda h: h.get("chunk_id") or page_key(h))
        by_id = {c.chunk_id: c for c in self.chunks}
        out: list[SearchHit] = []
        for item in fused[:k]:
            chunk = by_id.get(item.get("chunk_id", ""))
            if chunk is not None:
                out.append(_to_hit(chunk, item["score"], item.get("component_scores", {}), source="fused"))
            else:
                out.append(SearchHit(chunk_id=page_key(item), score=item["score"], doc_id=item.get("doc_id", ""),
                                     page=int(item.get("page") or 0), element_type="page_image",
                                     text=str(item.get("text") or "")[:1200], source="pages",
                                     element_ids=list(item.get("element_ids") or []),
                                     component_scores=item.get("component_scores", {}),
                                     meta={"image_path": item.get("image_path")}))
        return out


def _chunk_meta(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id, "revision": chunk.revision, "page": chunk.page, "page_end": chunk.page_end,
        "section_number": chunk.section_number, "element_type": chunk.element_type,
        "table_id": chunk.table_id, "figure_id": chunk.figure_id, "parent_chunk_id": chunk.parent_chunk_id,
    }


def _to_hit(chunk: Chunk, score: float, components: dict[str, float], source: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id, score=score, doc_id=chunk.doc_id, page=chunk.page, page_end=chunk.page_end,
        section_number=chunk.section_number, element_type=chunk.element_type, text=chunk.text,
        source=source, bbox=chunk.bbox, element_ids=list(chunk.element_ids), table_id=chunk.table_id,
        figure_id=chunk.figure_id, component_scores=components,
        # revision belongs in the citation ("[pd, R3.2 V1.2, §6.6, p.4, Table 6.8]"),
        # and SearchHit keeps it in `meta` so the contract stays as published
        meta={"parent_chunk_id": chunk.parent_chunk_id, "kind": chunk.meta.get("kind"),
              "image_path": chunk.meta.get("image_path"), "revision": chunk.revision,
              "table_number": chunk.meta.get("table_number"),
              "figure_number": chunk.meta.get("figure_number")},
    )


# --------------------------------------------------------------------------- #
# convenience wrappers used by the agent and the CLI
# --------------------------------------------------------------------------- #
def search_text(index: SearchIndex, query: str, k: int = 10, **filters: Any) -> list[SearchHit]:
    """Public tool signature: ``search_text(query, k, filters)``."""
    return index.search_text(query, k=k, filters=Filters.from_kwargs(**filters))


def search_pages(index: SearchIndex, query: str, k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
    """Public tool signature: ``search_pages(query, k)``."""
    return index.search_pages(query, k=k, docs=kwargs.get("docs"))


IDENTIFIER_QUERY_RE = re.compile(
    r"\b(?:t[A-Z][A-Za-z0-9]+|TEST\.[A-Z0-9.]+|TD\s*\d+(?:\.\d+)+|[A-Z]{2,}_[A-Za-z_]+|vRd-[0-9.]+)\b")


def looks_identifier_heavy(query: str) -> bool:
    """Route identifier-heavy questions to keyword + graph tools (WS4 planner)."""
    return bool(IDENTIFIER_QUERY_RE.search(query))
