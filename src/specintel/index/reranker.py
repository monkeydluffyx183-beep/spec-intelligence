"""WS2 — reranking and fusion.

``LexicalReranker``            deterministic cross-attention-free reranker that
                              scores query/term coverage, identifier matches and
                              proximity; the offline default.
``CrossEncoderReranker``      a real cross-encoder (``bge-reranker-base``) via
                              ``sentence-transformers`` when installed.
``reciprocal_rank_fusion``    RRF over several ranked lists (text, pages, graph),
                              with per-list weights — the fusion of Task 2, Step 7
                              and the "fused" column of the ablation table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.index.bm25 import identifiers_in, tokenize

# weights for the lexical reranker's components (kept in one place so the
# ablation notes can explain exactly what changed between configs)
COVERAGE_WEIGHT = 1.0
IDENTIFIER_WEIGHT = 2.5
PROXIMITY_WEIGHT = 0.35
LENGTH_PENALTY = 0.0004


class Reranker:
    """Interface: reorder candidate hits for a query."""

    name = "base"

    def rerank(self, query: str, hits: Sequence[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class LexicalReranker(Reranker):
    """Score candidates by query coverage, identifier hits and term proximity."""

    name: str = "lexical"
    text_key: str = "text"

    def score(self, query: str, text: str) -> dict[str, float]:
        q_terms = [t for t in dict.fromkeys(tokenize(query))]
        d_terms = tokenize(text)
        if not q_terms:
            return {"coverage": 0.0, "identifiers": 0.0, "proximity": 0.0, "total": 0.0}
        d_set = set(d_terms)
        covered = sum(1 for t in q_terms if t in d_set)
        coverage = covered / len(q_terms)

        q_ids = identifiers_in(query)
        d_ids = identifiers_in(text)
        identifier_score = (len(q_ids & d_ids) / len(q_ids)) if q_ids else 0.0

        proximity = 0.0
        positions = [d_terms.index(t) for t in q_terms if t in d_set]
        if len(positions) > 1:
            span = max(positions) - min(positions) + 1
            proximity = min(1.0, len(positions) / span)

        total = (COVERAGE_WEIGHT * coverage + IDENTIFIER_WEIGHT * identifier_score
                 + PROXIMITY_WEIGHT * proximity - LENGTH_PENALTY * len(d_terms))
        return {"coverage": coverage, "identifiers": identifier_score, "proximity": proximity, "total": total}

    def rerank(self, query: str, hits: Sequence[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for hit in hits:
            text = str(hit.get(self.text_key) or hit.get("meta", {}).get("text") or "")
            parts = self.score(query, text)
            enriched = dict(hit)
            enriched["rerank_score"] = parts["total"]
            enriched["rerank_parts"] = parts
            scored.append(enriched)
        scored.sort(key=lambda h: -h["rerank_score"])
        return scored[:k] if k else scored


@dataclass
class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker (needs ``sentence-transformers`` and a model)."""

    name: str = "cross-encoder"
    model_name: str = "BAAI/bge-reranker-base"
    _model: Any = field(default=None, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: Sequence[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
        model = self._load()
        pairs = [(query, str(h.get("text") or h.get("meta", {}).get("text") or "")) for h in hits]
        if not pairs:
            return []
        scores = model.predict(pairs)
        scored = []
        for hit, score in zip(hits, scores):
            enriched = dict(hit)
            enriched["rerank_score"] = float(score)
            scored.append(enriched)
        scored.sort(key=lambda h: -h["rerank_score"])
        return scored[:k] if k else scored


# --------------------------------------------------------------------------- #
# reciprocal rank fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: dict[str, Sequence[dict[str, Any]]], *, k: int = 60,
                           weights: dict[str, float] | None = None, id_key: str = "chunk_id",
                           key_fn: Any | None = None) -> list[dict[str, Any]]:
    """Fuse ranked lists with RRF: ``score(d) = Σ w_l / (k + rank_l(d))``.

    ``key_fn`` lets page hits (which have no chunk id) participate by building a
    key such as ``doc_id:page``; every contributing score is recorded so the
    trace can show *why* a result surfaced.
    """
    weights = weights or {}
    fused: dict[str, dict[str, Any]] = {}
    for list_name, hits in rankings.items():
        weight = float(weights.get(list_name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            key = str(key_fn(hit)) if key_fn else str(hit.get(id_key) or hit.get("id") or "")
            if not key:
                continue
            entry = fused.setdefault(key, {"id": key, "hit": dict(hit), "score": 0.0, "components": {}})
            contribution = weight / (k + rank)
            entry["score"] += contribution
            entry["components"][list_name] = round(contribution, 6)
            # prefer the richer payload (a text hit carries text/page metadata)
            if len(str(hit.get("text") or "")) > len(str(entry["hit"].get("text") or "")):
                merged = dict(entry["hit"])
                merged.update({kk: vv for kk, vv in hit.items() if vv not in (None, "", [])})
                entry["hit"] = merged
            entry["hit"]["source"] = list_name if not entry["components"].get("__fused__") else "fused"
    out = sorted(fused.values(), key=lambda e: -e["score"])
    for entry in out:
        entry["hit"]["score"] = entry["score"]
        entry["hit"]["component_scores"] = entry["components"]
        entry["hit"]["source"] = "fused" if len(entry["components"]) > 1 else next(iter(entry["components"]), "fused")
    return [entry["hit"] for entry in out]


def weighted_score_fusion(rankings: dict[str, Sequence[dict[str, Any]]], *, weights: dict[str, float] | None = None,
                          key_fn: Any | None = None) -> list[dict[str, Any]]:
    """Score-sum fusion for lists whose scores are already comparable."""
    weights = weights or {}
    fused: dict[str, dict[str, Any]] = {}
    for list_name, hits in rankings.items():
        weight = float(weights.get(list_name, 1.0))
        for hit in hits:
            key = str(key_fn(hit)) if key_fn else str(hit.get("chunk_id") or hit.get("id") or "")
            if not key:
                continue
            entry = fused.setdefault(key, {"id": key, "hit": dict(hit), "score": 0.0, "components": {}})
            entry["score"] += weight * float(hit.get("score", 0.0))
            entry["components"][list_name] = round(float(hit.get("score", 0.0)), 6)
    out = sorted(fused.values(), key=lambda e: -e["score"])
    for entry in out:
        entry["hit"]["score"] = entry["score"]
        entry["hit"]["component_scores"] = entry["components"]
    return [entry["hit"] for entry in out]


def page_key(hit: dict[str, Any]) -> str:
    """Stable key for a page-image hit (no chunk id exists for pages)."""
    return f"{hit.get('doc_id')}:p{hit.get('page')}"


def get_reranker(config: Any | None = None, backend: str = "auto") -> Reranker:
    """Resolve a reranker; ``auto`` uses the cross-encoder only if installed."""
    if config is not None:
        backend = backend if backend != "auto" else str(config.get("index.reranker.backend", "auto"))
    if backend in ("none", "off"):
        return LexicalReranker(name="none")
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross-encoder":
        model = str(config.get("index.reranker.model", "BAAI/bge-reranker-base")) if config else \
            "BAAI/bge-reranker-base"
        return CrossEncoderReranker(model_name=model)
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            return CrossEncoderReranker(model_name=str(config.get("index.reranker.model", "BAAI/bge-reranker-base"))
                                        if config else "BAAI/bge-reranker-base")
        except Exception:
            return LexicalReranker()
    raise ValueError(f"unknown reranker backend '{backend}'")
