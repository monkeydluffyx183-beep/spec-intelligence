"""WS2 — dense vector index.

A numpy brute-force cosine index is the default: for a few thousand chunks it is
faster than a graph index *and* exact, which matters when the evaluation has to
be reproducible.  The same class persists to ``.npz`` + ``.json``, and
``ANNIndex`` wraps FAISS when it is installed for the full-corpus runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from specintel.index.embeddings import EmbeddingBackend, _normalise, require_numpy


@dataclass
class VectorIndex:
    """Exact cosine index over chunk embeddings."""

    dim: int = 384
    ids: list[str] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)
    matrix: Any = None                     # np.ndarray (n, dim), L2 normalised
    backend_name: str = "unknown"

    # -- build ------------------------------------------------------------- #
    def build(self, ids: Sequence[str], texts: Sequence[str], embedder: EmbeddingBackend,
              meta: Sequence[dict[str, Any]] | None = None) -> "VectorIndex":
        np_ = require_numpy()
        vectors = embedder.encode(list(texts))
        self.matrix = _normalise(np_.asarray(vectors, dtype="float32"))
        self.dim = int(self.matrix.shape[1]) if self.matrix.ndim == 2 else self.dim
        self.ids = list(ids)
        self.meta = [dict(m) for m in (meta or [{} for _ in ids])]
        self.backend_name = embedder.name
        return self

    def add(self, ids: Sequence[str], vectors: Any, meta: Sequence[dict[str, Any]] | None = None) -> None:
        np_ = require_numpy()
        block = _normalise(np_.asarray(vectors, dtype="float32"))
        self.matrix = block if self.matrix is None else np_.vstack([self.matrix, block])
        self.ids.extend(ids)
        self.meta.extend(dict(m) for m in (meta or [{} for _ in ids]))

    def __len__(self) -> int:
        return len(self.ids)

    # -- search ------------------------------------------------------------ #
    def search_vectors(self, query_vector: Any, k: int = 10,
                       doc_filter: Callable[[int], bool] | None = None) -> list[tuple[int, float]]:
        np_ = require_numpy()
        if self.matrix is None or not len(self.ids):
            return []
        q = np_.asarray(query_vector, dtype="float32").reshape(-1)
        q = q / (np_.linalg.norm(q) or 1.0)
        sims = self.matrix @ q
        if doc_filter is not None:
            mask = np_.array([doc_filter(i) for i in range(len(self.ids))], dtype=bool)
            sims = np_.where(mask, sims, -1e9)
        k = min(k, len(self.ids))
        order = np_.argpartition(-sims, k - 1)[:k] if k > 0 else []
        order = sorted(order, key=lambda i: -sims[int(i)])
        return [(int(i), float(sims[int(i)])) for i in order]

    def search(self, query: str, embedder: EmbeddingBackend, k: int = 10,
               doc_filter: Callable[[int], bool] | None = None) -> list[dict[str, Any]]:
        vector = embedder.encode([query])[0]
        out: list[dict[str, Any]] = []
        for idx, score in self.search_vectors(vector, k=k, doc_filter=doc_filter):
            out.append({"chunk_id": self.ids[idx], "score": score, "meta": self.meta[idx]})
        return out

    # -- persistence ------------------------------------------------------- #
    def save(self, directory: str | Path, name: str = "dense") -> Path:
        np_ = require_numpy()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np_.save(d / f"{name}.npy", self.matrix if self.matrix is not None else np_.zeros((0, self.dim)))
        (d / f"{name}.json").write_text(json.dumps({
            "ids": self.ids, "meta": self.meta, "dim": self.dim, "backend": self.backend_name,
        }), encoding="utf-8")
        return d / f"{name}.npy"

    @staticmethod
    def load(directory: str | Path, name: str = "dense") -> "VectorIndex":
        np_ = require_numpy()
        d = Path(directory)
        payload = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
        index = VectorIndex(dim=int(payload.get("dim", 384)), ids=list(payload.get("ids", [])),
                            meta=list(payload.get("meta", [])), backend_name=payload.get("backend", "unknown"))
        matrix = np_.load(d / f"{name}.npy")
        index.matrix = matrix if matrix.size else None
        return index


class ANNIndex(VectorIndex):
    """FAISS-backed approximate index, used when the corpus outgrows NumPy.

    The interface is identical to :class:`VectorIndex`, so the ablation runner
    can switch backends with one config value; the evaluation report records
    which one produced the numbers.
    """

    backend_name = "faiss"

    def __init__(self, dim: int = 384, nlist: int = 64) -> None:
        super().__init__(dim=dim)
        self.nlist = nlist
        self._faiss_index: Any = None

    def build(self, ids: Sequence[str], texts: Sequence[str], embedder: EmbeddingBackend,
              meta: Sequence[dict[str, Any]] | None = None) -> "ANNIndex":
        import faiss  # type: ignore

        super().build(ids, texts, embedder, meta)
        self._faiss_index = faiss.IndexFlatIP(self.dim)
        self._faiss_index.add(self.matrix)
        return self

    def search_vectors(self, query_vector: Any, k: int = 10,
                       doc_filter: Callable[[int], bool] | None = None) -> list[tuple[int, float]]:
        np_ = require_numpy()
        if self._faiss_index is None or self.matrix is None:
            return super().search_vectors(query_vector, k=k, doc_filter=doc_filter)
        q = np_.asarray(query_vector, dtype="float32").reshape(1, -1)
        q = q / (np_.linalg.norm(q) or 1.0)
        scores, idxs = self._faiss_index.search(q, min(max(k * 4, k), len(self.ids)))
        out = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        if doc_filter is not None:
            out = [(i, s) for i, s in out if doc_filter(i)]
        return out[:k]
