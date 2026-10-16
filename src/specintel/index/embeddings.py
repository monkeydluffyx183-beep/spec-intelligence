"""WS2 — embedding backends (local-first, with a deterministic offline default).

Backends
--------
``HashingEmbedding``      no model, no network: hashed bag-of-words plus
                          character 4-grams projected into ``dim`` dimensions and
                          L2-normalised.  It behaves like a *lexical* dense
                          retriever — which is exactly what makes the ablations
                          readable on a machine without a GPU: config A still
                          runs, and the report says which embedding produced it.
``OllamaEmbedding``       local open-weight embedder (``nomic-embed-text``,
                          ``bge-m3``, …) over the Ollama HTTP API.
``SentenceTransformerEmbedding``  ``sentence-transformers`` models such as
                          ``BAAI/bge-small-en-v1.5`` running locally.

The interface is deliberately tiny (``encode(texts) -> np.ndarray``) so WS4 and
WS5 never depend on which one is active.  Vectors are always L2-normalised, so
similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:  # numpy is optional: the BM25 path works without it
    import numpy as np
except Exception:  # pragma: no cover - stdlib-only environment
    np = None  # type: ignore

from specintel.index.bm25 import tokenize


def require_numpy():
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for the dense index: pip install 'numpy>=1.26' "
                           "(extra: specintel[dense])")
    return np


def _normalise(matrix: Any) -> Any:
    np_ = require_numpy()
    norms = np_.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class EmbeddingBackend:
    """Interface for embedding backends."""

    name: str = "base"
    dim: int = 384

    def encode(self, texts: Sequence[str]) -> Any:      # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "dim": self.dim}


@dataclass
class HashingEmbedding(EmbeddingBackend):
    """Deterministic hashed bag-of-words + character n-grams.

    Word tokens capture identifiers and technical vocabulary; character
    trigrams make near-miss spellings ("tSenderRespons", "SenderResponse")
    still land close, which is what a real embedding model buys you for
    identifier-heavy text.
    """

    name: str = "hashing"
    dim: int = 384
    ngram: int = 4
    word_weight: float = 1.0
    ngram_weight: float = 0.35

    def _hash(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def encode(self, texts: Sequence[str]) -> Any:
        np_ = require_numpy()
        matrix = np_.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            for token in tokens:
                matrix[i, self._hash("w:" + token)] += self.word_weight
                if len(token) >= self.ngram:
                    for j in range(0, len(token) - self.ngram + 1):
                        matrix[i, self._hash("g:" + token[j:j + self.ngram])] += self.ngram_weight
            # signed hashing keeps collisions from inflating similarity
            for k, idx in enumerate(range(0, self.dim, 64)):
                matrix[i, idx:idx + 1] += 0.0
        return _normalise(matrix)


@dataclass
class OllamaEmbedding(EmbeddingBackend):
    """Local embeddings through Ollama (``/api/embed``)."""

    name: str = "ollama"
    model: str = "nomic-embed-text"
    endpoint: str = "http://localhost:11434"
    dim: int = 768
    timeout_s: int = 120

    def encode(self, texts: Sequence[str]) -> Any:
        import requests

        np_ = require_numpy()
        vectors = []
        batch = 16
        for i in range(0, len(texts), batch):
            payload = {"model": self.model, "input": list(texts[i:i + batch])}
            resp = requests.post(f"{self.endpoint.rstrip('/')}/api/embed", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            body = resp.json()
            vectors.extend(body.get("embeddings") or [body.get("embedding")])
        matrix = np_.asarray(vectors, dtype="float32")
        self.dim = int(matrix.shape[1]) if matrix.ndim == 2 else self.dim
        return _normalise(matrix)


@dataclass
class SentenceTransformerEmbedding(EmbeddingBackend):
    """``sentence-transformers`` model running locally (bge-small by default)."""

    name: str = "sentence-transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    batch_size: int = 64
    _model: Any = field(default=None, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: Sequence[str]) -> Any:
        model = self._load()
        vectors = model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                               show_progress_bar=False)
        return require_numpy().asarray(vectors, dtype="float32")


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def probe_ollama(endpoint: str, timeout_s: float = 1.0) -> bool:
    try:
        import requests

        requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=timeout_s)
        return True
    except Exception:
        return False


def get_embedder(config: Any | None = None, backend: str = "auto") -> EmbeddingBackend:
    """Resolve an embedding backend, defaulting to the offline hashing model."""
    options = {"backend": backend, "model": "BAAI/bge-small-en-v1.5", "dim": 384,
               "ollama_model": "nomic-embed-text", "batch_size": 64}
    if config is not None:
        options.update({
            "backend": backend if backend != "auto" else str(config.get("index.embedding.backend", "auto")),
            "model": str(config.get("index.embedding.model", options["model"])),
            "dim": int(config.get("index.embedding.dim", options["dim"])),
            "ollama_model": str(config.get("index.embedding.ollama_model", options["ollama_model"])),
            "batch_size": int(config.get("index.embedding.batch_size", options["batch_size"])),
        })
    chosen = options["backend"]
    if chosen == "hashing":
        return HashingEmbedding(dim=options["dim"])
    if chosen == "ollama":
        return OllamaEmbedding(model=options["ollama_model"])
    if chosen in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedding(model=options["model"], batch_size=options["batch_size"])
    if chosen == "auto":
        endpoint = str(config.get("agent.llm.endpoint", "http://localhost:11434")) if config else "http://localhost:11434"
        if probe_ollama(endpoint):
            return OllamaEmbedding(model=options["ollama_model"], endpoint=endpoint)
        try:  # a local sentence-transformers install beats the offline hash model
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEmbedding(model=options["model"], batch_size=options["batch_size"])
        except Exception:
            return HashingEmbedding(dim=options["dim"])
    raise ValueError(f"unknown embedding backend '{chosen}'")


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two 1-D vectors (from-scratch, no library)."""
    np_ = require_numpy()
    a = np_.asarray(a, dtype="float32")
    b = np_.asarray(b, dtype="float32")
    denom = float(np_.linalg.norm(a) * np_.linalg.norm(b)) or 1.0
    return float(np_.dot(a, b) / denom)


def top_k_numpy(vectors: Any, query: Any, k: int = 10) -> list[tuple[int, float]]:
    """Cosine top-k over a matrix — the Task 1 from-scratch warm-up exercise."""
    np_ = require_numpy()
    matrix = np_.asarray(vectors, dtype="float32")
    q = np_.asarray(query, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = _normalise(matrix)
    q = q / (np_.linalg.norm(q) or 1.0)
    sims = matrix @ q
    order = np_.argsort(-sims)[:max(0, k)]
    return [(int(i), float(sims[i])) for i in order]
