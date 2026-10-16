"""WS2 — BM25 keyword index built from scratch (no search library).

Why BM25 matters for this corpus: the questions are full of exact identifiers
(``tSenderResponse``, ``TEST.PD.PROT.SRC.3``, ``PE_SRC_Ready``, ``Table 6.8``)
that dense retrieval blurs.  Dense search wins on paraphrases; BM25 wins on
identifiers, which is why the assignment asks for both and for an ablation that
shows where each one helps.

Implementation notes
--------------------
* Inverted index ``term -> {doc_index: tf}`` with document lengths, built in one
  pass; ``k1``/``b`` are configurable.
* Identifiers get a boost: a query token that appears verbatim (after case
  folding) in the document scores ``identifier_boost`` times its BM25 term
  contribution.  Camel-case names are additionally indexed as separate parts
  (``tSenderResponse`` → ``tsenderresponse``, ``sender``, ``response``) so a
  question that says "sender response time" still finds the timer.
* Deletion/update is not needed: indexes are rebuilt per run and versioned by
  ``data/index/<doc_id>/bm25.json``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# tokens such as tSenderResponse, PE_SRC_Ready, TEST.PD.PROT.SRC.3, vRd-1.5, 6.8
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.\-][A-Za-z0-9_]+)*|\d+(?:\.\d+)*")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "be", "been",
    "with", "that", "this", "these", "those", "it", "its", "as", "at", "by", "from", "which", "what", "how",
    "when", "where", "why", "does", "do", "did", "can", "could", "should", "shall", "may", "might", "must",
    "there", "their", "them", "they", "then", "than", "also", "into", "than", "so", "if", "not", "no", "any",
    "all", "each", "more", "most", "other", "some", "such", "only", "own", "same", "too", "very", "s", "t",
}


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokenise spec text, preserving identifiers and their parts."""
    out: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        token = raw.lower()
        if not keep_stopwords and token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        out.append(token)
        # camel case / underscore expansion for identifier-heavy questions
        if any(c.isupper() for c in raw) or "_" in raw:
            for part in CAMEL_RE.findall(raw.replace("_", " ")):
                part = part.lower()
                if len(part) > 2 and part not in STOPWORDS and part != token:
                    out.append(part)
    return out


@dataclass
class BM25Index:
    """A BM25 index over arbitrary string ids with metadata."""

    k1: float = 1.5
    b: float = 0.75
    identifier_boost: float = 3.0
    ids: list[str] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)
    postings: dict[str, dict[int, int]] = field(default_factory=dict)   # term -> {doc: tf}
    lengths: list[int] = field(default_factory=list)
    _avgdl: float = 0.0

    # -- build ------------------------------------------------------------- #
    def add(self, key: str, text: str, **metadata: Any) -> None:
        """Add one document; ``key`` is its id, ``metadata`` is stored verbatim.

        The parameter is named ``key`` rather than ``doc_id`` on purpose: the
        metadata dict commonly *contains* a ``doc_id`` field, which would
        otherwise collide with the positional argument.
        """
        idx = len(self.ids)
        self.ids.append(key)
        self.meta.append(metadata)
        tokens = tokenize(text)
        self.lengths.append(len(tokens) or 1)
        for term, tf in Counter(tokens).items():
            self.postings.setdefault(term, {})[idx] = tf
        self._avgdl = 0.0

    def extend(self, items: Iterable[tuple[str, str, dict[str, Any]]]) -> None:
        for key, text, metadata in items:
            self.add(key, text, **metadata)

    @property
    def avgdl(self) -> float:
        if self._avgdl == 0.0 and self.lengths:
            self._avgdl = sum(self.lengths) / len(self.lengths)
        return self._avgdl or 1.0

    def __len__(self) -> int:
        return len(self.ids)

    # -- search ------------------------------------------------------------ #
    def score(self, query: str, doc_filter: Any | None = None) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs sorted by descending score."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: dict[int, float] = {}
        n_docs = max(len(self.ids), 1)
        for term in dict.fromkeys(q_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings.items():
                if doc_filter is not None and not doc_filter(doc_idx):
                    continue
                dl = self.lengths[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                contribution = idf * (tf * (self.k1 + 1)) / denom
                exact = self.meta[doc_idx].get("exact_terms") or set()
                if term in exact or term in (self.meta[doc_idx].get("identifiers") or set()):
                    contribution *= self.identifier_boost
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def search(self, query: str, k: int = 10, doc_filter: Any | None = None) -> list[dict[str, Any]]:
        """Return hit dicts: ``{chunk_id, score, meta}``."""
        out: list[dict[str, Any]] = []
        for doc_idx, score in self.score(query, doc_filter=doc_filter)[:max(0, k)]:
            out.append({"chunk_id": self.ids[doc_idx], "score": score, "meta": self.meta[doc_idx]})
        return out

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b, "identifier_boost": self.identifier_boost, "ids": self.ids,
                "meta": self.meta, "postings": self.postings, "lengths": self.lengths}

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "BM25Index":
        index = BM25Index(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75),
                          identifier_boost=payload.get("identifier_boost", 3.0))
        index.ids = list(payload.get("ids", []))
        index.meta = [dict(m) if not isinstance(m, (str, int, float)) else {"value": m}
                      for m in payload.get("meta", [])]
        for m in index.meta:                       # sets are not JSON
            for key in ("exact_terms", "identifiers"):
                if isinstance(m.get(key), list):
                    m[key] = set(m[key])
        index.postings = {t: {int(k): int(v) for k, v in p.items()}
                          for t, p in payload.get("postings", {}).items()}
        index.lengths = list(payload.get("lengths", []))
        return index

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        for m in payload["meta"]:
            for key in ("exact_terms", "identifiers"):
                if isinstance(m.get(key), set):
                    m[key] = sorted(m[key])
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        return BM25Index.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_bm25(chunks: Sequence[Any], *, k1: float = 1.5, b: float = 0.75,
               identifier_boost: float = 3.0) -> BM25Index:
    """Build an index over :class:`~specintel.common.models.Chunk` objects."""
    index = BM25Index(k1=k1, b=b, identifier_boost=identifier_boost)
    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        meta = {
            "doc_id": getattr(chunk, "doc_id", ""),
            "revision": getattr(chunk, "revision", ""),
            "page": getattr(chunk, "page", 0),
            "page_end": getattr(chunk, "page_end", 0),
            "section_number": getattr(chunk, "section_number", ""),
            "element_type": getattr(chunk, "element_type", ""),
            "table_id": getattr(chunk, "table_id", None),
            "figure_id": getattr(chunk, "figure_id", None),
            "parent_chunk_id": getattr(chunk, "parent_chunk_id", None),
            "identifiers": set(_identifiers_in(text)),
        }
        index.add(getattr(chunk, "chunk_id", f"c{len(index)}"), text, **meta)
    return index


IDENTIFIER_RE = re.compile(
    r"\b(?:t[A-Z][A-Za-z0-9]+|n[A-Z][A-Za-z0-9]+|[A-Z]{2,}[A-Za-z_]*(?:_[A-Za-z0-9]+)*|"
    r"[A-Z][A-Za-z]+\.(?:SRC|SNK|SRCACC|ACC|DBGACC|AUDIOACC)|"
    r"(?:TEST\.[A-Z0-9.]+|TD\.[0-9.]+(?:\.[A-Z0-9.]+)*|COMMON\.[A-Z0-9.]+|"
    r"[A-Z]{2,}\.[A-Z]{2,}\.[A-Z0-9.]+)|vRd-[0-9.]+)\b"
)


def identifiers_in(text: str) -> list[str]:
    """Extract spec identifiers from text (test IDs, timers, states, params)."""
    return _identifiers_in(text)


def _identifiers_in(text: str) -> set[str]:
    return {m.lower() for m in IDENTIFIER_RE.findall(text or "")}


def explain_score(index: BM25Index, query: str, chunk_id: str) -> dict[str, float]:
    """Per-term contribution breakdown for one document (used in LOGS/traces)."""
    if chunk_id not in index.ids:
        return {}
    doc_idx = index.ids.index(chunk_id)
    dl = index.lengths[doc_idx] or 1
    n_docs = max(len(index.ids), 1)
    out: dict[str, float] = {}
    for term in dict.fromkeys(tokenize(query)):
        postings = index.postings.get(term)
        if not postings or doc_idx not in postings:
            continue
        tf = postings[doc_idx]
        idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
        contribution = idf * (tf * (index.k1 + 1)) / (tf + index.k1 * (1 - index.b + index.b * dl / index.avgdl))
        if term in (index.meta[doc_idx].get("identifiers") or set()):
            contribution *= index.identifier_boost
        out[term] = round(contribution, 4)
    return out
