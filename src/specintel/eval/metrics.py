"""WS5 — metric implementations (pure functions, so they are unit-testable).

Every metric the assignment asks for lives here:

======================  =====================================================
metric                  definition
======================  =====================================================
recall@k                gold evidence (doc, page) present in the top-k hits
mrr / ndcg@k            rank quality of the first gold evidence hit
answer correctness      key-point recall, key-point precision and token F1
faithfulness            share of answer claims supported by retrieved evidence
citation accuracy       share of citations that match a gold evidence locator
refusal accuracy        balanced accuracy of declining when the corpus cannot
                        answer and answering when it can
======================  =====================================================

The key-point metrics are the machine-checkable part of "correctness": a question
implicitly proposes the numbers, identifiers and named facts that a correct answer
must contain, and the gold answer is where those come from.  Human or model
judging is the tie-breaker for wording, and the harness keeps the raw answers in
``eval/results/answers_<config>.jsonl`` so a reviewer can grade them without
re-running anything.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.index.bm25 import identifiers_in, tokenize

# numbers *with* their unit are strong key points ("24 ms", "30 ms", "0x12")
VALUE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|ms|µs|us|ns|s|V|mV|A|mA|W|kW|kΩ|Ω|°C|bytes?|bits?|b)?")
IDENT_RE = re.compile(r"\b(?:TEST\.[A-Z0-9.#]+|TD\s?\d+(?:\.\d+)+|\[?[A-Z]{2,}[._][A-Za-z0-9._]+\]?|"
                      r"[A-Z]{2,}_[A-Za-z0-9_]+|PE_[A-Za-z_]+|PE\.[A-Za-z.]+|[A-Za-z]+\.(?:SRC|SNK|IDX)[A-Za-z0-9.]*|"
                      r"vRd-[0-9.]+|t[A-Z][A-Za-z]+|n[A-Z][A-Za-z]+|B\d+(?:\.\.\d+)?|0x[0-9A-F]{2})\b")

STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "with", "that",
        "this", "it", "as", "at", "by", "from", "which", "what", "when", "where", "how", "does", "do",
        "shall", "must", "may", "can", "not", "no", "its", "their", "they", "them", "than", "then", "into",
        "each", "also", "only", "both", "after", "before", "during", "other", "same", "such", "has", "have",
        "was", "were", "will", "would", "should", "been", "more", "most", "less", "least", "one", "two",
        "all", "any", "but", "if", "so", "up", "out", "over", "under", "between", "within", "without"}


# --------------------------------------------------------------------------- #
# retrieval quality
# --------------------------------------------------------------------------- #
def hit_matches_evidence(hit: Any, evidence: dict[str, Any], *, page_tolerance: int = 0) -> bool:
    """Does a retrieved hit contain the gold evidence locator?"""
    if str(getattr(hit, "doc_id", "")) != str(evidence.get("doc_id", "")):
        return False
    gold_page = int(evidence.get("page") or 0)
    pages = {int(getattr(hit, "page", 0) or 0)}
    end = getattr(hit, "page_end", None)
    if end:
        pages |= set(range(int(getattr(hit, "page", 0)), int(end) + 1))
    if gold_page and any(abs(p - gold_page) <= page_tolerance for p in pages):
        return True
    element_id = evidence.get("element_id")
    if element_id and element_id in (getattr(hit, "element_ids", None) or []):
        return True
    return False


def recall_at_k(hits: Sequence[Any], evidence: Sequence[dict[str, Any]], k: int,
                *, page_tolerance: int = 0) -> float:
    """1.0 when any gold evidence locator appears in the first ``k`` hits."""
    if not evidence:
        return float("nan")
    top = list(hits)[:k]
    return 1.0 if any(hit_matches_evidence(h, e, page_tolerance=page_tolerance)
                      for h in top for e in evidence) else 0.0


def reciprocal_rank(hits: Sequence[Any], evidence: Sequence[dict[str, Any]], *, page_tolerance: int = 0) -> float:
    for i, hit in enumerate(hits, start=1):
        if any(hit_matches_evidence(hit, e, page_tolerance=page_tolerance) for e in evidence):
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: Sequence[Any], evidence: Sequence[dict[str, Any]], k: int,
              *, page_tolerance: int = 0) -> float:
    """Binary-gain nDCG: gain 1 for the first hit that matches any gold evidence."""
    top = list(hits)[:k]
    dcg = 0.0
    for i, hit in enumerate(top, start=1):
        if any(hit_matches_evidence(hit, e, page_tolerance=page_tolerance) for e in evidence):
            dcg = 1.0 / math.log2(i + 1)
            break
    ideal = 1.0 / math.log2(2.0)
    return dcg / ideal


# --------------------------------------------------------------------------- #
# answer quality
# --------------------------------------------------------------------------- #
def key_points(text: str) -> set[str]:
    """The checkable facts in a text: numbers with units and identifiers."""
    values = {v.strip().lower().replace(",", "") for v in VALUE_RE.findall(text) if any(ch.isdigit() for ch in v)}
    ids = {i.strip("[]").lower() for i in identifiers_in(text)}
    ids |= {i.lower().strip("[]") for i in IDENT_RE.findall(text)}
    ids = {_normalise_identifier(i) for i in ids}
    points = values | ids
    return {p for p in points if len(p) >= 2 and p not in {"1", "2", "3", "0"}}


def _normalise_identifier(token: str) -> str:
    token = token.strip("[]().;:,").lower()
    token = re.sub(r"^(the|a)\s+", "", token)
    token = token.replace("specification", "spec").replace("release", "r")
    return token


def key_point_recall(prediction: str, gold: str) -> float:
    """Share of the gold answer's checkable facts that the prediction contains."""
    gold_points = key_points(gold)
    if not gold_points:
        return float("nan")
    pred = prediction.lower()
    pred_norm = {_normalise_identifier(t) for t in identifiers_in(prediction)}
    pred_norm |= {v.strip().lower().replace(",", "") for v in VALUE_RE.findall(prediction)}
    hits = 0
    for point in gold_points:
        if point in pred or point in pred_norm or any(point in p or p in point for p in pred_norm):
            hits += 1
    return hits / len(gold_points)


def key_point_precision(prediction: str, gold: str) -> float:
    """Share of the prediction's checkable facts that the gold answer also states."""
    pred_points = key_points(prediction)
    if not pred_points:
        return float("nan")
    gold_points = key_points(gold)
    gold_norm = {_normalise_identifier(t) for t in identifiers_in(gold)}
    gold_norm |= {v.strip().lower().replace(",", "") for v in VALUE_RE.findall(gold)}
    hits = 0
    for point in pred_points:
        if point in gold_points or point in gold_norm or any(point in g or g in point for g in gold_norm):
            hits += 1
    return hits / len(pred_points)


def token_f1(prediction: str, gold: str) -> float:
    """Content-token F1 (SQuAD-style, stop words removed)."""
    pred_tokens = [t for t in tokenize(prediction) if t not in STOP]
    gold_tokens = [t for t in tokenize(gold) if t not in STOP]
    if not pred_tokens or not gold_tokens:
        return float("nan")
    from collections import Counter

    pred_counts, gold_counts = Counter(pred_tokens), Counter(gold_tokens)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def key_point_precision_vs_context(prediction: str, support: str) -> float:
    """Share of the prediction's facts that the gold answer *or its evidence* states.

    Scored against the evidence as well as the gold answer on purpose: a correct
    answer may state a fact the gold answer left implicit ("24 ms" when the gold
    answer says "the limits of Table 6.8"), and that must not count as an error.
    """
    pred_points = key_points(prediction)
    if not pred_points:
        return float("nan")
    support_low = support.lower()
    support_norm = {_normalise_identifier(t) for t in identifiers_in(support)}
    support_norm |= {v.strip().lower().replace(",", "") for v in VALUE_RE.findall(support)}
    hits = 0
    for point in pred_points:
        if point in support_low or point in support_norm or any(point in sp or sp in point for sp in support_norm):
            hits += 1
    return hits / len(pred_points)


def content_recall(prediction: str, gold: str) -> float:
    """Share of the gold answer's content words that the prediction repeats.

    Used only for gold answers that carry no checkable facts at all (pure prose,
    e.g. "the test is run when the charger advertises a shared capacity"): there
    is nothing to compare as a set, so the score falls back to word overlap and
    the fallback is visible in the per-question JSON.
    """
    gold_terms = [t for t in tokenize(gold) if len(t) > 3]
    if not gold_terms:
        return float("nan")
    pred = set(tokenize(prediction))
    unique = list(dict.fromkeys(gold_terms))
    return sum(1 for t in unique if t in pred) / len(unique)


def answer_correctness(prediction: str, gold: str, *, decline_expected: bool = False,
                       gold_context: str = "", used_context: str = "") -> dict[str, float]:
    """Correctness = F1 over *facts* (numbers, identifiers, named objects).

    Wording is not correctness: an extractive answer that states the right timer
    limits in one line is correct even though it shares few tokens with the gold
    prose.  The facts are therefore scored as a set and correctness is their F1:

    * ``key_point_recall`` — are the gold answer's checkable facts all present?
    * ``key_point_precision`` — could those facts have come from anything the
      answer was allowed to use: the gold answer, its evidence, or the context
      the system actually retrieved for this question?  A fact no source
      supports is an invention and counts against the answer; a true spec fact
      the gold answer happened not to repeat does not.

    Verbosity is a separate axis.  ``key_point_precision_strict`` measures the
    facts that are on-topic for the gold answer alone, and ``answer_tokens``
    reports length, so an answer that states four true facts where one was asked
    for is visibly less focused without being called incorrect.

    Gold answers that carry no checkable facts at all (pure prose) fall back to
    ``content_recall`` — word overlap — and the fallback is reported per
    question so nobody mistakes it for a fact-level score.

    For questions the corpus cannot answer, correctness is undefined and the
    honest decline is scored by :func:`refusal_scores` instead.
    """
    if decline_expected:
        return {"key_point_recall": float("nan"), "key_point_precision": float("nan"),
                "key_point_precision_strict": float("nan"), "prose_recall": float("nan"),
                "token_f1": float("nan"), "correctness": float("nan"), "answer_tokens": 0.0}
    support = f"{gold}\n{gold_context}"
    grounded = f"{support}\n{used_context}"
    recall = key_point_recall(prediction, gold)
    prose = content_recall(prediction, gold)
    if math.isnan(recall) and not math.isnan(prose):
        recall = prose
    precision = key_point_precision_vs_context(prediction, grounded)
    strict = key_point_precision_vs_context(prediction, support)
    f1 = token_f1(prediction, gold)
    if math.isnan(recall) or math.isnan(precision):
        correctness = 0.0 if math.isnan(f1) else f1
    else:
        correctness = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
    return {"key_point_recall": recall, "key_point_precision": precision,
            "key_point_precision_strict": strict, "prose_recall": prose,
            "token_f1": f1, "correctness": correctness,
            "answer_tokens": float(len(tokenize(prediction)))}


def faithfulness(answer: str, evidence: Sequence[Any], *, min_overlap: float = 0.34) -> float:
    """Share of the answer's factual sentences that the retrieved evidence supports.

    This is deliberately independent of the agent's own verifier: it re-derives
    support from the retrieved hits so a bug in the verifier cannot inflate the
    metric.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", answer) if len(s.strip()) > 25]
    if not sentences:
        return float("nan")
    supported = 0
    for sentence in sentences:
        terms = {t for t in tokenize(sentence) if t not in STOP and len(t) > 2}
        if not terms:
            supported += 1
            continue
        best = 0.0
        for hit in evidence:
            hit_terms = set(tokenize(getattr(hit, "text", "")))
            best = max(best, len(terms & hit_terms) / len(terms))
        if best >= min_overlap:
            supported += 1
    return supported / len(sentences)


# --------------------------------------------------------------------------- #
# citations and refusals
# --------------------------------------------------------------------------- #
def citation_scores(citations: Sequence[Any], evidence: Sequence[dict[str, Any]],
                    *, page_tolerance: int = 0) -> dict[str, float]:
    """Citation accuracy: precision/recall of citations against the gold locators."""
    if not citations:
        return {"citation_precision": 0.0, "citation_recall": 0.0,
                "citation_accuracy": 0.0 if evidence else float("nan")}
    from specintel.common.models import Citation

    matched_citations = 0
    for citation in citations:
        c = citation if isinstance(citation, Citation) else Citation.from_dict(citation)
        if any(_citation_matches(c, e, page_tolerance) for e in evidence):
            matched_citations += 1
    covered_evidence = 0
    for e in evidence:
        if any(_citation_matches(citation if isinstance(citation, Citation) else Citation.from_dict(citation),
                                 e, page_tolerance) for citation in citations):
            covered_evidence += 1
    precision = matched_citations / len(citations)
    recall = covered_evidence / len(evidence) if evidence else float("nan")
    return {"citation_precision": precision, "citation_recall": recall,
            "citation_accuracy": precision}


REGION_RE = re.compile(r"\b(Table|Figure)\s+([A-Za-z0-9.\-]+)", re.I)


def _citation_matches(citation: Any, evidence: dict[str, Any], page_tolerance: int = 0) -> bool:
    """Does this citation point at the gold evidence?

    Three ways to match, in decreasing specificity: the same document *and* page,
    the same element id, or the same named object — a citation to "Table 6.8" on
    page 3 of a table whose rows continue on page 4 is the same evidence, and the
    gold evidence records the page the table starts on.
    """
    if str(getattr(citation, "doc_id", "")) != str(evidence.get("doc_id", "")):
        return False
    gold_page = int(evidence.get("page") or 0)
    page = int(getattr(citation, "page", 0) or 0)
    if gold_page and page and abs(page - gold_page) <= page_tolerance:
        return True
    element_id = evidence.get("element_id")
    if element_id and element_id in (getattr(citation, "element_ids", None) or []):
        return True
    citation_region = REGION_RE.search(str(getattr(citation, "region", "")) or "")
    gold_region = REGION_RE.search(str(evidence.get("region", "")) or "")
    if citation_region and gold_region:
        return (citation_region.group(1).lower() == gold_region.group(1).lower()
                and citation_region.group(2).strip(" .") == gold_region.group(2).strip(" ."))
    return False


def refusal_scores(declined: bool, answerable: bool) -> dict[str, float]:
    """Refusal accuracy: correct decline on unanswerable, correct answer otherwise."""
    correct_decline = declined and not answerable
    missed_decline = (not declined) and (not answerable)
    correct_answer = (not declined) and answerable
    over_refusal = declined and answerable
    return {"correct_decline": float(correct_decline), "false_answer": float(missed_decline),
            "correct_answer": float(correct_answer), "over_refusal": float(over_refusal),
            "refusal_correct": float(correct_decline or correct_answer)}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Aggregate:
    """Averaged metrics with per-key counts."""

    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, value: float | None) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        total, count = self.values.get(key, 0.0), self.counts.get(key, 0)
        self.values[key] = total + value
        self.counts[key] = count + 1

    def add_many(self, mapping: dict[str, float | None]) -> None:
        for key, value in mapping.items():
            self.add(key, value)

    def mean(self, key: str) -> float | None:
        count = self.counts.get(key, 0)
        return self.values[key] / count if count else None

    def summary(self) -> dict[str, float]:
        return {k: round(self.values[k] / self.counts[k], 4) for k in self.values if self.counts[k]}

    def merge(self, other: "Aggregate") -> None:
        for key, total in other.values.items():
            self.values[key] = self.values.get(key, 0.0) + total
            self.counts[key] = self.counts.get(key, 0) + other.counts.get(key, 0)


def percentile(values: Iterable[float], p: float) -> float:
    """Simple percentile (linear interpolation) for latency reporting."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return float("nan")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[int(position)]
    return data[lower] + (data[upper] - data[lower]) * (position - lower)
