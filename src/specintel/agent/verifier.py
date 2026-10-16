"""WS4 — citation verification: every sentence must be backed by its evidence.

The verifier runs *after* an answer is drafted and before it is returned:

1. **Locator check** — the citation's document and page must exist in the
   evidence the tools actually returned, and the element id it names must be one
   of the evidence element ids.  A citation to a page the tools never opened is a
   hallucinated locator and is dropped.
2. **Support check** — the sentence's content tokens, identifiers and numeric
   values must be present in the cited evidence.  A sentence whose numbers or
   identifiers do not occur in the cited evidence is flagged as unsupported.
3. **Coverage** — sentences with no citation at all are unsupported by
   construction.

The output is a :class:`VerificationResult` with per-claim verdicts, the kept
answer (unsupported claims removed when ``drop_unsupported=True``) and a citation
accuracy figure for the trace.  The evaluation measures faithfulness independently
(``specintel.eval.metrics.faithfulness``) so a bug here cannot inflate WS5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from specintel.common.models import Citation, SearchHit
from specintel.index.bm25 import identifiers_in, tokenize

CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")
NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|ms|µs|us|ns|V|mV|A|mA|W|kW|kΩ|Ω|°C|bits?|bytes?)?")
SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9\[])")
#: One or more citation markers at the very start of a fragment.
LEADING_MARKERS_RE = re.compile(r"^\s*(?:\[[^\[\]]+?\]\s*)+")
#: A ``Key: value`` segment — how the composer renders one row or one field of a
#: table (``Timer: tSenderResponse; Min: 24; Max: 30; Unit: ms``).
LABELLED_SEGMENT_RE = re.compile(r"^[A-Za-z][\w /()µ%.,\-]{0,40}:")
SEGMENT_SPLIT = re.compile(r";\s*")


def is_labelled_row(text: str) -> bool:
    """Is this block a table row the composer rendered as ``label: value`` pairs?

    Such a block is *one* claim even though it contains several colons and
    semicolons: splitting it on the sentence terminator would drop the numbers
    ("Min: 24; Max: 30") into a fragment of their own, without the citation
    marker that ends the row, and the verifier would then delete them — which is
    how a correct answer to "what are the limits of tSenderResponse?" came back
    as a definition with no numbers in it.
    """
    segments = [seg for seg in SEGMENT_SPLIT.split(str(text)) if seg.strip()]
    labelled = sum(1 for seg in segments if LABELLED_SEGMENT_RE.match(seg.strip()))
    return labelled >= 2


@dataclass
class Claim:
    """One sentence of the answer and the verdict on it."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    locator_ok: bool = False
    support: float = 0.0
    supported: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": [c.to_marker() for c in self.citations],
                "locator_ok": self.locator_ok, "support": round(self.support, 3),
                "supported": self.supported, "reason": self.reason}


@dataclass
class VerificationResult:
    """Verdict over one answer."""

    claims: list[Claim] = field(default_factory=list)
    kept: str = ""
    dropped: list[str] = field(default_factory=list)
    citation_accuracy: float = 0.0
    unsupported_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {"claims": [c.to_dict() for c in self.claims], "kept": self.kept,
                "dropped": self.dropped, "citation_accuracy": round(self.citation_accuracy, 4),
                "unsupported_ratio": round(self.unsupported_ratio, 4), "notes": self.notes}


class CitationVerifier:
    """Check citations of a drafted answer against the evidence that was retrieved."""

    def __init__(self, *, support_threshold: float = 0.55, drop_unsupported: bool = True,
                 require_citation: bool = True) -> None:
        self.support_threshold = support_threshold
        self.drop_unsupported = drop_unsupported
        self.require_citation = require_citation

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def verify(self, answer: str, evidence: Sequence[SearchHit],
               citations: Sequence[Citation] | None = None) -> VerificationResult:
        """Split the answer into claims and check each one."""
        result = VerificationResult()
        # CITATION_RE captures the text *inside* the brackets, so the lookup table
        # has to be keyed the same way — keying it on the bracketed marker meant
        # no sentence ever matched a citation the agent had produced.
        by_marker = {c.to_marker().strip("[]").strip(): c for c in (citations or [])}
        sentences = self.split_claims(answer)
        if not sentences:
            result.notes.append("empty answer")
            return result
        for sentence in sentences:
            claim = Claim(text=sentence)
            markers = CITATION_RE.findall(sentence)
            claim.citations = [by_marker[m.strip()] for m in markers if m.strip() in by_marker]
            if markers and not claim.citations:
                claim.reason = "citation marker does not match any citation the agent produced"
            if not markers and self.require_citation:
                claim.reason = claim.reason or "sentence carries no citation"
            claim.support, supporting_hit = self._support(sentence, evidence)
            claim.locator_ok = self._locator_ok(claim.citations, evidence, supporting_hit)
            claim.supported = (claim.support >= self.support_threshold and claim.locator_ok
                               and (claim.citations or not self.require_citation))
            if claim.supported:
                claim.reason = claim.reason or f"supported by {supporting_hit.locator if supporting_hit else 'evidence'}"
            result.claims.append(claim)

        supported = [c for c in result.claims if c.supported]
        result.citation_accuracy = len(supported) / len(result.claims)
        result.unsupported_ratio = 1.0 - result.citation_accuracy
        result.kept = " ".join(c.text for c in supported) if self.drop_unsupported else answer
        result.dropped = [c.text for c in result.claims if not c.supported]
        if result.dropped and self.drop_unsupported:
            result.notes.append(f"dropped {len(result.dropped)} unsupported claim(s)")
        return result

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_claims(answer: str) -> list[str]:
        """Sentences, keeping table rows whole and markers with their sentence.

        The composer writes the citation marker *after* the sentence it supports
        — "tSenderResponse is 24 ms to 30 ms. [pd, §6.6, p.4, Table 6.8]" — and a
        naive split on sentence endings leaves the marker as a claim of its own
        (with no citation) and the sentence it belongs to without one.  Every
        claim then fails the citation check and the answer is emptied, which is
        exactly how this bug first showed up: the agent declined a question it
        had answered correctly.  Markers are therefore re-attached to the
        sentence they follow, and a fragment that is nothing but markers is not
        a claim at all.
        """
        out: list[str] = []
        for block in str(answer).split("\n"):
            block = block.strip()
            if not block:
                continue
            if (block.startswith("|") or block.startswith("- ") or block.startswith("* ")
                    or is_labelled_row(block)):
                # a table row (pipe form or ``label: value`` form) is one claim
                out.append(block)
                continue
            for piece in (s.strip() for s in SENTENCE_SPLIT.split(block) if s.strip()):
                # The composer writes the marker after the sentence it supports:
                # "…is 24 ms. [pd, §6.6, p.4] The Sink then…".  Splitting on the
                # terminator leaves the marker at the *start* of the next
                # fragment, which strands every claim without a citation and
                # empties correct answers — so markers are handed back to the
                # sentence they belong to, and a fragment that was nothing but
                # markers disappears.
                leading = LEADING_MARKERS_RE.match(piece)
                if leading and out:
                    out[-1] = f"{out[-1].rstrip()} {leading.group(0).strip()}"
                    piece = piece[leading.end():].strip()
                if piece:
                    out.append(piece)
        return out

    def _support(self, sentence: str, evidence: Sequence[SearchHit]) -> tuple[float, SearchHit | None]:
        """How well the best evidence hit supports this sentence (0..1)."""
        terms = {t for t in tokenize(sentence) if len(t) > 2}
        numbers = {n for n in NUMERIC_RE.findall(sentence) if any(ch.isdigit() for ch in n)}
        ids = {i.lower() for i in identifiers_in(sentence)}
        if not terms and not numbers and not ids:
            return 0.0, None
        best, best_hit = 0.0, None
        for hit in evidence:
            hit_terms = set(tokenize(hit.text))
            hit_text = hit.text.lower()
            score = len(terms & hit_terms) / len(terms) if terms else 0.0
            if numbers:
                present = sum(1 for n in numbers if n.lower() in hit_text)
                score = 0.5 * score + 0.5 * (present / len(numbers))
            if ids:
                present_ids = sum(1 for i in ids if i in hit_text or i in {x.lower() for x in hit.element_ids})
                score = 0.7 * score + 0.3 * (present_ids / len(ids))
            if score > best:
                best, best_hit = score, hit
        return min(best, 1.0), best_hit

    @staticmethod
    def _locator_ok(citations: Sequence[Citation], evidence: Sequence[SearchHit],
                    supporting_hit: SearchHit | None) -> bool:
        """Does a citation point at evidence the tools really returned?"""
        if not citations:
            return False
        element_ids = {e for hit in evidence for e in hit.element_ids}
        pages = {(hit.doc_id, hit.page) for hit in evidence}
        for hit in evidence:                                     # a stitched hit covers its page range
            for page in range(hit.page, (hit.page_end or hit.page) + 1):
                pages.add((hit.doc_id, page))
        for citation in citations:
            if (citation.doc_id, citation.page) in pages:
                return True
            if element_ids and set(citation.element_ids or []) & element_ids:
                return True
            if supporting_hit is not None and citation.doc_id == supporting_hit.doc_id \
                    and citation.page == supporting_hit.page:
                return True
        return False


def verify_answer(answer: str, evidence: Sequence[SearchHit],
                  citations: Sequence[Citation] | None = None, **kwargs: Any) -> VerificationResult:
    """Module-level convenience used by the tests and the evaluation."""
    return CitationVerifier(**kwargs).verify(answer, evidence, citations)
