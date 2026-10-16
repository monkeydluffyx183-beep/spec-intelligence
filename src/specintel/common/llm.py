"""Shared LLM layer: local-first generation with a deterministic offline backend.

Backends
--------
``ExtractiveLLM``       no model: composes an answer by selecting the sentences
                        of the retrieved evidence that best answer the question,
                        always attached to their source.  It is the default on a
                        machine without a local model, and it keeps the
                        pipeline honest: every sentence in the answer comes from
                        the corpus, never from model memory.  The evaluation
                        report records ``model=extractive`` so nobody mistakes
                        it for a generative result.
``OllamaLLM``           local open-weight model (``qwen2.5:7b-instruct`` …),
                        deterministic settings by default for evaluation.
``OpenAICompatibleLLM`` any OpenAI-compatible endpoint (vLLM, SGLang, LM Studio,
                        or a documented hosted API — the assignment allows one if
                        the cost per question is reported).

All backends return :class:`LLMResponse` with token accounting, so the agent can
enforce its token budget and WS5 can report tokens per question.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from specintel.index.bm25 import identifiers_in, tokenize


@dataclass
class LLMResponse:
    text: str
    model: str = "extractive"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    backend: str = "extractive"
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "backend": self.backend, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_s": round(self.latency_s, 3),
                "error": self.error}


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, len(text.split()))


# --------------------------------------------------------------------------- #
# deterministic backend
# --------------------------------------------------------------------------- #
SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")
CLAIM_HINT = re.compile(r"\b(shall|must|should|is|are|sends?|starts?|runs?|enters?|means|equals?)\b", re.I)


class ExtractiveLLM:
    """Deterministic answer composer over retrieved evidence.

    The question's content terms are scored against each sentence of the
    evidence; the best sentences (in evidence order) become the answer, and each
    sentence keeps the citation of the evidence it came from.  Numbers, timer
    names and identifiers are treated as high-value terms, so "24 ms to 30 ms"
    or "PE_SRC_Negotiate_Capability" survive into the answer.
    """

    name = "extractive"
    backend = "extractive"

    def __init__(self, max_sentences: int = 4, min_score: float = 0.12,
                 per_item_limit: int = 2, focus_ratio: float = 0.55) -> None:
        self.max_sentences = max_sentences
        self.min_score = min_score
        #: a unit is only added while it scores at least this fraction of the best
        #: unit; past that elbow the units come from neighbouring sections and
        #: their facts are about something else, which reads as an unfocused
        #: answer even when every fact in it is true
        self.focus_ratio = focus_ratio
        #: how many units one piece of evidence may contribute; a table row is a
        #: unit, so a table that answers the question contributes several
        self.per_item_limit = per_item_limit

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 700,
                 evidence: Sequence[dict[str, Any]] | None = None, question: str = "") -> LLMResponse:
        """``evidence`` is a list of ``{text, citation}``; ``question`` drives selection.

        Evidence is split into *units*: table rows become "column: value" clauses
        (one unit per row) and prose becomes sentences.  Units are scored against
        the question, the best ones are kept in evidence order and joined — so
        the answer quotes the row or sentence that answers the question instead
        of reproducing a whole table.
        """
        t0 = time.perf_counter()
        q_terms = set(tokenize(question or prompt))
        q_ids = identifiers_in(question or prompt)
        q_numbers = set(_NUM_RE.findall(question or prompt))
        scored: list[tuple[float, int, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for i, item in enumerate(evidence or []):
            rank_bonus = max(0.0, 0.12 - 0.01 * i)      # earlier evidence is slightly preferred
            for unit in _units(str(item.get("text", ""))):
                key = unit.strip().lower()
                if len(unit) < 25 or key in seen:
                    continue
                seen.add(key)
                score = _sentence_score(unit, q_terms, q_ids, q_numbers) + rank_bonus
                if score >= self.min_score:
                    scored.append((score, i, unit.strip(), item))
        scored.sort(key=lambda t: (-t[0], t[1]))
        # drop units that merely mention a word of the question (see _is_relevant);
        # if the gate would empty the answer, keep the unfiltered ranking
        relevant = [c for c in scored if _is_relevant(c[2], q_terms, q_ids)]
        if relevant:
            scored = relevant
        # when the question names an identifier (tSenderResponse, PS_RDY, 0x12 …),
        # units that contain it are the answer; units that do not are neighbours
        # of the answer and only fill in when the answer is short
        if q_ids:
            with_id = [c for c in scored if q_ids & identifiers_in(c[2])]
            if with_id:
                # the units that name the identifier are the answer; other units
                # from the *same* piece of evidence are its neighbours (the next
                # row of the same table), so they are only used if the answer
                # would otherwise be a single sentence
                others = [c for c in scored if not (q_ids & identifiers_in(c[2]))
                          and c[1] != with_id[0][1]]
                scored = with_id + others if len(with_id) == 1 else with_id + others[:1]
        if scored:
            elbow = max(self.min_score, self.focus_ratio * scored[0][0])
            focused = [c for c in scored if c[0] >= elbow]
            if focused:
                scored = focused
        chosen = _select_diverse(scored, self.max_sentences, self.per_item_limit)
        chosen.sort(key=lambda t: t[1])
        answer = " ".join(self._with_citation(sentence, item) for _, _, sentence, item in chosen)
        if not answer.strip() and scored:
            _, i, sentence, item = scored[0]
            answer = self._with_citation(sentence, item)
        return LLMResponse(text=answer.strip(), model=self.name, backend=self.backend,
                           prompt_tokens=estimate_tokens(prompt + system + str(evidence)),
                           completion_tokens=estimate_tokens(answer),
                           latency_s=time.perf_counter() - t0)

    @staticmethod
    def _with_citation(sentence: str, item: dict[str, Any]) -> str:
        citation = item.get("citation") or item.get("marker") or ""
        return f"{sentence} {citation}".strip() if citation and citation not in sentence else sentence

    def complete_json(self, prompt: str, *, schema_hint: str = "", **kwargs: Any) -> tuple[dict[str, Any], LLMResponse]:
        resp = self.complete(prompt, **kwargs)
        return {"raw": resp.text}, resp


_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|V|A|W|%|kΩ|Ω)?")


BREADCRUMB_RE = re.compile(r"^(?:[^>\n]{1,60}>\s*){1,4}")
CAPTION_ONLY_RE = re.compile(r"^(?:Table|Figure)\s+[\w.\-]+\s*[.:]?\s*(?:[A-Z][^.]{0,80})?$")


def _strip_decorations(line: str) -> str:
    """Drop section breadcrumbs and bare caption lines.

    Chunks carry their heading path ("6 Protocol Layer Messages > 6.3 Control
    Messages") and their caption; both help a reader and hurt an answer, because
    they read as sentences.  The breadcrumb prefix is removed and a line that is
    nothing but a caption is dropped.
    """
    line = line.strip()
    if ">" in line and line.count(">") <= 4 and len(line.split()) <= 24:
        stripped = BREADCRUMB_RE.sub("", line).strip()
        if len(stripped) > 20:
            line = stripped
        else:
            return ""
    if CAPTION_ONLY_RE.match(line):
        return ""
    return line


#: front matter and legal/companion-document sentences.  They rank highly for
#: almost any question (they name the document and the topic) and they answer
#: nothing, so they are removed before the answer is composed.
BOILERPLATE_RE = re.compile(
    r"(all rights reserved|copyright\s*©|licensed material|do not redistribute"
    r"|adopters agreement|evaluation agreement|is a companion document"
    r"|companion document to|this document (?:defines|describes|covers|specifies)"
    r"|this specification (?:defines|describes|covers|specifies)"
    r"|this test specification (?:defines|describes|covers|specifies)"
    r"|this appendix (?:records|lists|documents) the changes"
    r"|the reader is referred to|is published by the)", re.I)


def _is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_RE.search(text))


TITLE_LINE_RE = re.compile(r"^(?:Table|Figure)\s+[\w.\-]+\s*[.:]?\s*(.*)$")


def _table_caption(line: str) -> str:
    """The caption line of a table/image chunk, if this line is one."""
    match = TITLE_LINE_RE.match(line.strip())
    if not match:
        return ""
    tail = match.group(1).strip(" .")
    return f"Table {line.split()[1]} {tail}".strip() if line.strip().lower().startswith("table") \
        else f"Figure {line.split()[1]} {tail}".strip()


def _is_heading_like(text: str) -> bool:
    """A bare heading ("USB Power Delivery Specification") answers nothing.

    Chunks carry their heading path, and the document title is the first chunk of
    every document, so both rank highly for almost any question.  A line with no
    sentence ending, no digits and no more than nine words is a heading: it is
    kept as context in the index but not offered as an answer sentence.
    """
    stripped = text.strip()
    if len(stripped) > 90 or not stripped:
        return False
    if re.search(r"[0-9]", stripped) or re.search(r"[.;:!?]\s|[,;:]$", stripped):
        return False
    return len(stripped.split()) <= 9


def _units(text: str) -> list[str]:
    """Split evidence into answer-sized units: table rows or sentences."""
    if "|" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header: list[str] = []
        out: list[str] = []
        caption = ""
        for line in lines:
            if not line.startswith("|"):
                found = _table_caption(line)
                if found and not caption:
                    caption = found
                cleaned = _strip_decorations(line)
                if cleaned and not _is_heading_like(cleaned):
                    out.append(cleaned)           # heading/caption line
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells if c):
                continue                              # markdown separator row
            if not header:
                header = cells
                continue
            parts = [f"{header[i]}: {c}" for i, c in enumerate(cells[:len(header)]) if c and i < len(header)]
            unit = "; ".join(parts) if parts else "; ".join(cells)
            # a row means nothing without its table: "Counter: tSenderResponse;
            # Minimum: 24 ms" only answers "which counters are defined" if the
            # row knows it belongs to "Table 6.9 Counters"
            out.append(f"{caption}. {unit}" if caption else unit)
        return out or [text]
    cleaned = []
    for chunk in text.splitlines():
        value = _strip_decorations(chunk)
        if value:
            cleaned.append(value)
    if not cleaned:
        cleaned = [text]
    units = [s.strip() for line in cleaned for s in SENTENCE_SPLIT.split(line) if s.strip()]
    kept = [u for u in units if not _is_heading_like(u) and not _is_boilerplate(u)]
    return kept or units


def _select_diverse(scored: list[tuple[float, int, str, dict[str, Any]]], limit: int,
                    per_item_limit: int = 2) -> list[tuple[float, int, str, dict[str, Any]]]:
    """Keep the best units, at most ``per_item_limit`` from any single piece of evidence."""
    per_item: dict[int, int] = {}
    chosen: list[tuple[float, int, str, dict[str, Any]]] = []
    for candidate in scored:
        idx = candidate[1]
        if per_item.get(idx, 0) >= per_item_limit:
            continue
        per_item[idx] = per_item.get(idx, 0) + 1
        chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return chosen


def _is_relevant(unit: str, q_terms: set[str], q_ids: set[str]) -> bool:
    """Does this unit plausibly answer the question, or is it a neighbour?

    A unit that only echoes one word of a long question ("... clarified PS_RDY
    ordering" for a question about what PS_RDY signals) is a neighbour: it comes
    from the same section, but its facts are about something else, and padding an
    answer with it is how a correct answer turns into an incorrect one.  Short
    questions are held to a looser bar, because a question with two content terms
    cannot support a two-term requirement.
    """
    if q_ids and (q_ids & identifiers_in(unit)):
        return True
    shared = len(set(tokenize(unit)) & q_terms)
    required = 1 if len(q_terms) <= 3 else 2
    return shared >= required


def _sentence_score(sentence: str, q_terms: set[str], q_ids: set[str],
                    q_numbers: set[str] | None = None) -> float:
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    overlap = len(set(tokens) & q_terms) / max(1, len(q_terms))
    identifier = 1.0 if (q_ids and q_ids & identifiers_in(sentence)) else 0.0
    numbers = set(_NUM_RE.findall(sentence))
    number_bonus = 0.2 if (q_numbers and q_numbers & numbers) else (0.1 if numbers else 0.0)
    claim_bonus = 0.05 if CLAIM_HINT.search(sentence) else 0.0
    length_penalty = 0.2 if len(tokens) > 55 else 0.0
    return overlap + 1.2 * identifier + number_bonus + claim_bonus - length_penalty


# --------------------------------------------------------------------------- #
# HTTP backends
# --------------------------------------------------------------------------- #
@dataclass
class LLMOptions:
    endpoint: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    max_tokens: int = 700
    timeout_s: int = 120
    api_key: str = ""


class OllamaLLM:
    """Local open-weight model through the Ollama HTTP API."""

    backend = "ollama"

    def __init__(self, options: LLMOptions) -> None:
        self.options = options
        self.name = options.model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None,
                 **kwargs: Any) -> LLMResponse:
        import requests

        t0 = time.perf_counter()
        payload = {
            "model": self.options.model,
            "prompt": prompt,
            "system": system or None,
            "stream": False,
            "options": {"temperature": self.options.temperature,
                        "num_predict": max_tokens or self.options.max_tokens},
        }
        try:
            resp = requests.post(f"{self.options.endpoint.rstrip('/')}/api/generate", json=payload,
                                 timeout=self.options.timeout_s)
            resp.raise_for_status()
            body = resp.json()
            text = body.get("response", "")
            return LLMResponse(text=text, model=self.options.model, backend=self.backend,
                               prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
                               completion_tokens=int(body.get("eval_count", 0) or 0),
                               latency_s=time.perf_counter() - t0)
        except Exception as exc:                       # degrade to a described failure
            return LLMResponse(text="", model=self.options.model, backend=self.backend,
                               latency_s=time.perf_counter() - t0, error=str(exc))


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (vLLM, SGLang, LM Studio, hosted API)."""

    backend = "openai-compatible"

    def __init__(self, options: LLMOptions) -> None:
        self.options = options
        self.name = options.model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None,
                 **kwargs: Any) -> LLMResponse:
        import requests

        t0 = time.perf_counter()
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {self.options.api_key}"} if self.options.api_key else {}
        payload = {"model": self.options.model, "messages": messages,
                   "temperature": self.options.temperature,
                   "max_tokens": max_tokens or self.options.max_tokens}
        try:
            resp = requests.post(f"{self.options.endpoint.rstrip('/')}/v1/chat/completions", json=payload,
                                 headers=headers, timeout=self.options.timeout_s)
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage", {})
            return LLMResponse(text=body["choices"][0]["message"]["content"], model=self.options.model,
                               backend=self.backend, prompt_tokens=int(usage.get("prompt_tokens", 0)),
                               completion_tokens=int(usage.get("completion_tokens", 0)),
                               latency_s=time.perf_counter() - t0)
        except Exception as exc:
            return LLMResponse(text="", model=self.options.model, backend=self.backend,
                               latency_s=time.perf_counter() - t0, error=str(exc))


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model answer (fenced or bare)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if 0 <= start < end else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


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


def get_llm(config: Any | None = None, backend: str = "auto") -> Any:
    """Resolve the generation backend (``auto`` prefers a reachable Ollama)."""
    options = LLMOptions()
    if config is not None:
        options = LLMOptions(
            endpoint=str(config.get("agent.llm.endpoint", options.endpoint)),
            model=str(config.get("agent.llm.model", options.model)),
            temperature=float(config.get("agent.llm.temperature", 0.0)),
            max_tokens=int(config.get("agent.llm.max_tokens", options.max_tokens)),
            api_key=str(config.get("agent.llm.api_key", "") or ""),
        )
        if backend == "auto":
            backend = str(config.get("agent.llm.backend", "auto"))
    if backend == "extractive":
        return ExtractiveLLM()
    if backend == "ollama":
        return OllamaLLM(options)
    if backend == "openai-compatible":
        return OpenAICompatibleLLM(options)
    if backend == "auto":
        return OllamaLLM(options) if probe_ollama(options.endpoint) else ExtractiveLLM()
    raise ValueError(f"unknown LLM backend '{backend}'")


def generate(llm: Any, prompt: str, *, evidence: Sequence[dict[str, Any]] | None = None,
             question: str = "", system: str = "", max_tokens: int | None = None) -> LLMResponse:
    """Call any backend with a uniform signature (extractive needs the evidence)."""
    try:
        return llm.complete(prompt, system=system, max_tokens=max_tokens or 700, evidence=evidence,
                            question=question)
    except TypeError:                                  # backend without the extra kwargs
        return llm.complete(prompt, system=system, max_tokens=max_tokens or 700)
