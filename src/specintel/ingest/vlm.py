"""WS1 — VLM adapters for figure structuring.

The pipeline is *local-first*: the backends below are tried in order and the
deterministic heuristic backend is always available, so a clone without a GPU
still produces structured figures (with an honest confidence and a note about
what could not be read from the labels).

Backends
--------
``HeuristicVLM``       no model: reads the labels inside the crop, classifies
                       the diagram, returns partial structure + notes.
``OllamaVLM``          local open-weight VLM (e.g. ``qwen2.5-vl:7b``) over the
                       Ollama HTTP API.
``OpenAICompatibleVLM``any OpenAI-compatible server (vLLM, SGLang, LM Studio).

Every backend returns :class:`specintel.ingest.figures.StructureResult`, and
the JSON is validated against the schema for the figure type so a hallucinated
key or a malformed transition is rejected instead of entering the graph.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from specintel.ingest.figures import StructureResult, classify_figure, structure_heuristic

STATE_SCHEMA_KEYS = {"states", "transitions"}
TIMING_SCHEMA_KEYS = {"signals", "events", "timers"}

PROMPT = (
    "You are structuring one figure from a USB Power Delivery / Type-C / USB4 / Tech 2 engineering "
    "specification. Return JSON only.\n"
    "If the figure is a state machine: {\"kind\": \"state_machine\", \"description\": str, "
    "\"states\": [str], \"transitions\": [{\"from\": str, \"to\": str, \"trigger\": str, \"actions\": [str]}]}\n"
    "If the figure is a timing or waveform diagram: {\"kind\": \"timing\", \"description\": str, "
    "\"signals\": [str], \"events\": [str], \"timers\": [str]}\n"
    "Otherwise: {\"kind\": \"block\", \"description\": str, \"labels\": [str]}\n"
    "Copy state and timer names exactly as printed. Do not invent entities that are not in the image. "
    "The caption is: {caption}."
)


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class VLMBackend:
    """Interface every figure-structuring backend implements."""

    name = "base"

    def structure_figure(self, image_path: str | None, *, caption: str = "", labels: str = "",
                         has_raster: bool = False) -> StructureResult:
        raise NotImplementedError

    # -- helpers shared by HTTP backends --------------------------------- #
    @staticmethod
    def encode_image(image_path: str | None) -> str | None:
        if not image_path:
            return None
        p = Path(image_path)
        if not p.exists():
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")

    @staticmethod
    def parse_json(text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from a model answer."""
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = text.find("{")
            end = text.rfind("}")
            candidate = text[start:end + 1] if 0 <= start < end else None
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def validate_structure(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Reject malformed structured-figure JSON (Task 2, Step 4)."""
    notes: list[str] = []
    kind = str(payload.get("kind", "")).lower()
    out: dict[str, Any] = {}
    if kind == "state_machine" or STATE_SCHEMA_KEYS & set(payload):
        states = [str(s) for s in payload.get("states", []) if isinstance(s, (str, int))][:60]
        transitions: list[dict[str, Any]] = []
        for tr in payload.get("transitions", []) or []:
            if not isinstance(tr, dict) or "from" not in tr or "to" not in tr:
                notes.append(f"rejected transition without from/to: {tr!r}")
                continue
            if states and (str(tr["from"]) not in states or str(tr["to"]) not in states):
                notes.append(f"rejected transition with unknown endpoint: {tr.get('from')} -> {tr.get('to')}")
                continue
            actions = tr.get("actions") or []
            transitions.append({
                "from": str(tr["from"]), "to": str(tr["to"]),
                "trigger": str(tr.get("trigger", "")),
                "actions": [str(a) for a in actions] if isinstance(actions, list) else [str(actions)],
            })
        out = {"kind": "state_machine", "states": states, "transitions": transitions}
    elif kind == "timing" or TIMING_SCHEMA_KEYS & set(payload):
        out = {"kind": "timing",
               "signals": [str(s) for s in payload.get("signals", [])][:30],
               "events": [str(s) for s in payload.get("events", [])][:30],
               "timers": [str(s) for s in payload.get("timers", [])][:30]}
    else:
        out = {"kind": kind or "block", "labels": [str(s) for s in payload.get("labels", [])][:40]}
    if payload.get("description"):
        out["description"] = str(payload["description"])[:1200]
    return out, notes


# --------------------------------------------------------------------------- #
# deterministic backend
# --------------------------------------------------------------------------- #
class HeuristicVLM(VLMBackend):
    """Offline default: structure what the text layer inside the crop gives us."""

    name = "heuristic"

    def structure_figure(self, image_path: str | None, *, caption: str = "", labels: str = "",
                         has_raster: bool = False) -> StructureResult:
        result = structure_heuristic(labels, caption, has_raster=has_raster)
        result.backend = self.name
        if has_raster and not labels.strip():
            result.notes.append("raster figure with no text layer: needs a real VLM/OCR to describe it")
            result.confidence = 0.1
        return result


# --------------------------------------------------------------------------- #
# HTTP backends
# --------------------------------------------------------------------------- #
@dataclass
class HttpOptions:
    endpoint: str = "http://localhost:11434"
    model: str = "qwen2.5-vl:7b"
    timeout_s: int = 120
    max_tokens: int = 700
    temperature: float = 0.0


class OllamaVLM(VLMBackend):
    """Local open-weight VLM served by Ollama (``/api/generate``)."""

    name = "ollama"

    def __init__(self, options: HttpOptions) -> None:
        self.options = options

    def structure_figure(self, image_path: str | None, *, caption: str = "", labels: str = "",
                         has_raster: bool = False) -> StructureResult:
        import requests  # imported lazily so the core pipeline needs no dependencies

        prompt = PROMPT.format(caption=caption or "(none)")
        if labels.strip():
            prompt += "\nText extracted around the figure (may be incomplete):\n" + labels[:2000]
        payload = {
            "model": self.options.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "images": [self.encode_image(image_path)] if image_path else [],
            "options": {"temperature": self.options.temperature, "num_predict": self.options.max_tokens},
        }
        resp = requests.post(f"{self.options.endpoint.rstrip('/')}/api/generate", json=payload,
                             timeout=self.options.timeout_s)
        resp.raise_for_status()
        body = resp.json()
        parsed = self.parse_json(body.get("response", "")) or {}
        structure, notes = validate_structure(parsed)
        kind = structure.pop("kind", "block")
        description = structure.pop("description", "") or f"{kind} figure"
        return StructureResult(kind=kind, description=description, structure=structure,
                               confidence=0.8, notes=notes, backend=self.name)


class OpenAICompatibleVLM(VLMBackend):
    """Any OpenAI-compatible vision endpoint (vLLM, SGLang, LM Studio, hosted APIs)."""

    name = "openai-compatible"

    def __init__(self, options: HttpOptions) -> None:
        self.options = options

    def structure_figure(self, image_path: str | None, *, caption: str = "", labels: str = "",
                         has_raster: bool = False) -> StructureResult:
        import requests

        prompt = PROMPT.format(caption=caption or "(none)")
        if labels.strip():
            prompt += "\nText extracted around the figure (may be incomplete):\n" + labels[:2000]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        encoded = self.encode_image(image_path)
        if encoded:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        payload = {
            "model": self.options.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.options.temperature,
            "max_tokens": self.options.max_tokens,
        }
        resp = requests.post(f"{self.options.endpoint.rstrip('/')}/v1/chat/completions", json=payload,
                             timeout=self.options.timeout_s)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        parsed = self.parse_json(text) or {}
        structure, notes = validate_structure(parsed)
        kind = structure.pop("kind", "block")
        description = structure.pop("description", "") or f"{kind} figure"
        return StructureResult(kind=kind, description=description, structure=structure,
                               confidence=0.8, notes=notes, backend=self.name)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def probe(endpoint: str, timeout_s: float = 1.5) -> bool:
    """Return True when an HTTP endpoint answers (used by ``auto``)."""
    try:
        import requests

        requests.get(endpoint, timeout=timeout_s)
        return True
    except Exception:
        return False


def get_vlm(config: Any | None = None, backend: str = "auto") -> VLMBackend:
    """Resolve the VLM backend, falling back to the heuristic when unreachable."""
    options = HttpOptions()
    if config is not None:
        options = HttpOptions(
            endpoint=str(config.get("agent.vlm.endpoint", options.endpoint)),
            model=str(config.get("agent.vlm.model", options.model)),
            timeout_s=int(config.get("agent.vlm.timeout_s", options.timeout_s)),
        )
        backend = backend if backend != "auto" else str(config.get("agent.vlm.backend", "auto"))
    if backend in ("heuristic", "none", "auto"):
        if backend == "heuristic" or backend == "none":
            return HeuristicVLM()
        if probe(options.endpoint):
            return OllamaVLM(options)
        return HeuristicVLM()
    if backend == "ollama":
        return OllamaVLM(options)
    if backend == "openai-compatible":
        return OpenAICompatibleVLM(options)
    raise ValueError(f"unknown VLM backend '{backend}'")


__all__ = ["VLMBackend", "HeuristicVLM", "OllamaVLM", "OpenAICompatibleVLM", "HttpOptions", "get_vlm",
           "validate_structure", "classify_figure"]
