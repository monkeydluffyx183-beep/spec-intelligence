"""Shared configuration: a tiny YAML-subset loader with env overrides.

The repository intentionally avoids a hard PyYAML dependency so that a clean
clone only needs CPython to run the deterministic pipeline.  The loader below
understands the subset of YAML used by ``config.yaml``: nested mappings,
lists of scalars or mappings, inline ``{a: 1, b: 2}`` maps, ``[a, b]`` lists,
quoted strings, block scalars (``>-``), comments and blank lines.

Overrides
---------
* ``SPECINTEL_CONFIG=/path/to/other.yaml`` selects a different file.
* ``SPECINTEL__AGENT__MAX_STEPS=12`` overrides ``agent.max_steps``
  (double underscore = nesting separator), values are coerced to bool/int/float
  when they look like one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


# --------------------------------------------------------------------------- #
# YAML subset parser
# --------------------------------------------------------------------------- #
def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    if (s[0] == s[-1]) and s[0] in "\"'" and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if low in {"null", "none", "~"}:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_inline(text: str) -> list[str]:
    """Split an inline ``[a, b]`` / ``{a: 1, b: 2}`` body on top-level commas."""
    parts, depth, buf = [], 0, []
    in_quote = ""
    for ch in text:
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in "\"'":
            in_quote = ch
            buf.append(ch)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_comment(s: str) -> str:
    """Drop a trailing ``# comment`` unless it is inside quotes."""
    out: list[str] = []
    quote = ""
    for ch in s:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (not out or out[-1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _flow_balance(s: str) -> int:
    """Bracket depth of an inline ``[...]`` / ``{...}`` value."""
    depth, quote = 0, ""
    for ch in s:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth


def _parse_value(raw: str) -> Any:
    s = _strip_comment(raw).strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1].strip()
        return [_parse_value(p) for p in _split_inline(body)] if body else []
    if s.startswith("{") and s.endswith("}"):
        body = s[1:-1].strip()
        out: dict[str, Any] = {}
        if body:
            for item in _split_inline(body):
                if ":" in item:
                    k, v = item.split(":", 1)
                    out[str(_coerce_scalar(k))] = _parse_value(v)
        return out
    return _coerce_scalar(s)


BLOCK_MARKERS = {">-", ">", "|", "|-", ">+", "|+"}


def _significant_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(indent, content)`` for every line that carries data."""
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if content.endswith(":") and "#" in content:  # tolerate 'key: # comment'
            content = content.split("#", 1)[0].strip()
        out.append((indent, content))
    return out


def _is_map_entry(content: str) -> bool:
    """True when a list item body such as ``doc_id: pd`` opens a mapping."""
    if content.startswith(("http://", "https://", "/", "\"", "'")):
        return False
    key, sep, _ = content.partition(":")
    if not sep:
        return False
    key = key.strip()
    return bool(key) and " " not in key and len(key) <= 60


def _parse_block(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    """Parse the block starting at ``lines[i]`` whose entries sit at ``indent``."""
    if i >= len(lines):
        return None, i
    if lines[i][1].startswith("-") and (lines[i][1] == "-" or lines[i][1][1:2] in ("", " ")):
        items: list[Any] = []
        while i < len(lines) and lines[i][0] == indent and _is_list_item(lines[i][1]):
            body = lines[i][1][1:].strip()
            if not body:
                sub, i = _parse_block(lines, i + 1, lines[i + 1][0] if i + 1 < len(lines) else indent + 2)
                items.append(sub)
            elif _is_map_entry(body):
                # the item opens a mapping on the same line; continuation lines are deeper
                sub_lines = [(indent + 2, body)]
                j = i + 1
                while j < len(lines) and lines[j][0] > indent:
                    sub_lines.append((max(indent + 2, lines[j][0]), lines[j][1]))
                    j += 1
                value, _ = _parse_block(sub_lines, 0, indent + 2)
                items.append(value)
                i = j
            else:
                items.append(_parse_value(body))
                i += 1
        return items, i

    mapping: dict[str, Any] = {}
    while i < len(lines) and lines[i][0] == indent and not _is_list_item(lines[i][1]):
        content = lines[i][1]
        key, sep, val = content.partition(":")
        if not sep:
            i += 1
            continue
        key = str(_coerce_scalar(key))
        rest = val.strip()
        if rest in BLOCK_MARKERS:
            j = i + 1
            chunk: list[str] = []
            while j < len(lines) and lines[j][0] > indent:
                chunk.append(lines[j][1])
                j += 1
            mapping[key] = (" ".join(chunk) if rest.startswith(">") else "\n".join(chunk))
            i = j
        elif rest:
            if rest[:1] in "[{" and _flow_balance(rest) > 0:
                # multi-line flow collection: keep reading deeper lines
                j = i + 1
                while j < len(lines) and _flow_balance(rest) > 0:
                    rest += " " + lines[j][1].strip()
                    j += 1
                mapping[key] = _parse_value(rest)
                i = j
            else:
                mapping[key] = _parse_value(rest)
                i += 1
        else:
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt and nxt[0] > indent:
                child, i = _parse_block(lines, i + 1, nxt[0])
                mapping[key] = child
            else:
                mapping[key] = None
                i += 1
    return mapping, i


def _is_list_item(content: str) -> bool:
    return content == "-" or content.startswith("- ")


def load_yaml_text(text: str) -> dict[str, Any]:
    """Parse the YAML subset used by ``config.yaml`` into nested dicts.

    Supports nested mappings, lists of scalars and of mappings, inline
    ``[a, b]`` lists, inline ``{a: 1}`` maps, quoted strings, block scalars
    (``>-``) and full-line comments.
    """
    lines = _significant_lines(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    return value if isinstance(value, dict) else {"value": value}


# --------------------------------------------------------------------------- #
# Config object
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    """Dict-like configuration with dotted access and path resolution."""

    data: dict[str, Any] = field(default_factory=dict)
    root: Path = REPO_ROOT
    source: Path | None = None

    # -- access ------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(dotted)
        return value

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path(self, dotted: str, default: str | None = None) -> Path:
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"no path configured at '{dotted}'")
        p = Path(str(raw)).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def documents(self) -> list[dict[str, Any]]:
        docs = self.get("corpus.documents", []) or []
        return [d for d in docs if isinstance(d, dict) and d.get("doc_id")]

    def doc_meta(self, doc_id: str) -> dict[str, Any]:
        for d in self.documents():
            if d["doc_id"] == doc_id:
                return d
        return {"doc_id": doc_id, "title": doc_id, "revision": "unknown", "revision_key": "0"}


def _apply_env_overrides(cfg: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env if env is not None else dict(os.environ)
    for key, value in env.items():
        if not key.startswith("SPECINTEL__"):
            continue
        dotted = key[len("SPECINTEL__"):].lower().replace("__", ".")
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _parse_value(value)
    return cfg


_CACHE: dict[str, Config] = {}


def load_config_fresh(path: str | Path | None = None) -> Config:
    """Load the configuration with the cache bypassed.

    An ablation that changes a single option (one score floor for every
    retriever) must not mutate the cached object that the next run reads.
    """
    _CACHE.pop(str(Path(path or os.environ.get("SPECINTEL_CONFIG") or DEFAULT_CONFIG)
                   .expanduser().resolve()), None)
    return load_config(path, use_cache=False)


def load_config(path: str | Path | None = None, *, use_cache: bool = True) -> Config:
    """Load ``config.yaml`` (plus env overrides) into a :class:`Config`."""
    env_path = os.environ.get("SPECINTEL_CONFIG")
    cfg_path = Path(path or env_path or DEFAULT_CONFIG).expanduser()
    key = str(cfg_path.resolve())
    if use_cache and key in _CACHE:
        return _CACHE[key]
    text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
    data = load_yaml_text(text)
    data = _apply_env_overrides(data)
    config = Config(data=data, root=REPO_ROOT, source=cfg_path if cfg_path.exists() else None)
    if use_cache:
        _CACHE[key] = config
    return config


def ensure_dirs(config: Config, keys: Iterable[str]) -> list[Path]:
    """Create the configured directories for ``keys`` and return them."""
    made = []
    for k in keys:
        p = config.path(k)
        p.mkdir(parents=True, exist_ok=True)
        made.append(p)
    return made
