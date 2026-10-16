"""One log format for every workstream (see INTERFACES.md § log format).

Every record carries the same key=value shape so a single run can be traced
from a PDF page to a cited answer:

    2026-10-16T09:12:03Z INFO  ws2.index  event=index.build doc=pd \
        chunks=1284 dim=384 duration_s=3.412 rss_mb=214

Usage
-----
    from specintel.common.logging import get_logger, log_event, timed

    log = get_logger("ws1.ingest", config=cfg)
    with timed(log, "parse.page", doc="pd", page=7) as t:
        ...                      # t["rows"] = 42  -> merged into the event
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)-18s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: One line of the text log format, as documented in docs/INTERFACES.md.  Every
#: event is prefixed with ``event=`` so a run can be traced with a plain grep:
#: ``grep 'event=tool.call' logs/*.log``.  The test suite asserts against this
#: pattern, so the format cannot drift without a test noticing.
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL|WARN)\s+"
    r"(?P<logger>[\w.]+)\s+event=(?P<event>[\w.\-]+)(?:\s+(?P<fields>.*))?$")

_WS_OF = {
    "ws1": "WS1 Document AI & OCR",
    "ws2": "WS2 Multimodal Retrieval",
    "ws3": "WS3 Knowledge Graph",
    "ws4": "WS4 Agentic RAG",
    "ws5": "WS5 Evaluation & Integration",
    "common": "shared",
}

_CONFIGURED: dict[str, Any] = {"file": None, "json": False}


class _KeyValueFormatter(logging.Formatter):
    """Renders one event as ``<ts> <level> <logger> event=<name> key=value …``."""

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if not name.startswith("specintel"):
            return super().format(record)          # libraries keep their own shape
        head = (f"{self.formatTime(record, self.datefmt)} {record.levelname:<5s} "
                f"{record.name:<18s} event={record.getMessage()}")
        parts = []
        for k in sorted(getattr(record, "kv", None) or {}):
            v = getattr(record, "kv")[k]
            if isinstance(v, float):
                v = f"{v:.3f}"
            elif isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, default=str, separators=(",", ":"))
            parts.append(f"{k}={v}")
        line = f"{head} {' '.join(parts)}" if parts else head
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        return f"{line}\n{record.exc_text}" if record.exc_text else line


class _JsonFormatter(logging.Formatter):
    """Structured (JSON lines) output for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime(DATE_FORMAT, time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "pid": record.process,
        }
        payload.update(getattr(record, "kv", {}) or {})
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
    json_format: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``specintel`` logger tree once per process."""
    root = logging.getLogger("specintel")
    if _CONFIGURED["file"] is not None and not force:
        return root
    root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    root.handlers.clear()
    root.propagate = False
    fmt_cls = _JsonFormatter if json_format else _KeyValueFormatter
    if console and not json_format:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt_cls(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt_cls(LOG_FORMAT, DATE_FORMAT) if not json_format else fmt_cls())
        root.addHandler(fh)
    if json_format:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt_cls())
        root.addHandler(sh)
    _CONFIGURED.update({"file": log_file, "json": json_format})
    return root


def setup_from_config(config: Any) -> logging.Logger:
    """Configure logging from the ``logging:`` section of ``config.yaml``."""
    log_file = None
    if config.get("logging.file"):
        log_file = config.path("logging.file")
    return setup_logging(
        level=config.get("logging.level", "INFO"),
        log_file=log_file,
        console=bool(config.get("logging.console", True)),
        json_format=str(config.get("logging.format", "text")) == "json",
    )


def get_logger(name: str, config: Any | None = None) -> logging.Logger:
    """Return ``specintel.<name>``; configures logging on first use.

    ``name`` should start with a workstream tag (``ws1``…``ws5``) so the
    owning workstream is visible in every line.
    """
    if config is not None:
        setup_from_config(config)
    elif not logging.getLogger("specintel").handlers:
        setup_logging(log_file=os.environ.get("SPECINTEL_LOG_FILE"))
    return logging.getLogger(f"specintel.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event: ``event=<name> key=value ...``."""
    logger.log(level, event, extra={"kv": {k: v for k, v in fields.items() if v is not None}})


def log_exception(logger: logging.Logger, event: str, exc: BaseException, **fields: Any) -> None:
    """Log an exception as a structured event, keeping the traceback."""
    fields = {**fields, "error_type": type(exc).__name__, "error": str(exc)[:300]}
    logger.error(event, extra={"kv": fields}, exc_info=exc)


@contextlib.contextmanager
def timed(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log ``event`` with duration, RSS and merged fields.

    The yielded dict is merged into the log record, so callers can report
    counts and sizes discovered inside the block::

        with timed(log, "ocr.pass", doc="pd", pages=8) as t:
            ...                        # do work
            t["blocks"] = 412
    """
    from specintel.common.resources import rss_mb

    start = time.perf_counter()
    payload: dict[str, Any] = dict(fields)
    try:
        yield payload
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        payload.update(duration_s=time.perf_counter() - start, rss_mb=rss_mb(), status="error")
        log_exception(logger, event, exc, **payload)
        raise
    payload.update(duration_s=time.perf_counter() - start, rss_mb=rss_mb(), status="ok")
    log_event(logger, event, level=level, **payload)
