"""HTTP API + demo UI for the pipeline, in the standard library only.

Why no framework: the demo has to start on a GPU box, a reviewer's laptop and a
clean clone without a dependency install, and the whole surface is five JSON
endpoints.  ``http.server`` with a thread pool does that in one file, and the UI
is one static page that talks to the same endpoints a script would.

Endpoints::

    GET  /                     demo UI (static)
    GET  /api/health           liveness + which artefacts are loaded
    GET  /api/stats            index, graph and evaluation numbers
    GET  /api/docs             the document registry with revisions and pages
    POST /api/ask              {"question": str, "docs": [str], "trace": bool, "top_k": int}
    POST /api/search           {"query": str, "k": int, "tables_only": bool}
    POST /api/graph            {"kind": str, "name": str, ...}

The server binds ``0.0.0.0`` and never checks the ``Host`` or ``Origin`` header:
reviewers reach it through a preview proxy whose host name is not known in
advance, and rejecting that host would take the demo down.  It is a read-only
demo over licensed documents, so there is nothing to protect behind an origin
check; ``--token`` adds a shared secret if it is ever exposed further.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from specintel.common.config import Config, load_config
from specintel.common.logging import get_logger, log_event, log_exception
from specintel.common.resources import hardware_profile

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024


class DemoState:
    """Lazily built, shared by every request thread.

    Loading the index and the graph takes about a second and building the agent
    is cheap, so everything is built on first use and then reused; the lock keeps
    two simultaneous first requests from building twice.
    """

    def __init__(self, config: Config, token: str = "") -> None:
        self.config = config
        self.token = token
        self.log = get_logger("ws5.server", config)
        # reentrant: building the agent builds the toolkit, and both take this lock
        self._lock = threading.RLock()
        self._agent: Any = None
        self._toolkit: Any = None
        self._llm: Any = None
        self._started = time.time()
        self.requests = 0
        self.errors = 0

    # -- lazily built pieces ------------------------------------------------- #
    @property
    def toolkit(self) -> Any:
        if self._toolkit is None:
            with self._lock:
                if self._toolkit is None:
                    from specintel.agent.tools import Toolkit

                    self._toolkit = Toolkit.from_config(self.config)
                    log_event(self.log, "server.toolkit.loaded",
                              tables=len(getattr(self._toolkit, "tables", []) or []),
                              figures=len(getattr(self._toolkit, "figures", []) or []))
        return self._toolkit

    @property
    def agent(self) -> Any:
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    from specintel.agent.loop import AgentConfig, build_agent
                    from specintel.common.llm import get_llm

                    self._llm = get_llm(self.config, "auto")
                    self._agent = build_agent(
                        self.config, toolkit=self.toolkit, llm=self._llm,
                        agent_config=AgentConfig(max_steps=8, budget_s=30.0, top_k=8))
                    log_event(self.log, "server.agent.loaded", model=getattr(self._llm, "name", ""),
                              backend=getattr(self._llm, "backend", ""))
        return self._agent

    @property
    def uptime_s(self) -> float:
        return round(time.time() - self._started, 1)

    # -- payloads ------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        processed = self.config.path("paths.processed")
        docs = sorted(p.name for p in processed.iterdir()) if processed.exists() else []
        return {
            "status": "ok" if docs else "degraded",
            "documents": docs,
            "model": getattr(self._llm, "name", "extractive"),
            "backend": getattr(self._llm, "backend", "extractive"),
            "uptime_s": self.uptime_s,
            "requests": self.requests,
            "errors": self.errors,
            "hardware": hardware_profile(),
        }

    def stats(self) -> dict[str, Any]:
        toolkit = self.toolkit
        graph = toolkit.store.stats().to_dict() if getattr(toolkit, "store", None) is not None else {}
        chunks = list(getattr(toolkit.index, "chunks", []) or [])
        payload: dict[str, Any] = {
            "index": {"chunks": len(chunks), "documents": sorted({c.doc_id for c in chunks})},
            "elements": {"tables": len(getattr(toolkit, "tables", []) or []),
                         "figures": len(getattr(toolkit, "figures", []) or [])},
            "graph": graph,
        }
        summary_path = self.config.root / "eval" / "results" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final = summary.get("configs", ["F"])[-1]
            payload["evaluation"] = {
                "n_questions": summary.get("n_questions"),
                "config": final,
                "targets": summary.get("targets_measured", {}),
                "measured": summary.get("aggregates", {}).get(final, {}).get("overall", {}),
                "generated_at": summary.get("generated_at"),
            }
        return payload

    def documents(self) -> list[dict[str, Any]]:
        """The document registry, as the UI's document picker needs it.

        ``config.documents()`` is the contract (a list of mappings under
        ``corpus.documents``); the dict-keyed lookup that used to be here returned
        nothing at all, so the picker was empty and every demo query silently ran
        across all seven documents.
        """
        processed = self.config.path("paths.processed")
        registry = []
        for entry in self.config.documents():
            doc_id = entry.get("doc_id", "")
            registry.append({"doc_id": doc_id, "title": entry.get("title", ""),
                             "revision": entry.get("revision", ""),
                             "family": entry.get("family") or entry.get("namespace", ""),
                             "source": entry.get("source") or entry.get("main_file", ""),
                             # the registry names every document of the corpus; this
                             # flag says which ones this checkout has actually ingested
                             "ingested": bool(doc_id) and (processed / doc_id).exists()})
        return registry

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        question = str(body.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")
        docs = body.get("docs") or None
        if isinstance(docs, str):
            docs = [d.strip() for d in docs.split(",") if d.strip()]
        top_k = int(body.get("top_k") or 8)
        agent = self.agent
        if top_k != 8:
            agent.agent_config.top_k = max(1, min(top_k, 20))
        started = time.perf_counter()
        result = agent.answer(question, docs=docs)
        trace = getattr(result, "trace", None)
        return {
            "question": question,
            "answer": result.answer,
            "declined": bool(result.declined),
            "decline_reason": getattr(result, "decline_reason", ""),
            "citations": [c.to_marker() for c in (result.citations or [])],
            "tool_calls": result.tool_calls,
            "latency_s": round(time.perf_counter() - started, 3),
            "evidence": [
                {"doc_id": h.doc_id, "revision": h.meta.get("revision", ""), "page": h.page,
                 "section": h.section_number, "element_type": h.element_type,
                 "score": round(float(h.score), 4), "source": h.source,
                 "table_id": h.table_id, "figure_id": h.figure_id,
                 "text": str(h.text)[:600]}
                for h in (result.evidence or [])
            ],
            "trace": trace.to_dict() if (trace is not None and body.get("trace", True)) else None,
        }

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        from specintel.index.search import Filters

        query = str(body.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        filters = Filters(docs=tuple(body.get("docs") or ()),
                          tables_only=bool(body.get("tables_only")),
                          figures_only=bool(body.get("figures_only")))
        hits = self.toolkit.index.search_text(query, k=int(body.get("k") or 8), filters=filters)
        return {"query": query, "hits": [
            {"chunk_id": h.chunk_id, "score": round(float(h.score), 4), "doc_id": h.doc_id,
             "page": h.page, "section": h.section_number, "element_type": h.element_type,
             "table_id": h.table_id, "figure_id": h.figure_id,
             "component_scores": h.component_scores, "text": str(h.text)[:900]}
            for h in hits]}

    def graph(self, body: dict[str, Any]) -> dict[str, Any]:
        kind = str(body.get("kind") or "neighbors")
        result = self.toolkit.call("graph_query", kind=kind, name=str(body.get("name") or ""),
                                   src=str(body.get("src") or ""), dst=str(body.get("dst") or ""),
                                   depth=int(body.get("depth") or 1),
                                   limit=int(body.get("limit") or 25))
        return {"kind": kind, "text": result.text, "graph": result.graph,
                "error": result.error}


class Handler(BaseHTTPRequestHandler):
    server_version = "spec-intelligence/1.0"
    state: DemoState

    # -- plumbing ------------------------------------------------------------ #
    def log_message(self, fmt: str, *args: Any) -> None:          # noqa: A003 - base class API
        log_event(self.state.log, "server.request", client=self.address_string(),
                  message=fmt % args)

    def _send(self, code: int, payload: Any, *, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _authorised(self) -> bool:
        token = self.state.token
        if not token:
            return True
        if self.headers.get("X-Specintel-Token") == token:
            return True
        return False

    # -- routing ------------------------------------------------------------- #
    def do_OPTIONS(self) -> None:                                 # noqa: N802 - base class API
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Specintel-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:                                     # noqa: N802 - base class API
        path = urlparse(self.path).path
        started = time.perf_counter()
        try:
            if path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                return self._serve_static(path.split("/static/", 1)[1])
            if path == "/api/health":
                return self._send(200, self.state.health())
            if path == "/api/stats":
                return self._send(200, self.state.stats())
            if path == "/api/docs":
                return self._send(200, {"documents": self.state.documents()})
            if path in ("/favicon.ico",):
                return self._send(204, b"", content_type="image/x-icon")
            return self._send(404, {"error": f"no such path: {path}"})
        except Exception as exc:                                  # noqa: BLE001 - keep serving
            self.state.errors += 1
            log_exception(self.state.log, "server.get.failed", exc, path=path)
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            log_event(self.state.log, "server.get", path=path,
                      duration_s=round(time.perf_counter() - started, 3))

    def do_POST(self) -> None:                                    # noqa: N802 - base class API
        path = urlparse(self.path).path
        started = time.perf_counter()
        self.state.requests += 1
        try:
            if not self._authorised():
                return self._send(401, {"error": "missing or wrong X-Specintel-Token"})
            body = self._read_body()
            if path == "/api/ask":
                return self._send(200, self.state.ask(body))
            if path == "/api/search":
                return self._send(200, self.state.search(body))
            if path == "/api/graph":
                return self._send(200, self.state.graph(body))
            return self._send(404, {"error": f"no such path: {path}"})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:                                  # noqa: BLE001 - keep serving
            self.state.errors += 1
            log_exception(self.state.log, "server.post.failed", exc, path=path)
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            log_event(self.state.log, "server.post", path=path,
                      duration_s=round(time.perf_counter() - started, 3))

    def _serve_static(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        if not str(target).startswith(str(STATIC_DIR)) or not target.is_file():
            return self._send(404, {"error": f"no such file: {name}"})
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), content_type=ctype)


def build_server(config: Config, *, host: str = "0.0.0.0", port: int = 8000,
                 token: str = "") -> ThreadingHTTPServer:
    state = DemoState(config, token=token)
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.state = state                                          # type: ignore[attr-defined]
    return httpd


def serve(config: Config, *, host: str = "0.0.0.0", port: int = 8000, token: str = "") -> int:
    """Run the demo server until interrupted."""
    httpd = build_server(config, host=host, port=port, token=token)
    log = get_logger("ws5.server", config)
    log_event(log, "server.listen", host=host, port=port, static=str(STATIC_DIR))
    print(f"\n  spec-intelligence demo\n  http://{host}:{port}/  (Ctrl-C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:                                     # pragma: no cover - interactive
        print("\n  stopping\n")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:                    # pragma: no cover - thin wrapper
    import argparse

    parser = argparse.ArgumentParser(prog="specintel-server", description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    return serve(load_config(), host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
