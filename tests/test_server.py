"""WS5 — the demo server: HTTP surface, auth, error handling and static page.

The server is started on an ephemeral port in a background thread, so the tests
exercise the real request path (routing, JSON encoding, status codes) rather than
calling the handlers directly.  The content assertions stay loose because the
fixture corpus is two small synthetic documents.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from specintel.server.app import DemoState, build_server


@pytest.fixture(scope="module")
def server(ingested_tiny):
    from specintel.index.build import build_indexes
    from specintel.kg.build import build_graph

    build_indexes(ingested_tiny, doc_ids=["pd", "typec"])
    build_graph(ingested_tiny, doc_ids=["pd", "typec"], use_llm=False)
    httpd: ThreadingHTTPServer = build_server(ingested_tiny, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def state(ingested_tiny):
    return DemoState(ingested_tiny)


def get(url: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post(url: str, payload: dict, *, headers: dict | None = None):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# state object
# --------------------------------------------------------------------------- #
def test_state_health_names_the_documents_it_loaded(state):
    payload = state.health()
    assert payload["status"] == "ok"
    assert {"pd", "typec"} <= set(payload["documents"])
    assert payload["hardware"], "the hardware profile travels with every health check"


def test_state_documents_reads_the_registry(state):
    """The document picker must be populated (it once came back empty)."""
    docs = state.documents()
    by_id = {d["doc_id"]: d for d in docs}
    assert {"doc_id", "title", "revision", "family", "ingested"} <= set(docs[0])
    assert by_id["pd"]["ingested"] is True and by_id["typec"]["ingested"] is True
    assert by_id["pd"]["title"], "the picker shows the document title"
    assert any(not d["ingested"] for d in docs), "the registry also lists documents not yet ingested"


def test_state_ask_requires_a_question(state):
    with pytest.raises(ValueError):
        state.ask({})


def test_state_search_requires_a_query(state):
    with pytest.raises(ValueError):
        state.search({})


def test_state_ask_returns_the_cited_answer_shape(state):
    payload = state.ask({"question": "What are the limits of tSenderResponse?"})
    assert {"question", "answer", "declined", "citations", "evidence", "latency_s"} <= set(payload)
    assert payload["answer"]
    if not payload["declined"]:
        assert payload["citations"], "an answer served to the demo UI is cited"
        assert all(c.startswith("[") for c in payload["citations"])
        assert payload["evidence"], "the demo UI shows the evidence behind the citations"
        assert {"doc_id", "page", "section", "score", "text"} <= set(payload["evidence"][0])
    assert payload["trace"], "the trace is returned by default (the UI draws it)"


def test_state_ask_can_omit_the_trace(state):
    payload = state.ask({"question": "What are the limits of tSenderResponse?", "trace": False})
    assert payload["trace"] is None


def test_state_graph_answers_a_neighbour_query(state):
    payload = state.graph({"kind": "neighbors", "name": "tSenderResponse"})
    assert payload["kind"] == "neighbors"
    assert payload["error"] == ""
    assert payload["graph"] is not None


def test_health_degrades_when_no_documents_are_ingested(tmp_path, ingested_tiny):
    import copy

    from specintel.common.config import Config

    data = copy.deepcopy(ingested_tiny.data)
    data["paths"]["processed"] = str(tmp_path / "nothing_here")
    empty = DemoState(Config(data=data, root=tmp_path, source=None))
    payload = empty.health()
    assert payload["status"] == "degraded"
    assert payload["documents"] == []


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
def test_index_page_is_served(server):
    with urllib.request.urlopen(f"{server}/", timeout=20) as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert "spec" in body.lower() and "<html" in body.lower()


def test_health_and_stats_endpoints(server):
    status, health = get(f"{server}/api/health")
    assert status == 200 and health["status"] == "ok"
    status, stats = get(f"{server}/api/stats")
    assert status == 200
    assert stats["index"]["chunks"] > 0
    assert isinstance(stats["graph"], dict)


def test_docs_endpoint_lists_what_was_ingested(server):
    status, payload = get(f"{server}/api/docs")
    assert status == 200
    by_id = {d["doc_id"]: d for d in payload["documents"]}
    assert by_id["pd"]["ingested"] is True and by_id["typec"]["ingested"] is True
    assert by_id["pd"]["title"]


def test_ask_endpoint_returns_a_cited_answer(server):
    status, payload = post(f"{server}/api/ask", {"question": "What are the limits of tSenderResponse?"})
    assert status == 200
    assert payload["answer"] and not payload["declined"]
    assert payload["citations"], "the demo must not serve an uncited answer"
    assert payload["latency_s"] >= 0


def test_ask_endpoint_rejects_an_empty_question(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(f"{server}/api/ask", {"question": "   "})
    assert excinfo.value.code == 400
    assert "question is required" in json.loads(excinfo.value.read().decode("utf-8"))["error"]


def test_search_endpoint_honours_filters(server):
    status, payload = post(f"{server}/api/search", {"query": "timer", "k": 3, "tables_only": True})
    assert status == 200
    assert len(payload["hits"]) <= 3
    assert all(hit["element_type"] == "table" for hit in payload["hits"])


def test_graph_endpoint_returns_nodes_and_edges(server):
    status, payload = post(f"{server}/api/graph", {"kind": "neighbors", "name": "tSenderResponse"})
    assert status == 200
    assert payload["graph"] is not None


def test_unknown_paths_are_404_not_500(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(f"{server}/api/nope")
    assert excinfo.value.code == 404


def test_cors_preflight_is_allowed(server):
    request = urllib.request.Request(f"{server}/api/ask", method="OPTIONS")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204
        assert response.headers.get("Access-Control-Allow-Origin") == "*"


def test_a_token_is_enforced_when_one_is_configured(ingested_tiny):
    httpd = build_server(ingested_tiny, host="127.0.0.1", port=0, token="s3cret")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(f"{base}/api/ask", {"question": "What are the limits of tSenderResponse?"})
        assert excinfo.value.code == 401
        status, payload = post(f"{base}/api/ask", {"question": "What are the limits of tSenderResponse?"},
                               headers={"X-Specintel-Token": "s3cret"})
        assert status == 200 and payload["answer"]
        # the read-only health check stays open so a probe can see the service is up
        assert get(f"{base}/api/health")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_errors_are_counted_on_the_state(server, state):
    before = state.errors
    with pytest.raises(urllib.error.HTTPError):
        post(f"{server}/api/ask", {"question": ""})
    assert state.errors == before, "a bad request is not an internal error"
