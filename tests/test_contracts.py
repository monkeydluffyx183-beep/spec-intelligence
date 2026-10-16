"""Contracts, configuration and logging: the interfaces every workstream shares.

These tests are the machine-checkable half of ``docs/INTERFACES.md``.  If one of
them has to change, the interface changed and another workstream needs to know.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from specintel.common import models
from specintel.common.config import Config, load_config
from specintel.common.logging import LOG_LINE_RE, get_logger, log_event, setup_logging
from specintel.common.resources import hardware_profile, rss_mb


def test_citation_marker_format():
    citation = models.Citation(doc_id="pd", revision="R3.2 V1.2", section_number="6.6", page=4,
                               region="Table 6.8")
    assert citation.to_marker() == "[pd, R3.2 V1.2, §6.6, p.4, Table 6.8]"


def test_every_contract_round_trips_through_dict():
    """Each contract must survive to_dict/from_dict: traces and results are JSON."""
    samples = {
        "Element": models.Element(element_id="pd:1", doc_id="pd", page=1, element_type="paragraph",
                                  text="PS_RDY indicates ready.", section_number="6.3",
                                  revision="R3.2 V1.2"),
        "Table": models.Table(table_id="pd:table:6.8", doc_id="pd", number="6.8", caption="Timers",
                              headers=[["Timer", "Min"]], rows=[["tSenderResponse", "24"]], pages=[3],
                              revision="R3.2 V1.2"),
        "Chunk": models.Chunk(chunk_id="pd:1", doc_id="pd", page=1, text="…", revision="R3.2 V1.2",
                              element_type="paragraph"),
        "SearchHit": models.SearchHit(chunk_id="c1", score=0.5, doc_id="pd", page=2, text="…"),
        "Node": models.Node(node_id="timer:tSenderResponse", node_type="Timer",
                            name="tSenderResponse", doc_id="pd"),
        "Edge": models.Edge(edge_id="e1", src_id="a", dst_id="b", edge_type="DEFINED_IN",
                            doc_id="pd", section_number="6.6", page=3),
        "ToolResult": models.ToolResult(tool="search_text", text="…"),
        "AgentTrace": models.AgentTrace(run_id="run_1", question="q", model="extractive"),
        "GoldQuestion": models.GoldQuestion(qid="S1", question="q", gold_answer="a", category="S",
                                            content_type="prose", author="A", split="held_out"),
    }
    for name, value in samples.items():
        data = value.to_dict()
        assert isinstance(data, dict) and data, name
        json.dumps(data, default=str)                            # must be JSON serialisable
        restored = type(value).from_dict(data)
        assert isinstance(restored, type(value)), name
        assert restored.to_dict() == data, f"{name} does not round-trip exactly"


def test_config_resolves_paths_and_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml", use_cache=False)
    assert config.path("paths.processed").is_absolute()
    assert isinstance(config.get("index.embedding.backend", ""), str)
    monkeypatch.setenv("SPECINTEL__AGENT__TOP_K", "11")
    fresh = load_config(Path(__file__).resolve().parents[1] / "config.yaml", use_cache=False)
    assert fresh.get("agent.top_k") == 11 or fresh.get("agent.top_k") == "11"


def test_log_lines_match_the_documented_format(tmp_path: Path):
    log_file = tmp_path / "run.jsonl"
    setup_logging(level="INFO", log_file=log_file, console=False, force=True)
    logger = get_logger("test.ws1")
    log_event(logger, "ingest.doc", doc="pd", pages=8, duration_s=0.5, rss_mb=120.0, status="ok")
    text = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert LOG_LINE_RE.match(text), text
    assert "event=ingest.doc" in text and "doc=pd" in text and "status=ok" in text


def test_hardware_profile_reports_what_latency_claims_need():
    profile = hardware_profile()
    for key in ("platform", "python", "cpu_count", "ram_gb"):
        assert key in profile
    assert rss_mb() > 0


def test_trace_step_records_a_pdf_page_to_answer_path():
    trace = models.AgentTrace(run_id="run_x", question="What is tSenderResponse?")
    trace.plan = ["type=table | documents=pd | step 1: search_text"]
    trace.steps.append(models.TraceStep(step=1, kind="tool", name="search_text", n_hits=8,
                                        top_score=3.2, duration_s=0.01,
                                        observation="[1] pd §6.6 p.3"))
    trace.answer = "24 ms to 30 ms [pd, R3.2 V1.2, §6.6, p.4, Table 6.8]"
    trace.citations = [models.Citation(doc_id="pd", revision="R3.2 V1.2", section_number="6.6",
                                       page=4, region="Table 6.8")]
    trace.latency_s = 0.42
    data = trace.to_dict()
    assert data["steps"][0]["name"] == "search_text"
    assert data["citations"][0]["region"] == "Table 6.8"
    assert data["latency_s"] == 0.42
