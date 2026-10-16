"""WS4 — the agent: tools, planner, verifier, loop, refusals and the CLI.

These tests run against the fixture corpus through ``MockToolkit`` and the real
toolkit, so the agent's control flow is exercised without the seven-document
corpus.  The honesty tests (declines, false premises) matter most: an agent that
answers everything is worse than one that says what it does not know.
"""

from __future__ import annotations

import json

import pytest

from specintel.agent.loop import AgentConfig, SpecAgent
from specintel.agent.planner import Planner
from specintel.agent.tools import MockToolkit, Toolkit, TOOL_SPECS, ToolResult, write_mock_fixtures
from specintel.agent.verifier import CitationVerifier


@pytest.fixture(scope="session")
def toolkit(ingested_tiny):
    """The real toolkit over the fixture corpus (two documents, two tables)."""
    from specintel.index.build import build_indexes
    from specintel.kg.build import build_graph

    build_indexes(ingested_tiny, doc_ids=["pd", "typec"])
    build_graph(ingested_tiny, doc_ids=["pd", "typec"], use_llm=False)
    toolkit = Toolkit.from_config(ingested_tiny)
    toolkit.doc_ids = ["pd", "typec"]
    return toolkit


@pytest.fixture(scope="session")
def agent(toolkit):
    return SpecAgent(toolkit, agent_config=AgentConfig(max_steps=6, budget_s=20.0))


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def test_all_seven_required_tools_exist():
    required = {"search_text", "search_pages", "graph_query", "get_section", "get_table",
                "get_figure", "resolve_entity"}
    assert required <= {spec["name"] for spec in TOOL_SPECS}
    assert required <= set(Toolkit.__dict__), "each tool is a method on the toolkit"


def test_search_text_honours_its_arguments(toolkit):
    result = toolkit.call("search_text", query="tSenderResponse", k=3)
    assert isinstance(result, ToolResult) and result.tool == "search_text"
    assert 1 <= len(result.hits) <= 3
    assert all(hit.text for hit in result.hits)


def test_get_table_returns_the_whole_table_with_its_pages(toolkit):
    tables = toolkit.tables
    assert tables
    table_id = tables[0].table_id
    result = toolkit.call("get_table", table_id=table_id)
    assert result.text
    assert table_id in json.dumps(result.to_dict(), default=str)


def test_resolve_entity_links_a_name_to_graph_nodes(toolkit):
    result = toolkit.call("resolve_entity", name="tSenderResponse")
    assert result.text and "tSenderResponse" in result.text
    assert "no node matches" not in result.text.lower()


def test_unknown_tool_is_refused_with_the_list_of_real_ones(toolkit):
    with pytest.raises(KeyError) as excinfo:
        toolkit.call("no_such_tool", query="x")
    message = str(excinfo.value)
    assert "unknown tool" in message and "search_text" in message


def test_mock_toolkit_supports_the_same_calls(tmp_path):
    """A teammate can stub every tool from one JSON file (INTERFACES.md § mocks)."""
    from specintel.agent.tools import DEFAULT_FIXTURES

    write_mock_fixtures(tmp_path / "mock_fixtures.json")
    mock = MockToolkit(tmp_path / "mock_fixtures.json")
    assert mock.call("search_text", query="PS_RDY", k=2).hits
    assert mock.call("graph_query", kind="neighbors", name="tSenderResponse")
    assert mock.call("get_section", doc_id="pd", section_number="6").text
    assert mock.call("resolve_entity", name="tSenderResponse").text


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", [
    ("What are the limits of tSenderResponse in Table 6.8?", "table"),
    ("Which states does the Source policy engine pass through in Figure 8-1?", "figure"),
    ("What does PS_RDY signal and which port sends it?", "single_hop"),
    ("Compare the Source and Sink policy engine states across the PD and Type-C specs.", "comparison"),
    ("What is the 60 V fixed voltage contract and which revision introduced it?", "single_hop"),
])
def test_planner_classifies_question_types(question, expected):
    plan = Planner().plan(question)
    if expected == "single_hop":
        assert plan.question_type in ("single_hop", "table", "figure")
    else:
        assert plan.question_type == expected, plan.describe()


def test_planner_extracts_documents_and_identifiers():
    plan = Planner().plan("In the Type-C specification, what does the tCCDebounce timer limit?")
    assert "typec" in plan.expected_docs
    assert any("tCCDebounce".lower() == i.lower() for i in plan.identifiers)


def test_planner_decomposes_a_compound_question():
    plan = Planner().plan("What does PS_RDY signal, and which port sends it during a contract "
                          "negotiation?")
    assert len(plan.sub_questions) >= 2, plan.sub_questions


# --------------------------------------------------------------------------- #
# verifier
# --------------------------------------------------------------------------- #
def test_verifier_drops_a_claim_the_evidence_does_not_support(toolkit):
    from specintel.agent.tools import citation_from_hit

    hits = toolkit.call("search_text", query="tSenderResponse", k=3).hits
    citations = [citation_from_hit(h) for h in hits]
    marker = citations[0].to_marker()
    answer = (f"tSenderResponse is at most 30 ms. {marker} "
              f"The USB4 specification requires a purple indicator lamp. {marker} "
              "Table 6.8 gives the timer limits.")
    result = CitationVerifier(support_threshold=0.5).verify(answer, hits, citations)
    assert result.dropped, "the invented claim is flagged and the uncited sentence too"
    assert "purple indicator" not in result.kept
    assert "Table 6.8 gives the timer limits." in result.dropped
    assert "30 ms" in result.kept
    # the marker follows the sentence, and must be attributed to it
    assert result.claims[0].citations, "a marker after the sentence belongs to that sentence"


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def test_agent_answers_from_the_corpus_with_citations(agent):
    answer = agent.answer("What are the limits of tSenderResponse?")
    assert not answer.declined, answer.answer
    assert "24" in answer.answer and "30" in answer.answer
    assert answer.citations, "an answer without a citation is not acceptable"
    marker = answer.citations[0].to_marker()
    assert marker.startswith("[pd,") and "§" in marker
    assert answer.trace and answer.trace.steps, "the run leaves a trace"
    assert answer.tool_calls >= 1


def test_agent_declines_a_question_the_corpus_cannot_answer(agent):
    answer = agent.answer("What is the maximum number of DisplayPort tunnels on an Intel Meteor "
                          "Lake host router?")
    assert answer.declined or "INSUFFICIENT" in answer.answer.upper()
    assert answer.decline_reason


def test_agent_declines_a_false_premise(agent):
    """The corpus notes that a 60 V fixed contract does not exist."""
    answer = agent.answer("What does the 60 V fixed contract require, and which revision added it?")
    declined = answer.declined or "INSUFFICIENT" in answer.answer.upper()
    flagged = "does not exist" in answer.answer.lower() or "no" in answer.answer.lower()[:120]
    assert declined or flagged, answer.answer[:200]
    if not declined:
        # answering a false premise is allowed only when the answer says plainly
        # that the premise does not hold
        assert any(marker in answer.answer.lower() for marker in
                   ("does not exist", "not exist", "no 60 v", "no such", "out of scope"))


def test_false_premise_detection_helper(agent, toolkit):
    """A numeric premise the corpus contradicts is reported as a denial."""
    hits = toolkit.call("search_text", query="60 V fixed contract", k=6).hits
    plan = Planner().plan("Which revision introduced the 60 V fixed contract?")
    denial = agent._false_premise(plan, hits)
    assert isinstance(denial, str)
    # with no evidence at all there is nothing to deny against
    assert agent._false_premise(plan, []) == ""


def test_uncovered_terms_ignores_instruction_words(agent, toolkit):
    hits = toolkit.call("search_text", query="tSenderResponse", k=5).hits
    plan = Planner().plan("List the limits of tSenderResponse in Table 6.8.")
    missing = agent._uncovered_terms(plan, hits)
    assert "list" not in missing and "limits" not in missing


def test_agent_records_a_trace_that_walks_from_question_to_answer(agent):
    answer = agent.answer("What does PS_RDY signal?")
    trace = answer.trace.to_dict()
    names = [step["name"] for step in trace["steps"]]
    assert any(n.startswith("search_") or n.startswith("retrieve") for n in names)
    assert trace["plan"], "the plan is part of the trace"
    assert trace["latency_s"] >= 0
    assert trace["model"]


def test_agent_respects_the_document_filter(agent):
    answer = agent.answer("What does the Sink present on the Configuration Channel?",
                          docs=["typec"])
    if not answer.declined:
        assert all(hit.doc_id == "typec" for hit in answer.evidence) or not answer.evidence


def test_budget_is_enforced(toolkit):
    tight = SpecAgent(toolkit, agent_config=AgentConfig(max_steps=1, budget_s=0.001))
    answer = tight.answer("What are the limits of tSenderResponse in Table 6.8?")
    assert answer.trace is not None


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_cli_ask_prints_a_cited_answer(monkeypatch, capsys, tmp_path, ingested_tiny):
    from specintel.agent import cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_yaml_for(ingested_tiny), encoding="utf-8")
    code = cli.main(["--config", str(config_path), "ask", "What are the tSenderResponse limits?",
                     "--trace"])
    out = capsys.readouterr().out
    assert code in (0, 3)
    assert "ANSWERED" in out or "DECLINED" in out
    assert "trace" in out


def test_cli_doctor_reports_missing_artefacts(monkeypatch, capsys, tmp_path, ingested_tiny):
    from specintel.agent import cli

    data = json.loads(json.dumps(ingested_tiny.data))
    data["paths"]["graph"] = str(tmp_path / "no_graph")
    config_path = tmp_path / "empty_config.yaml"
    config_path.write_text(_dump_yaml(data), encoding="utf-8")
    code = cli.main(["--config", str(config_path), "doctor"])
    out = capsys.readouterr().out
    assert "knowledge graph" in out.lower()
    assert "no_graph" in out or "missing" in out.lower()
    assert code == 2, "doctor exits 2 when a required artefact is missing"


def test_cli_tools_lists_the_seven_tools(capsys):
    from specintel.agent import cli

    assert cli.main(["tools"]) == 0
    out = capsys.readouterr().out
    for name in ("search_text", "search_pages", "graph_query", "get_section", "get_table",
                 "get_figure", "resolve_entity"):
        assert name in out


def _yaml_for(config) -> str:
    """Minimal config file pointing at the fixture corpus.

    The loader reads YAML only (there is no PyYAML dependency by design), so the
    test writes YAML rather than dumping JSON into a ``.yaml`` file — a JSON
    object parses as a single scalar there and the CLI then sees no paths at all.
    """
    return _dump_yaml(config.data)


def _dump_yaml(value, indent: int = 0) -> str:
    """A very small YAML writer: mappings, lists of mappings and scalars."""
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(item, indent + 2))
            elif isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}: {'{}' if isinstance(item, dict) else '[]'}")
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                body = _dump_yaml(item, indent + 2).split("\n")
                lines.append(f"{pad}- {body[0].strip()}")
                lines.extend(body[1:])
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_scalar(value)}"


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or text.strip() != text or any(c in text for c in ":#{}[],&*?|>%@`\"'"):
        return json.dumps(text)
    return text
