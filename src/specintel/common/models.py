"""The contract: typed data models every workstream exchanges.

These dataclasses are the executable form of ``INTERFACES.md``.  Producers and
consumers are named in the docstrings so a contract change is easy to route.

======================================  ============  ==========================
Contract                                Produced by   Consumed by
======================================  ============  ==========================
:class:`Element`                        WS1           WS2, WS3, WS5
:class:`Table` / :class:`Figure`        WS1           WS2, WS3, WS4
:class:`Chunk`                          WS2           WS4, WS5
:class:`SearchHit`                      WS2           WS4, WS5
:class:`Node` / :class:`Edge`           WS3           WS4, WS5
:class:`GraphQuery`                     WS3 API       WS4
:class:`Citation` / :class:`TraceStep`  WS4           WS5
:class:`GoldQuestion`                   WS5 (authored by all)  WS5
======================================  ============  ==========================

All timestamps are ISO-8601 UTC.  ``bbox`` is ``[x0, y0, x1, y1]`` in PDF
points with the origin at the top-left of the page (PyMuPDF convention).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# enums (kept as plain constants so JSON stays readable)
# --------------------------------------------------------------------------- #
ELEMENT_TYPES = (
    "title",
    "section_heading",
    "paragraph",
    "list_item",
    "table",
    "table_caption",
    "figure",
    "figure_caption",
    "figure_description",
    "note",
    "code",
    "header",
    "footer",
    "toc",
)

EXTRACTORS = (
    "pymupdf",
    "docling",
    "marker",
    "pdfplumber",
    "unlimited-ocr",
    "deepseek-ocr",
    "mock-ocr",
    "synthetic",
)

EXTRACTION_METHODS = ("rule", "llm", "vlm", "ocr", "parser")

NODE_TYPES = (
    "Document",
    "Section",
    "Table",
    "Figure",
    "Message",
    "DataObject",
    "Field",
    "State",
    "Timer",
    "Counter",
    "Parameter",
    "Signal",
    "Pin",
    "Role",
    "TestCase",
    "Check",
    "Assertion",
    "VIFField",
    "PowerProfile",
    "Tech2Phase",
    "Tech2Packet",
    "Router",
    "Adapter",
    "Tunnel",
    "Protocol",
    "ECN",
    "Revision",
)

EDGE_TYPES = (
    "DEFINED_IN",
    "REFERENCES",
    "SENT_BY",
    "RESPONDS_TO",
    "CONTAINS",
    "TRANSITIONS_TO",
    "STARTS_TIMER",
    "STOPS_TIMER",
    "HAS_FIELD",
    "HAS_VALUE",
    "TESTS",
    "VERIFIES",
    "APPLIES_WHEN",
    "TUNNELS",
    "SUPERSEDES",
    "HAS_REVISION",
    "PART_OF",
    "DEPENDS_ON",
    "HAS_STATE",
)

QUESTION_CATEGORIES = {
    "S": "single-hop text",
    "T": "table lookup",
    "F": "figure / diagram",
    "X": "cross-document multi-hop",
    "C": "comparison",
    "G": "graph traversal",
    "V": "revision-aware",
    "U": "unanswerable / false premise",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


class ContractMixin:
    """``to_json`` / ``from_json`` helpers shared by every contract model."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in payload.items() if k in known})  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# WS1 — ingestion
# --------------------------------------------------------------------------- #
@dataclass
class Element(ContractMixin):
    """One typed block of a document (INTERFACES.md § element JSONL).

    Producer: WS1.  Consumers: WS2 (chunking), WS3 (extraction), WS4/WS5 (citations).
    """

    element_id: str
    doc_id: str
    revision: str
    element_type: str
    page: int
    text: str = ""
    markdown: str = ""
    section_number: str = ""
    section_title: str = ""
    section_path: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    image_path: str | None = None
    extractor: str = "pymupdf"
    order: int = 0
    table_id: str | None = None
    figure_id: str | None = None
    caption: str | None = None
    page_count: int = 1          # >1 for stitched tables
    parent_element_id: str | None = None   # WS2 parent–child / WS1 table parts
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def locator(self) -> str:
        """Human citation locator: ``doc §sec p.page``."""
        sec = f" §{self.section_number}" if self.section_number else ""
        return f"{self.doc_id}{sec} p.{self.page}"

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"

    @staticmethod
    def read_jsonl(path: str) -> Iterator["Element"]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield Element.from_dict(json.loads(line))

    @staticmethod
    def write_jsonl(path: str, elements: Iterable["Element"]) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for el in elements:
                fh.write(el.to_jsonl())
                n += 1
        return n


@dataclass
class Table(ContractMixin):
    """A stitched table with page provenance.  Producer WS1, consumers WS2–WS5."""

    table_id: str
    doc_id: str
    revision: str
    number: str                       # e.g. "6.8", "4-27", "Table 3"
    caption: str
    pages: list[int]
    headers: list[list[str]]
    rows: list[list[str]]
    section_number: str = ""
    bbox: list[float] | None = None
    image_path: str | None = None
    element_ids: list[str] = field(default_factory=list)
    stitched: bool = False
    extractor: str = "pymupdf"

    @property
    def markdown(self) -> str:
        """Markdown rendering, repeating header rows for stitched tables."""
        if not self.rows and not self.headers:
            return ""
        out: list[str] = []
        for header in self.headers:
            out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in header) + " |")
            out.append("| " + " | ".join("---" for _ in header) + " |")
        for row in self.rows:
            out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
        return "\n".join(out)

    @property
    def n_cols(self) -> int:
        sizes = [len(r) for r in self.rows] + [len(h) for h in self.headers]
        return max(sizes) if sizes else 0

    def to_text(self) -> str:
        head = f"Table {self.number}" + (f" — {self.caption}" if self.caption else "")
        return f"{head}\n{self.markdown}"


@dataclass
class Figure(ContractMixin):
    """A cropped figure plus optional VLM structure.  Producer WS1."""

    figure_id: str
    doc_id: str
    revision: str
    number: str
    caption: str
    page: int
    image_path: str | None = None
    bbox: list[float] | None = None
    kind: str = "unknown"            # state_machine | timing | block | photo | chart
    description: str = ""            # retrieval text from the VLM / heuristic
    structure: dict[str, Any] | None = None   # {states, transitions} | {signals, events, timers}
    section_number: str = ""
    references: list[str] = field(default_factory=list)   # "see Figure ..." mentions
    extractor: str = "pymupdf"
    hallucination_notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        head = f"Figure {self.number}" + (f" — {self.caption}" if self.caption else "")
        return f"{head}\n{self.description}".strip()


@dataclass
class PageImage(ContractMixin):
    """A rendered page used by the vision path.  Producer WS1, consumer WS2."""

    doc_id: str
    revision: str
    page: int
    image_path: str
    dpi: int = 160
    width: int = 0
    height: int = 0


# --------------------------------------------------------------------------- #
# WS2 — retrieval
# --------------------------------------------------------------------------- #
@dataclass
class Chunk(ContractMixin):
    """A retrievable unit.  Producer WS2, consumers WS4/WS5."""

    chunk_id: str
    doc_id: str
    revision: str
    element_type: str
    text: str
    page: int
    page_end: int = 0
    section_number: str = ""
    section_title: str = ""
    section_path: list[str] = field(default_factory=list)
    element_ids: list[str] = field(default_factory=list)
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = field(default_factory=list)
    table_id: str | None = None
    figure_id: str | None = None
    bbox: list[float] | None = None
    n_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def locator(self) -> str:
        sec = f" §{self.section_number}" if self.section_number else ""
        page = f" p.{self.page}" if self.page_end <= self.page else f" p.{self.page}-{self.page_end}"
        return f"{self.doc_id}{sec}{page}"

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"

    @staticmethod
    def write_jsonl(path: str, chunks: Iterable["Chunk"]) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for ch in chunks:
                fh.write(ch.to_jsonl())
                n += 1
        return n

    @staticmethod
    def read_jsonl(path: str) -> list["Chunk"]:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(Chunk.from_dict(json.loads(line)))
        return out


@dataclass
class SearchHit(ContractMixin):
    """A ranked retrieval result shared by text, page and graph tools."""

    chunk_id: str
    score: float
    doc_id: str
    page: int
    page_end: int = 0
    section_number: str = ""
    element_type: str = ""
    text: str = ""
    source: str = "text"                  # text | pages | graph | table | figure | fused
    bbox: list[float] | None = None
    element_ids: list[str] = field(default_factory=list)
    table_id: str | None = None
    figure_id: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def locator(self) -> str:
        sec = f" §{self.section_number}" if self.section_number else ""
        pages = f" p.{self.page}" if self.page_end <= self.page else f" p.{self.page}-{self.page_end}"
        return f"{self.doc_id}{sec}{pages}"

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"


# --------------------------------------------------------------------------- #
# WS3 — knowledge graph
# --------------------------------------------------------------------------- #
@dataclass
class Node(ContractMixin):
    """A graph node.  Producer WS3, consumers WS4/WS5."""

    node_id: str
    node_type: str
    name: str
    canonical_name: str = ""
    doc_id: str = ""
    revision: str = ""
    section_number: str = ""
    page: int = 0
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    element_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.canonical_name:
            self.canonical_name = self.name

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"


@dataclass
class Edge(ContractMixin):
    """A provenance-carrying relation.  Producer WS3, consumers WS4/WS5.

    ``extraction_method`` is one of rule | llm | vlm; ``confidence`` in [0, 1].
    ``stale`` marks a cross-document reference that points at an older revision
    than the target document's current one (assignment trap: test specs are
    older than the base specs they cite).
    """

    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str
    doc_id: str = ""
    revision: str = ""
    section_number: str = ""
    page: int = 0
    element_id: str = ""
    extraction_method: str = "rule"
    confidence: float = 1.0
    evidence: str = ""
    stale: bool = False
    properties: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"


@dataclass
class GraphQuery(ContractMixin):
    """The graph API surface exposed to the agent (WS3 → WS4)."""

    kind: str                          # neighbors | path | lookup | subgraph | stats
    name: str = ""
    node_id: str = ""
    src: str = ""
    dst: str = ""
    edge_types: Sequence[str] = field(default_factory=tuple)
    node_types: Sequence[str] = field(default_factory=tuple)
    direction: str = "both"            # in | out | both
    depth: int = 1
    limit: int = 25

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"


@dataclass
class GraphResult(ContractMixin):
    """Result of a :class:`GraphQuery`: hits plus provenance for citation."""

    query: GraphQuery
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_text(self, limit: int = 40) -> str:
        """Render as compact text for the LLM prompt."""
        lines = []
        for e in self.edges[:limit]:
            lines.append(f"{e.src_id} -[{e.edge_type}]-> {e.dst_id}  ({e.doc_id} §{e.section_number} p.{e.page})")
        if not lines:
            lines = [f"{n.node_type}:{n.name} ({n.doc_id} p.{n.page})" for n in self.nodes[:limit]]
        return "\n".join(lines) if lines else "(no graph results)"


# --------------------------------------------------------------------------- #
# WS4 — agent
# --------------------------------------------------------------------------- #
@dataclass
class Citation(ContractMixin):
    """``[doc, revision, §section, p.page, region]`` — WS4 output, WS5 checks."""

    doc_id: str
    revision: str
    section_number: str
    page: int
    region: str = ""                  # table/figure/paragraph identifier
    page_end: int = 0
    element_ids: list[str] = field(default_factory=list)
    quote: str = ""
    verified: bool | None = None
    verification_note: str = ""

    def to_marker(self) -> str:
        sec = f"§{self.section_number}" if self.section_number else "§—"
        region = f", {self.region}" if self.region else ""
        rev = f", {self.revision}" if self.revision else ""
        return f"[{self.doc_id}{rev}, {sec}, p.{self.page}{region}]"

    @property
    def locator(self) -> str:
        return self.to_marker()


@dataclass
class TraceStep(ContractMixin):
    """One agent step: plan → tool call → observation → reflection."""

    step: int
    kind: str                         # plan | tool | reflect | answer | decline | note
    name: str = ""                    # tool name or plan label
    args: dict[str, Any] = field(default_factory=dict)
    n_hits: int = 0
    top_score: float | None = None
    observation: str = ""
    duration_s: float = 0.0
    tokens: int = 0
    decision: str = ""
    citations: list[Citation] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentTrace(ContractMixin):
    """A full run: question → plan → tool calls → evidence → answer."""

    run_id: str
    question: str
    question_category: str = ""
    plan: list[str] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    declined: bool = False
    decline_reason: str = ""
    confidence: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    config_name: str = "F"
    model: str = "extractive"
    started_at: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"

    @staticmethod
    def write_jsonl(path: str, traces: Iterable["AgentTrace"]) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for t in traces:
                fh.write(t.to_jsonl())
                n += 1
        return n


@dataclass
class ToolResult(ContractMixin):
    """What every agent tool returns (INTERFACES.md § tool signatures)."""

    tool: str
    hits: list[SearchHit] = field(default_factory=list)
    text: str = ""
    graph: GraphResult | None = None
    table: Table | None = None
    figure: Figure | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def top_score(self) -> float | None:
        return max((h.score for h in self.hits), default=None)


# --------------------------------------------------------------------------- #
# WS5 — evaluation
# --------------------------------------------------------------------------- #
@dataclass
class GoldQuestion(ContractMixin):
    """One gold question.  Authored by every member, verified by a second."""

    qid: str
    question: str
    category: str                     # S T F X C G V U
    gold_answer: str
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    author: str = ""
    verifier: str = ""
    content_type: str = "prose"       # prose | bitfield_table | multipage_table |
                                      # state_machine | timing | test_procedure
    docs: list[str] = field(default_factory=list)
    answerable: bool = True
    notes: str = ""
    gold_citations: list[str] = field(default_factory=list)
    #: "dev" questions may be used while tuning; "held_out" questions are scored
    #: once (the assignment's ~20 hidden reviewer questions)
    split: str = "dev"

    @property
    def category_label(self) -> str:
        return QUESTION_CATEGORIES.get(self.category, self.category)

    @staticmethod
    def read_jsonl(path: str) -> list["GoldQuestion"]:
        out: list[GoldQuestion] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("//"):
                    out.append(GoldQuestion.from_dict(json.loads(line)))
        return out

    @staticmethod
    def write_jsonl(path: str, questions: Sequence["GoldQuestion"]) -> int:
        with open(path, "w", encoding="utf-8") as fh:
            for q in questions:
                fh.write(q.to_json() + "\n")
        return len(questions)


@dataclass
class QuestionResult(ContractMixin):
    """One graded question: retrieval metrics + answer metrics + provenance."""

    qid: str
    category: str
    content_type: str
    answerable: bool
    config_name: str = "F"
    answer: str = ""
    declined: bool = False
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, Any] = field(default_factory=dict)
    failure_stage: str = ""
    notes: str = ""
    latency_s: float = 0.0
    tool_calls: int = 0
    tokens: int = 0

    def to_jsonl(self) -> str:
        return self.to_json() + "\n"


@dataclass
class EvalSummary(ContractMixin):
    """Aggregate metrics for one config, per category and overall."""

    config_name: str
    description: str = ""
    n_questions: int = 0
    overall: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_content_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_approach: dict[str, dict[str, float]] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
