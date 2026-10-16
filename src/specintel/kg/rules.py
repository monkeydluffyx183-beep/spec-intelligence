"""WS3 — pass 1: deterministic entity and relation extraction.

Rules are used first because most of what this corpus contains is strictly
patterned: test ids (``TEST.PD.PROT.SRC.3``, ``[TEST.PD.PROT.SRC.3#2]``,
``TD 4.2.4`` and ``[TD.4.2.4.V.3]``), timers (``tSenderResponse``), counters
(``nCapsCount``), states (``PE_SRC_Ready``, ``Unattached.SRC``), parameters
(``vRd-1.5``), section and table references, and bit-field tables whose rows are
"B31..30 | PDO Type | 00b = Fixed Supply".  A regex is more precise and far
cheaper than a model here, and every extraction keeps its page and section.

What the rules produce
----------------------
* ``Document`` / ``Revision`` / ``Section`` / ``Table`` / ``Figure`` backbone;
* ``Message`` / ``DataObject`` / ``Field`` / ``Timer`` / ``Counter`` / ``State`` /
  ``Parameter`` / ``Signal`` / ``Pin`` / ``Role`` / ``PowerProfile`` /
  ``Tech2Phase`` / ``Tech2Packet`` / ``Router`` / ``Adapter`` / ``Tunnel`` /
  ``TestCase`` / ``Check`` / ``Assertion`` / ``ECN`` entities;
* relations ``DEFINED_IN``, ``PART_OF``, ``REFERENCES`` (with stale detection),
  ``SENT_BY``, ``RESPONDS_TO``, ``HAS_FIELD``, ``HAS_VALUE``, ``TRANSITIONS_TO``,
  ``STARTS_TIMER``, ``STOPS_TIMER``, ``TESTS``, ``VERIFIES``, ``DEPENDS_ON``,
  ``TUNNELS``, ``HAS_STATE``, ``SUPERSEDES``.

The output is validated by :mod:`specintel.kg.schema`; anything rejected is
written to ``rejections.jsonl`` with its reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from specintel.common.models import Edge, Element, Figure, Node, Table, new_id

# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #
TIMER_RE = re.compile(r"\bt[A-Z][A-Za-z0-9]{2,}\b")
COUNTER_RE = re.compile(r"\bn[A-Z][A-Za-z0-9]{2,}\b")
STATE_RE = re.compile(r"\b(?:PE|PRL|PE_SNK|PE_SRC|PE_CBL)_[A-Za-z_]{2,}\b|"
                      r"\b[A-Z][A-Za-z]+\.(?:SRC|SNK|DBGACC|AUDIOACC)\b|"
                      r"\b(?:Try|AttachWait|Attached|Unattached|TryWait)\.[A-Z]+\b|"
                      r"\b[A-Z][A-Za-z]+\.(?:SRC|SNK)\b")
TEST_ID_RE = re.compile(r"\b(?:TEST\.[A-Z0-9]+(?:\.[A-Z0-9]+)+|COMMON\.CHECK\.[A-Z0-9.]+|"
                        r"TD\s+\d+(?:\.\d+)+|TD\.\d+(?:\.\d+)+)\b")
CHECK_RE = re.compile(r"\[(?P<id>(?:TEST\.[A-Z0-9.]+|TD\.\d+(?:\.[A-Z0-9]+)+|COMMON\.CHECK\.[A-Z0-9.]+))\]")
ASSERTION_RE = re.compile(r"\bR-[A-Z]{2,}-\d{3,}\b")
PARAM_RE = re.compile(r"\bvRd-(?:USB|1\.5|3\.0)\b|\b\d+(?:\.\d+)?\s?(?:kΩ|Ω)\b")
SIGNAL_RE = re.compile(r"\b(?:VBUS|VCONN|VSAFE5V|VBUSON)\b")
PIN_RE = re.compile(r"\b(?:CC1|CC2|A2|A3|B2|B3)\b")
ROLE_RE = re.compile(r"\b(?:Source|Sink|DRP|eMarker|Power Transmitter|Power Receiver)\b")
PROFILE_RE = re.compile(r"\b(?:BPP|EPP|MPP)\b")
PHASE_RE = re.compile(r"\b(?:Ping|Configuration|Negotiation|Power Transfer)\b")
TECH2_PACKET_RE = re.compile(r"\b(?:Signal Strength|Control Error|End Power Transfer|End Negotiation) packet\b")
ROUTER_RE = re.compile(r"\b(?:Host Router|Device Router)\b")
ADAPTER_RE = re.compile(r"\b(?:Lane Adapter|USB3 Adapter|DP Adapter|PCIe Adapter|H2H Adapter|Protocol Adapter)\b")
ECN_RE = re.compile(r"\bECN[\s_]+([A-Za-z0-9_]+)")
VIF_FIELD_RE = re.compile(r"\b(?:PD_Port_Type|Type_C_State_Machine|[A-Z][A-Za-z]*_[A-Z][A-Za-z_]*VIF)\b")
SECTION_REF_RE = re.compile(r"(?:§|Section|section|Sections|sections)\s*([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)")
TABLE_REF_RE = re.compile(r"\bTable\s+([0-9A-Z]+(?:[.\-][0-9A-Z]+)*)")
FIGURE_REF_RE = re.compile(r"\bFigure\s+([0-9A-Z]+(?:[.\-][0-9A-Z]+)*)")
CHAPTER_REF_RE = re.compile(r"\b(?:Chapter|chapter|Appendix|Appendix\s)\s*([0-9]+|[A-Z])\b")
BARE_SECTION_RE = re.compile(r"\b([0-9]{1,2}\.[0-9]{1,2}(?:\.[0-9]{1,2}){0,2})\b")
NUMERIC_LIMIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|µs|us|V|A|W|%|kΩ|Ω)")
REVISION_RE = re.compile(r"\b(?:R[0-9]\.[0-9]|Release\s+[0-9]\.[0-9]|Version\s+[0-9]\.[0-9])\b(?:\s*V[0-9.]+)?")
DOC_NAME_RE = re.compile(r"\b(USB Power Delivery|USB PD|USB Type-C|Type-C|USB4|Tech 2|Shared Capacity)\b", re.I)

DOC_ALIASES = {
    "usb power delivery specification": "pd",
    "usb power delivery": "pd",
    "usb pd": "pd",
    "usb type-c": "typec",
    "type-c": "typec",
    "usb4": "usb4",
    "tech 2": "tech2",
    "shared capacity": "scc_cts",
}

MESSAGE_LEXICON = {
    "GoodCRC", "GotoMin", "Accept", "Reject", "Ping", "PS_RDY", "Get_Source_Cap", "Get_Sink_Cap",
    "DR_Swap", "PR_Swap", "VCONN_Swap", "Wait", "Soft_Reset", "Not_Supported", "Get_Source_Info",
    "Get_Status", "Source_Capabilities", "Request", "BIST", "Sink_Capabilities", "Battery_Status",
    "Alert", "Get_Country_Info", "Vendor_Defined", "Enter_USB", "EPR_Request", "EPR_Mode",
    "EPR_Source_Capabilities", "EPR_Sink_Capabilities", "Source_Info", "Status", "Get_Battery_Cap",
    "Firmware_Update", "Security_Request",
}
DATA_OBJECT_RE = re.compile(r"\b(?:Fixed Supply|Battery|Variable Supply|Programmable Power Supply|"
                            r"Adjustable Voltage Supply)\s*(?:Power Data Object|APDO|PDO)?\b")
BIT_RANGE_RE = re.compile(r"\bB\d{1,2}(?:\.\.\d{1,2})?\b")


@dataclass
class ExtractionResult:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    merges: list[dict[str, Any]] = field(default_factory=list)
    _seen_edges: dict[tuple[Any, ...], Edge] = field(default_factory=dict, repr=False)

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge, collapsing exact duplicates.

        The same relation is often asserted by more than one rule pass over the
        same element (for example a message mentioned twice in one paragraph).
        Repeats with identical provenance are merged into one edge that records
        how often it was seen, which keeps triple precision measurable: a
        duplicated structural edge is a defect, not evidence.
        """
        key = (edge.src_id, edge.edge_type, edge.dst_id, edge.doc_id, edge.page, edge.element_id)
        existing = self._seen_edges.get(key)
        if existing is not None:
            existing.properties["occurrences"] = int(existing.properties.get("occurrences", 1)) + 1
            existing.confidence = max(existing.confidence, edge.confidence)
            if not existing.evidence and edge.evidence:
                existing.evidence = edge.evidence
            return existing
        self._seen_edges[key] = edge
        self.edges.append(edge)
        return edge


@dataclass
class ExtractionContext:
    """Everything pass 1 needs about one document."""

    doc_id: str
    revision: str
    revision_key: str = "0"
    title: str = ""
    elements: list[Element] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    other_docs: dict[str, dict[str, Any]] = field(default_factory=dict)   # doc_id -> meta


# --------------------------------------------------------------------------- #
# extractor
# --------------------------------------------------------------------------- #
class RuleExtractor:
    """Deterministic extraction over one document."""

    def __init__(self, *, max_entities_per_type: int = 4000) -> None:
        self.max_entities_per_type = max_entities_per_type

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def node_id(node_type: str, name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().lower()).strip("_")
        return f"{node_type}:{slug}"

    def _node(self, result: ExtractionResult, node_type: str, name: str, *, doc_id: str, revision: str,
              section: str = "", page: int = 0, element_ids: Sequence[str] = (), **properties: Any) -> Node:
        # structural nodes must always be able to answer "which table/section is
        # this?" — the number is derived when the caller did not pass it
        if node_type == "Section" and "number" not in properties:
            properties["number"] = _section_number_of(name) or section
        if node_type == "Table" and "number" not in properties:
            properties["number"] = _trailing_number(name)
        if node_type == "Figure" and "number" not in properties:
            properties["number"] = _trailing_number(name)
        node_id = self.node_id(node_type, name)
        existing = next((n for n in result.nodes if n.node_id == node_id), None)
        if existing is not None:
            if page and not existing.page:
                existing.page = page
            if section and not existing.section_number:
                existing.section_number = section
            if element_ids:
                existing.element_ids = list(dict.fromkeys([*existing.element_ids, *element_ids]))
            if properties:
                existing.properties.update({k: v for k, v in properties.items() if v is not None})
            return existing
        node = Node(node_id=node_id, node_type=node_type, name=name, canonical_name=name, doc_id=doc_id,
                    revision=revision, section_number=section, page=page, element_ids=list(element_ids),
                    properties={k: v for k, v in properties.items() if v is not None})
        return result.add_node(node)

    def _edge(self, result: ExtractionResult, src: Node | str, dst: Node | str, edge_type: str, *,
              element: Element | None = None, table: Table | None = None, figure: Figure | None = None,
              method: str = "rule", confidence: float = 1.0, evidence: str = "", stale: bool = False,
              doc_id: str = "", section_number: str = "", page: int = 0, **properties: Any) -> Edge:
        src_id = src.node_id if isinstance(src, Node) else str(src)
        dst_id = dst.node_id if isinstance(dst, Node) else str(dst)
        anchor = element or table or figure
        doc_id = doc_id or getattr(anchor, "doc_id", "") or getattr(src, "doc_id", "") \
            or getattr(dst, "doc_id", "")
        section = section_number or getattr(getattr(anchor, "section_number", None), "__str__", lambda: "")()
        page = int(page or getattr(anchor, "page", 0) or (table.pages[0] if table and table.pages else 0) or 0)
        element_id = (getattr(anchor, "element_id", "") or getattr(anchor, "figure_id", "")
                      or getattr(anchor, "table_id", "") or "")
        if not element_id and isinstance(anchor, Figure):
            element_id = f"{doc_id}:figure:{anchor.number or anchor.figure_id}"
        if not element_id and isinstance(anchor, Table):
            element_id = f"{doc_id}:table:{anchor.number or anchor.table_id}"
        if not section and isinstance(anchor, Figure):
            section = anchor.section_number or ""
        edge = Edge(edge_id=new_id("e"), src_id=src_id, dst_id=dst_id, edge_type=edge_type, doc_id=doc_id,
                    revision=getattr(anchor, "revision", ""), section_number=section or "", page=page,
                    element_id=element_id, extraction_method=method, confidence=confidence,
                    evidence=evidence[:400], stale=stale,
                    properties={k: v for k, v in properties.items() if v is not None})
        return result.add_edge(edge)

    # -- main -------------------------------------------------------------- #
    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        result = ExtractionResult()
        doc = self._document(ctx, result)
        sections = self._sections(ctx, result, doc)
        tables = self._tables(ctx, result, doc, sections)
        figures = self._figures(ctx, result, doc, sections)
        self._messages(ctx, result, doc, sections, tables)
        self._timers(ctx, result, doc, sections, tables)
        self._states(ctx, result, doc, sections, figures)
        self._data_objects(ctx, result, doc, sections, tables)
        self._tests(ctx, result, doc, sections, tables)
        self._domain_entities(ctx, result, doc, sections)
        self._revision_history(ctx, result, doc, sections, tables)
        self._cross_references(ctx, result, doc, sections, tables, figures)
        self._tunnels(ctx, result, doc, sections, tables)
        return result

    # -- backbone ---------------------------------------------------------- #
    def _document(self, ctx: ExtractionContext, result: ExtractionResult) -> Node:
        doc = self._node(result, "Document", ctx.doc_id, doc_id=ctx.doc_id, revision=ctx.revision,
                         title=ctx.title or ctx.doc_id, revision_key=ctx.revision_key)
        if ctx.title:
            doc.canonical_name = ctx.title
            doc.aliases = list(dict.fromkeys([*doc.aliases, ctx.title, ctx.doc_id]))
        rev = self._node(result, "Revision", f"{ctx.doc_id}:{ctx.revision}", doc_id=ctx.doc_id,
                         revision=ctx.revision, key=ctx.revision_key)
        self._edge(result, doc, rev, "HAS_REVISION", method="rule", confidence=1.0,
                   doc_id=ctx.doc_id, page=1)
        return doc

    def _sections(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node) -> dict[str, Node]:
        sections: dict[str, Node] = {}
        for element in ctx.elements:
            if element.element_type != "section_heading" or not element.section_number:
                continue
            node = self._node(result, "Section", f"{ctx.doc_id}:§{element.section_number}", doc_id=ctx.doc_id,
                              revision=ctx.revision, section=element.section_number, page=element.page,
                              element_ids=[element.element_id], number=element.section_number,
                              title=element.section_title, path=element.section_path)
            node.canonical_name = f"§{element.section_number} {element.section_title}".strip()
            sections.setdefault(element.section_number, node)
            self._edge(result, node, doc, "PART_OF", element=element)
        # hierarchy: a section belongs to its parent section
        for number, node in sections.items():
            parent = ".".join(number.split(".")[:-1])
            while parent and parent not in sections:
                parent = ".".join(parent.split(".")[:-1])
            if parent in sections:
                self._edge(result, node, sections[parent], "PART_OF", element=ctx.elements[0] if ctx.elements else None,
                           confidence=0.9)
        return sections

    def _tables(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                sections: dict[str, Node]) -> dict[str, Node]:
        out: dict[str, Node] = {}
        for table in ctx.tables:
            key = f"{ctx.doc_id}:table:{table.number or table.table_id}"
            node = self._node(result, "Table", key, doc_id=ctx.doc_id, revision=ctx.revision,
                              section=table.section_number, page=table.pages[0] if table.pages else 0,
                              number=table.number, caption=table.caption, pages=table.pages,
                              stitched=table.stitched, rows=len(table.rows), cols=table.n_cols)
            node.canonical_name = f"Table {table.number}" if table.number else table.caption[:60]
            out[table.table_id] = node
            section = sections.get(table.section_number)
            if section is not None:
                self._edge(result, node, section, "PART_OF", table=table)
            self._edge(result, node, doc, "PART_OF", table=table, confidence=0.9)
        return out

    def _figures(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                 sections: dict[str, Node]) -> dict[str, Node]:
        out: dict[str, Node] = {}
        for figure in ctx.figures:
            key = f"{ctx.doc_id}:figure:{figure.number or figure.figure_id}"
            node = self._node(result, "Figure", key, doc_id=ctx.doc_id, revision=ctx.revision,
                              section=figure.section_number, page=figure.page, number=figure.number,
                              caption=figure.caption, kind=figure.kind,
                              image_path=figure.image_path, description=figure.description[:400])
            node.canonical_name = f"Figure {figure.number}" if figure.number else figure.caption[:60]
            out[figure.figure_id] = node
            section = sections.get(figure.section_number)
            if section is not None:
                self._edge(result, node, section, "PART_OF", figure=figure)
            self._edge(result, node, doc, "PART_OF", figure=figure, confidence=0.9)
            # the state machine inside the diagram becomes its own nodes
            states = (figure.structure or {}).get("states") or []
            for state_name in states:
                state = self._node(result, "State", state_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                   page=figure.page, section=figure.section_number,
                                   diagram=figure.number or figure.figure_id)
                self._edge(result, node, state, "HAS_STATE", figure=figure, confidence=0.8)
            for transition in (figure.structure or {}).get("transitions") or []:
                src = self._node(result, "State", str(transition.get("from", "")), doc_id=ctx.doc_id,
                                 revision=ctx.revision, page=figure.page, section=figure.section_number)
                dst = self._node(result, "State", str(transition.get("to", "")), doc_id=ctx.doc_id,
                                 revision=ctx.revision, page=figure.page, section=figure.section_number)
                if not src.name or not dst.name:
                    continue
                self._edge(result, src, dst, "TRANSITIONS_TO", figure=figure, method="vlm", confidence=0.75,
                           trigger=transition.get("trigger", ""),
                           actions=", ".join(transition.get("actions", []) or []),
                           evidence=f"Figure {figure.number}: {src.name} -> {dst.name}")
                for action in transition.get("actions", []) or []:
                    for timer in TIMER_RE.findall(str(action)):
                        timer_node = self._node(result, "Timer", timer, doc_id=ctx.doc_id, revision=ctx.revision,
                                                page=figure.page, section=figure.section_number)
                        self._edge(result, src, timer_node, "STARTS_TIMER", figure=figure, method="vlm",
                                   confidence=0.7, condition=str(transition.get("trigger", "")))
        return out

    # -- entities ---------------------------------------------------------- #
    def _iter_text(self, ctx: ExtractionContext) -> Iterator[Element]:
        for element in ctx.elements:
            if element.element_type in ("header", "footer", "toc"):
                continue
            yield element

    def _messages(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                  sections: dict[str, Node], tables: dict[str, Node]) -> None:
        found: dict[str, Node] = {}
        # 1) message tables give the authoritative names, senders and purposes
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header or "message" not in header[0]:
                continue
            name_col = 0
            sent_col = header.index("sent by") if "sent by" in header else None
            purpose_col = header.index("purpose") if "purpose" in header else None
            type_col = header.index("type") if "type" in header else None
            for row in table.rows:
                if not row or not row[name_col].strip():
                    continue
                name = row[name_col].strip()
                if name.lower() in ("message", "packet"):
                    continue
                node = self._node(result, "Message", name, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=table.section_number, page=table.pages[0] if table.pages else 0,
                                  message_type=row[type_col] if type_col is not None and type_col < len(row) else None,
                                  purpose=row[purpose_col] if purpose_col is not None and purpose_col < len(row) else None,
                                  family="table")
                found[name] = node
                table_node = tables.get(table.table_id)
                if table_node is not None:
                    self._edge(result, node, table_node, "DEFINED_IN", table=table, confidence=1.0)
                if sent_col is not None and sent_col < len(row):
                    for role_name in _roles_in(row[sent_col]):
                        role = self._node(result, "Role", role_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                          page=table.pages[0] if table.pages else 0)
                        self._edge(result, node, role, "SENT_BY", table=table, confidence=1.0,
                                   direction=row[sent_col])
        # 2) prose mentions
        for element in self._iter_text(ctx):
            for name in MESSAGE_LEXICON:
                if not re.search(rf"\b{re.escape(name)}\b", element.text):
                    continue
                node = found.get(name) or self._node(result, "Message", name, doc_id=ctx.doc_id,
                                                     revision=ctx.revision, section=element.section_number,
                                                     page=element.page)
                found.setdefault(name, node)
                if node.doc_id == ctx.doc_id and not node.section_number:
                    node.section_number = element.section_number
                section = sections.get(element.section_number)
                if section is not None:
                    self._edge(result, node, section, "DEFINED_IN", element=element, confidence=0.8)
            self._response_edges(element, result, found, ctx)
        # 3) senders stated in prose ("the Source sends PS_RDY")
        for element in self._iter_text(ctx):
            for match in re.finditer(r"\b(?:the\s+)?(Source|Sink)\s+(?:sends?|transmits?)\s+([A-Z][A-Za-z_]+)\b",
                                     element.text):
                role_name, message_name = match.group(1), match.group(2)
                if message_name not in found and message_name not in MESSAGE_LEXICON:
                    continue
                message = found.get(message_name) or self._node(result, "Message", message_name,
                                                                doc_id=ctx.doc_id, revision=ctx.revision,
                                                                section=element.section_number, page=element.page)
                role = self._node(result, "Role", role_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=element.section_number, page=element.page)
                self._edge(result, message, role, "SENT_BY", element=element, confidence=0.85,
                           evidence=match.group(0))

    def _response_edges(self, element: Element, result: ExtractionResult, found: dict[str, Node],
                        ctx: ExtractionContext) -> None:
        """Link "X is sent in response to Y" style statements."""
        for sentence in re.split(r"(?<=[.;])\s+", element.text):
            markers = ("in response to", "answers", "answer with", "responds with", "in reply to",
                       "respond to")
            if not any(m in sentence.lower() for m in markers):
                continue
            present = [n for n in found if re.search(rf"\b{re.escape(n)}\b", sentence)]
            if len(present) < 2:
                continue
            src = found[present[0]]
            dst = found[present[1]]
            if src.node_id == dst.node_id:
                continue
            self._edge(result, src, dst, "RESPONDS_TO", element=element, confidence=0.6,
                       evidence=sentence[:200])

    def _timers(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                sections: dict[str, Node], tables: dict[str, Node]) -> None:
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header:
                continue
            first = header[0]
            is_timer_table = first in ("timer", "counter") or "min" in header and "max" in header
            if not is_timer_table:
                continue
            node_type = "Counter" if first.startswith("counter") else "Timer"
            table_node = tables.get(table.table_id)
            min_col = header.index("min") if "min" in header else None
            max_col = header.index("max") if "max" in header else None
            unit_col = header.index("unit") if "unit" in header else None
            def_col = header.index("definition") if "definition" in header else None
            for row in table.rows:
                if not row or not row[0].strip():
                    continue
                name = row[0].strip()
                if name.lower() in ("timer", "counter") or name.startswith("Table"):
                    continue
                node = self._node(result, node_type, name, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=table.section_number, page=table.pages[0] if table.pages else 0,
                                  min=_cell(row, min_col), max=_cell(row, max_col), unit=_cell(row, unit_col),
                                  definition=_cell(row, def_col))
                if table_node is not None:
                    self._edge(result, node, table_node, "DEFINED_IN", table=table, confidence=1.0)
        # prose mentions of timers/counters without a table row
        for element in self._iter_text(ctx):
            for pattern, node_type in ((TIMER_RE, "Timer"), (COUNTER_RE, "Counter")):
                for name in pattern.findall(element.text):
                    node = self._node(result, node_type, name, doc_id=ctx.doc_id, revision=ctx.revision,
                                      section=element.section_number, page=element.page)
                    section = sections.get(element.section_number)
                    if section is not None:
                        self._edge(result, node, section, "DEFINED_IN", element=element, confidence=0.6)

    def _states(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node, sections: dict[str, Node],
                figures: dict[str, Node]) -> None:
        for element in self._iter_text(ctx):
            for name in set(STATE_RE.findall(element.text)):
                node = self._node(result, "State", name, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=element.section_number, page=element.page)
                section = sections.get(element.section_number)
                if section is not None:
                    self._edge(result, node, section, "DEFINED_IN", element=element, confidence=0.6)
            # "state X starts/stops timer tY"
            for match in re.finditer(r"\b(PE_[A-Za-z_]+|PRL_[A-Za-z_]+|PE_SNK_[A-Za-z_]+)\b[^.]{0,80}?"
                                     r"\b(start|started|starts|stop|stopped|stops)\w*\s+(t[A-Z][A-Za-z0-9]+)",
                                     element.text):
                state_name, verb, timer_name = match.group(1), match.group(2), match.group(3)
                state = self._node(result, "State", state_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                   section=element.section_number, page=element.page)
                timer = self._node(result, "Timer", timer_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                   section=element.section_number, page=element.page)
                relation = "STARTS_TIMER" if verb.startswith("start") else "STOPS_TIMER"
                self._edge(result, state, timer, relation, element=element, confidence=0.75,
                           evidence=match.group(0)[:200])

    def _data_objects(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                      sections: dict[str, Node], tables: dict[str, Node]) -> None:
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header or not header[0].startswith("bit"):
                continue
            caption = table.caption or ""
            name = re.sub(r"\s+bit field[s]?$", "", caption, flags=re.I).strip() or f"Table {table.number}"
            data_object = self._node(result, "DataObject", name, doc_id=ctx.doc_id, revision=ctx.revision,
                                     section=table.section_number, page=table.pages[0] if table.pages else 0,
                                     table=table.number, caption=caption)
            table_node = tables.get(table.table_id)
            if table_node is not None:
                self._edge(result, data_object, table_node, "DEFINED_IN", table=table, confidence=1.0)
            for row in table.rows:
                bits = row[0].strip() if row and row[0].strip() else ""
                if not BIT_RANGE_RE.fullmatch(bits.replace(" ", "")):
                    continue
                field_name = row[1].strip() if len(row) > 1 else ""
                if not field_name or field_name.lower() in ("field", "reserved"):
                    field_name = f"{field_name or 'Reserved'} {bits}".strip()
                field_node = self._node(result, "Field", f"{name}:{field_name}", doc_id=ctx.doc_id,
                                        revision=ctx.revision, section=table.section_number,
                                        page=table.pages[0] if table.pages else 0, bits=bits,
                                        data_object=name, description=(row[2] if len(row) > 2 else ""))
                field_node.canonical_name = field_name
                self._edge(result, data_object, field_node, "HAS_FIELD", table=table, confidence=1.0)
                for numbers in NUMERIC_LIMIT_RE.finditer(" ".join(row[2:])):
                    # a limit is a parameter of the field, not a self reference:
                    # "Field:X -[HAS_VALUE]-> Parameter:X = 21 V"
                    value, unit = numbers.group(1), numbers.group(2)
                    parameter = self._node(result, "Parameter", f"{field_name} = {value} {unit}".strip(),
                                           doc_id=ctx.doc_id, revision=ctx.revision,
                                           section=table.section_number,
                                           page=table.pages[0] if table.pages else 0,
                                           value=value, unit=unit, field=field_name)
                    parameter.canonical_name = f"{field_name} = {value} {unit}".strip()
                    self._edge(result, field_node, parameter, "HAS_VALUE", table=table, confidence=0.6,
                               value=value, unit=unit)
        # prose mentions of PDO types ("Fixed Supply PDO", "PPS APDO")
        for element in self._iter_text(ctx):
            for match in re.finditer(r"\b(Fixed Supply PDO|Battery PDO|Variable Supply PDO|"
                                     r"Programmable Power Supply APDO|PPS APDO|AVS APDO|"
                                     r"EPR Fixed Supply PDO|Fixed Supply Power Data Object)\b", element.text):
                name = match.group(1)
                node = self._node(result, "DataObject", name, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=element.section_number, page=element.page)
                section = sections.get(element.section_number)
                if section is not None:
                    self._edge(result, node, section, "DEFINED_IN", element=element, confidence=0.7)

    def _tests(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node, sections: dict[str, Node],
               tables: dict[str, Node]) -> None:
        test_nodes: dict[str, Node] = {}
        for element in self._iter_text(ctx):
            for match in TEST_ID_RE.finditer(element.text):
                raw = re.sub(r"\s+", " ", match.group(0)).strip()
                node = test_nodes.get(raw) or self._node(result, "TestCase", raw, doc_id=ctx.doc_id,
                                                         revision=ctx.revision, section=element.section_number,
                                                         page=element.page, area=_test_area(raw))
                test_nodes.setdefault(raw, node)
            for match in CHECK_RE.finditer(element.text):
                check_id = match.group("id")
                parent_id = re.sub(r"#\d+$|\.V\.\d+$|\.IDX\.\d+$", "", check_id)
                check = self._node(result, "Check", check_id, doc_id=ctx.doc_id, revision=ctx.revision,
                                   section=element.section_number, page=element.page, test_case=parent_id)
                parent = test_nodes.get(parent_id) or self._node(result, "TestCase", parent_id, doc_id=ctx.doc_id,
                                                                 revision=ctx.revision, page=element.page,
                                                                 area=_test_area(parent_id))
                test_nodes.setdefault(parent_id, parent)
                self._edge(result, check, parent, "PART_OF", element=element, confidence=1.0)
            for name in ASSERTION_RE.findall(element.text):
                self._node(result, "Assertion", name, doc_id=ctx.doc_id, revision=ctx.revision,
                           section=element.section_number, page=element.page, kind="requirement_id")
            for name in VIF_FIELD_RE.findall(element.text):
                self._node(result, "VIFField", name, doc_id=ctx.doc_id, revision=ctx.revision,
                           section=element.section_number, page=element.page)

        # requirement/test tables: TestCase → Timer/Message/State/Parameter
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header:
                continue
            test_col = _first_col(header, ("test case", "check", "test id", "verification"))
            if test_col is None:
                continue
            requirement_col = _first_col(header, ("requirement", "base specification reference", "spec reference"))
            verifies_col = _first_col(header, ("what it verifies", "verifies", "description", "what it checks"))
            limit_col = _first_col(header, ("limit", "timer checked", "timing checked", "pass criterion"))
            for row in table.rows:
                if not row or not row[test_col].strip():
                    continue
                raw_id = CHECK_RE.search(row[test_col]) or TEST_ID_RE.search(row[test_col])
                if raw_id is None:
                    continue
                identifier = (raw_id.group("id") if raw_id.re is CHECK_RE else raw_id.group(0)).strip()
                is_check = identifier.startswith("[") or CHECK_RE.match(identifier)
                node_type = "Check" if is_check else "TestCase"
                node = self._node(result, node_type, identifier, doc_id=ctx.doc_id, revision=ctx.revision,
                                  section=table.section_number, page=table.pages[0] if table.pages else 0)
                context_text = " ".join(cell for i, cell in enumerate(row)
                                        if i in (verifies_col, limit_col, requirement_col) and cell)
                limit_text = _cell(row, limit_col)
                for timer in TIMER_RE.findall(context_text):
                    timer_node = self._node(result, "Timer", timer, doc_id=ctx.doc_id, revision=ctx.revision,
                                            page=table.pages[0] if table.pages else 0)
                    relation = "VERIFIES" if node_type == "Check" else "TESTS"
                    self._edge(result, node, timer_node, relation, table=table, confidence=0.9,
                               limit=limit_text, evidence=context_text[:200])
                for message_name in MESSAGE_LEXICON:
                    if re.search(rf"\b{re.escape(message_name)}\b", context_text):
                        message = self._node(result, "Message", message_name, doc_id=ctx.doc_id,
                                             revision=ctx.revision, page=table.pages[0] if table.pages else 0)
                        relation = "DEPENDS_ON" if node_type == "Check" else "TESTS"
                        self._edge(result, node, message, relation, table=table, confidence=0.7,
                                   reason="mentioned in the check description")
                for state_name in STATE_RE.findall(context_text):
                    state = self._node(result, "State", state_name, doc_id=ctx.doc_id, revision=ctx.revision,
                                       page=table.pages[0] if table.pages else 0)
                    relation = "VERIFIES" if node_type == "Check" else "TESTS"
                    self._edge(result, node, state, relation, table=table, confidence=0.7, limit=limit_text)
                # requirement / assertion id in the row
                if requirement_col is not None and requirement_col < len(row):
                    for assertion in ASSERTION_RE.findall(row[requirement_col]):
                        assertion_node = self._node(result, "Assertion", assertion, doc_id=ctx.doc_id,
                                                    revision=ctx.revision, page=table.pages[0] if table.pages else 0,
                                                    kind="requirement_id")
                        self._edge(result, node, assertion_node, "TESTS", table=table, confidence=1.0)
                # references to the base specification inside the row
                if requirement_col is not None and requirement_col < len(row):
                    self._row_references(result, ctx, node, row[requirement_col], table)

    def _domain_entities(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                         sections: dict[str, Node]) -> None:
        """Signals, pins, roles, profiles, Tech 2 phases and packets."""
        for element in self._iter_text(ctx):
            section = sections.get(element.section_number)
            for node_type, pattern in (("Signal", SIGNAL_RE), ("Pin", PIN_RE), ("Role", ROLE_RE),
                                       ("PowerProfile", PROFILE_RE), ("Tech2Phase", PHASE_RE),
                                       ("Tech2Packet", TECH2_PACKET_RE), ("Router", ROUTER_RE),
                                       ("Adapter", ADAPTER_RE), ("Parameter", PARAM_RE)):
                for name in set(pattern.findall(element.text)):
                    entity = self._node(result, node_type, name, doc_id=ctx.doc_id, revision=ctx.revision,
                                        section=element.section_number, page=element.page)
                    if section is not None and node_type in ("Signal", "Pin", "Parameter"):
                        self._edge(result, entity, section, "DEFINED_IN", element=element, confidence=0.5)
        # Tech 2 packets have identifiers of their own (SIG.STR, CTL.ERR, EOP)
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if "packet" not in (header[0] if header else ""):
                continue
            for row in table.rows:
                if len(row) < 2 or not row[0].strip():
                    continue
                packet = self._node(result, "Tech2Packet", row[0].strip(), doc_id=ctx.doc_id,
                                    revision=ctx.revision, page=table.pages[0] if table.pages else 0,
                                    identifier=row[1].strip(), sent_during=_cell(row, 2))
                self._edge(result, packet, doc, "DEFINED_IN", table=table, confidence=0.9)

    def _revision_history(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                          sections: dict[str, Node], tables: dict[str, Node]) -> None:
        """Revision table rows become Revision nodes, ECN nodes and SUPERSEDES edges."""
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header or "revision" not in header[0] or "revision history" not in (table.caption or "").lower():
                continue
            previous: Node | None = None
            for row in table.rows:
                if not row or not row[0].strip():
                    continue
                key = re.sub(r"\s+", " ", row[0].strip())
                date = row[1].strip() if len(row) > 1 else ""
                changes = " ".join(row[2:]) if len(row) > 2 else ""
                revision = self._node(result, "Revision", f"{ctx.doc_id}:{key}", doc_id=ctx.doc_id,
                                      revision=key, section=table.section_number,
                                      page=table.pages[0] if table.pages else 0, key=key, date=date,
                                      changes=changes[:400])
                revision.canonical_name = key
                self._edge(result, doc, revision, "HAS_REVISION", table=table, confidence=1.0)
                if previous is not None:
                    self._edge(result, revision, previous, "SUPERSEDES", table=table, confidence=0.9,
                               reason="later row in the revision history table")
                previous = revision
                for match in ECN_RE.finditer(changes) or []:
                    ecn_name = f"ECN {match.group(1)}"
                    ecn = self._node(result, "ECN", ecn_name, doc_id=ctx.doc_id, revision=key,
                                     section=table.section_number, page=table.pages[0] if table.pages else 0)
                    self._edge(result, revision, ecn, "REFERENCES", table=table, confidence=0.9,
                               doc_id=ctx.doc_id, evidence=changes[:200])

    def _row_references(self, result: ExtractionResult, ctx: ExtractionContext, node: Node, text: str,
                        table: Table) -> None:
        """Cross-references written inside a table cell (often to another document)."""
        target_doc = _doc_from_text(text) or ctx.doc_id
        stale = _is_stale(text, ctx, target_doc)
        for match in SECTION_REF_RE.finditer(text):
            number = match.group(1)
            target = self._node(result, "Section", f"{target_doc}:§{number}", doc_id=target_doc,
                                revision="" if target_doc != ctx.doc_id else ctx.revision,
                                section=number, page=0)
            self._edge(result, node, target, "REFERENCES", table=table, confidence=0.8, stale=stale,
                       evidence=text[:200], reference=match.group(0))
        for match in TABLE_REF_RE.finditer(text):
            number = match.group(1)
            target = self._node(result, "Table", f"{target_doc}:table:{number}", doc_id=target_doc,
                                revision="" if target_doc != ctx.doc_id else ctx.revision, number=number)
            self._edge(result, node, target, "REFERENCES", table=table, confidence=0.8, stale=stale,
                       evidence=text[:200], reference=match.group(0))

    def _cross_references(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node,
                          sections: dict[str, Node], tables: dict[str, Node], figures: dict[str, Node]) -> None:
        for element in self._iter_text(ctx):
            source = sections.get(element.section_number)
            if source is None:
                continue
            text = element.text
            target_doc = _doc_from_text(text) or ctx.doc_id
            stale = _is_stale(text, ctx, target_doc)
            evidence = text[:200]
            for match in SECTION_REF_RE.finditer(text):
                number = match.group(1)
                if number == element.section_number:
                    continue
                node = self._node(result, "Section", f"{target_doc}:§{number}", doc_id=target_doc,
                                  revision="" if target_doc != ctx.doc_id else ctx.revision, section=number)
                self._edge(result, source, node, "REFERENCES", element=element, confidence=0.85 if not stale else 0.9,
                           stale=stale, reference=match.group(0), evidence=evidence)
            for match in TABLE_REF_RE.finditer(text):
                number = match.group(1)
                if number == element.section_number:
                    continue
                node = self._node(result, "Table", f"{target_doc}:table:{number}", doc_id=target_doc,
                                  revision="" if target_doc != ctx.doc_id else ctx.revision, number=number)
                self._edge(result, source, node, "REFERENCES", element=element, confidence=0.85, stale=stale,
                           reference=match.group(0), evidence=evidence)
            for match in FIGURE_REF_RE.finditer(text):
                number = match.group(1)
                node = self._node(result, "Figure", f"{target_doc}:figure:{number}", doc_id=target_doc,
                                  revision="" if target_doc != ctx.doc_id else ctx.revision, number=number)
                self._edge(result, source, node, "REFERENCES", element=element, confidence=0.85, stale=stale,
                           reference=match.group(0), evidence=evidence)
            for match in CHAPTER_REF_RE.finditer(text):
                number = match.group(1)
                node = self._node(result, "Section", f"{target_doc}:§{number}", doc_id=target_doc,
                                  revision="" if target_doc != ctx.doc_id else ctx.revision, section=number)
                self._edge(result, source, node, "REFERENCES", element=element, confidence=0.6, stale=stale,
                           reference=match.group(0), evidence=evidence)

    def _tunnels(self, ctx: ExtractionContext, result: ExtractionResult, doc: Node, sections: dict[str, Node],
                 tables: dict[str, Node]) -> None:
        for table in ctx.tables:
            header = [c.lower() for c in (table.headers[0] if table.headers else [])]
            if not header or "protocol" not in header[0]:
                continue
            if not any("adapter" in h for h in header):
                continue
            for row in table.rows:
                if len(row) < 3 or not row[0].strip():
                    continue
                protocol, adapter, section = row[0].strip(), row[1].strip(), row[2].strip()
                adapter_node = self._node(result, "Adapter", adapter, doc_id=ctx.doc_id, revision=ctx.revision,
                                          page=table.pages[0] if table.pages else 0,
                                          adapter_id=_cell(row, 3))
                protocol_node = self._node(result, "Protocol", protocol, doc_id=ctx.doc_id,
                                           revision=ctx.revision, page=table.pages[0] if table.pages else 0)
                tunnel = self._node(result, "Tunnel", f"{protocol} tunnel", doc_id=ctx.doc_id,
                                    revision=ctx.revision, page=table.pages[0] if table.pages else 0,
                                    protocol=protocol)
                self._edge(result, adapter_node, protocol_node, "TUNNELS", table=table, confidence=1.0,
                           section=section, evidence=f"{protocol} <- {adapter}")
                self._edge(result, tunnel, adapter_node, "CONTAINS", table=table, confidence=1.0)
                ref = SECTION_REF_RE.search(section)
                if ref:
                    number = ref.group(1)
                    target = self._node(result, "Section", f"{ctx.doc_id}:§{number}", doc_id=ctx.doc_id,
                                        revision=ctx.revision, section=number)
                    self._edge(result, adapter_node, target, "REFERENCES", table=table, confidence=0.8,
                               reference=section)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _section_number_of(name: str) -> str:
    """``pd:§6.6.2`` or ``Section 6.6.2`` → ``6.6.2``."""
    match = re.search(r"§\s*([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)", name)
    if match:
        return match.group(1)
    match = re.search(r"section\s+([0-9]+(?:\.[0-9]+)*)", name, re.I)
    return match.group(1) if match else ""


def _trailing_number(name: str) -> str:
    match = re.search(r"(?:table|figure)\s+([0-9A-Z]+(?:[.\-][0-9A-Z]+)*)", name, re.I)
    return match.group(1) if match else ""


def _cell(row: Sequence[str], idx: int | None) -> str:
    if idx is None or idx >= len(row):
        return ""
    return str(row[idx]).strip()


def _first_col(header: Sequence[str], names: Sequence[str]) -> int | None:
    for i, col in enumerate(header):
        for name in names:
            if name in col:
                return i
    return None


def _roles_in(text: str) -> list[str]:
    found = []
    for role in ("Source", "Sink", "Both", "Cable", "Transmitter", "Receiver"):
        if re.search(rf"\b{role}\b", text, re.I):
            found.append(role)
    return found


def _test_area(identifier: str) -> str:
    parts = identifier.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return identifier


def _doc_from_text(text: str) -> str | None:
    """Resolve the document a sentence refers to, if it names one."""
    lowered = text.lower()
    for alias, doc_id in DOC_ALIASES.items():
        if alias in lowered:
            return doc_id
    return None


OLD_REVISIONS = {"pd": ("r3.0", "r3.1"), "typec": ("release 2.3", "release 2.4", "r2.4")}


def _is_stale(text: str, ctx: ExtractionContext, target_doc: str) -> bool:
    """Flag a cross-reference that points at an older revision.

    The assignment calls this out: test specifications are written against
    earlier base-specification revisions, so a section or table number may now
    point somewhere else.  We detect it by looking for an older revision string
    next to the reference.
    """
    if target_doc == ctx.doc_id:
        return False
    lowered = text.lower()
    for old in OLD_REVISIONS.get(target_doc, ()):
        if old in lowered:
            return True
    current = str(ctx.other_docs.get(target_doc, {}).get("revision", "")).lower()
    for match in REVISION_RE.finditer(lowered):
        found = match.group(0).lower()
        if current and found and found.replace(" ", "") not in current.replace(" ", ""):
            return True
    return False


def extract_document(ctx: ExtractionContext) -> ExtractionResult:
    """Convenience wrapper used by the build script and the tests."""
    return RuleExtractor().extract(ctx)
