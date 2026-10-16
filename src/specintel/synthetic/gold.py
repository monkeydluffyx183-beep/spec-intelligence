"""Gold question set for the synthetic corpus, derived from the manifest.

Because the generator knows exactly what it wrote and on which page, it can
produce a *verified* gold set: every question carries the answer, the evidence
page that the answer was written from, and the content type that holds it.
``eval/gold.jsonl`` is therefore machine-consistent for the stand-in corpus,
while ``eval/gold_seed.jsonl`` (built by :func:`build_seed_questions`) carries
the assignment's Appendix A seed questions for the team to answer against the
real documents.

Categories follow Appendix A: S T F X C G V U.
Content types follow the Task 5 matrix: prose, bitfield_table, multipage_table,
state_machine, timing, test_procedure.
"""

from __future__ import annotations

from typing import Any, Iterable

from specintel.synthetic.gold_extra import EXTRA_TEMPLATES, HELD_OUT_IDS

# --------------------------------------------------------------------------- #
# question templates: (qid, category, question, answer, anchors, content_type,
#                      docs, author)
# `anchors` are manifest anchors; the pages are resolved at build time.
# --------------------------------------------------------------------------- #
QUESTION_TEMPLATES: list[dict[str, Any]] = [
    # --- single-hop text -------------------------------------------------- #
    dict(qid="S1", category="S", content_type="prose", docs=["pd"], author="ws1-owner",
         question="What does the PS_RDY message signal, and which port sends it during an explicit contract "
                  "negotiation?",
         answer="PS_RDY signals that the power supply has reached the requested operating point and that the new "
                "contract is in force. The port that controls the newly negotiated supply, in practice the Source, "
                "sends it; a Sink that sent a Request waits for PS_RDY before drawing the negotiated current.",
         anchors=["ps_rdy_role", "table:6.1"]),
    dict(qid="S2", category="S", content_type="prose", docs=["typec"], author="ws1-owner",
         question="Under what conditions must a USB Type-C cable be electronically marked?",
         answer="A cable shall be electronically marked when it can carry more than 3 A, when it is a captive cable "
                "that must report its rating, or when its USB4/USB 3.2 capability has to be signalled to the ports.",
         anchors=["typec:emarker", "sec:4.8"]),
    dict(qid="S3", category="S", content_type="test_procedure", docs=["scc_cts"], author="ws4-owner",
         question="What does the Shared Capacity Charger interoperability test verify (V.1-V.5), and under which "
                  "device conditions is it run?",
         answer="V.1 checks a single-port maximum contract, V.2 a second port attaching while the first is idle, "
                "V.3 a second port attaching while the first holds a full contract (the first contract must be "
                "renegotiated, not dropped), V.4 the renegotiation timing, and V.5 that released capacity is "
                "re-advertised. It runs on a Source with two or more ports that share one capacity, with the shared "
                "capacity advertised and two Sinks attached.",
         anchors=["table:3-2", "scc:conditions", "table:2-1"]),
    dict(qid="S4", category="S", content_type="prose", docs=["tech2"], author="ws2-owner",
         question="What happens during the Tech 2 Ping phase, and what does the Power Receiver send in response to "
                  "a digital ping?",
         answer="The Power Transmitter applies a digital ping (a short burst at the operating frequency) and waits "
                "for a response. The Power Receiver answers with a Signal Strength packet, which also reports "
                "placement quality.",
         anchors=["tech2:ping", "table:3@book1"]),
    dict(qid="S5", category="S", content_type="prose", docs=["usb4"], author="ws2-owner",
         question="Which protocols can USB4 tunnel, and which section of the USB4 specification defines each?",
         answer="USB 3.2 (USB4 §7.2), DisplayPort (§7.4), PCIe (§7.6) and Host-to-Host (§7.8).",
         anchors=["table:7-2", "usb4:7:intro"]),

    # --- table lookup ----------------------------------------------------- #
    dict(qid="T1", category="T", content_type="bitfield_table", docs=["pd"], author="ws1-owner",
         question="List the fields of a Fixed Supply PDO sent by a Source, with their bit positions.",
         answer="B31..30 PDO Type (00b = Fixed Supply), B29 Dual-Role Power, B28 USB Suspend Supported, B27 "
                "Unconstrained Power, B26 USB Communications Capable, B25 Dual-Role Data, B24..22 Peak Current, "
                "B21..20 Reserved, B19..10 Voltage in 50 mV units, B9..0 Maximum Current in 10 mA units.",
         anchors=["table:7.1"]),
    dict(qid="T2", category="T", content_type="multipage_table", docs=["pd"], author="ws1-owner",
         question="What are the minimum and maximum values of tSenderResponse, and in which table are they defined?",
         answer="Minimum 24 ms and maximum 30 ms, defined in Table 6.8 (Timer definitions and limits).",
         anchors=["table:6.8"]),
    dict(qid="T3", category="T", content_type="bitfield_table", docs=["typec"], author="ws1-owner",
         question="Which Rp values advertise Default USB power, 1.5 A and 3.0 A, and which CC voltage ranges does a "
                  "Sink use to detect each?",
         answer="56 kΩ (or 36 kΩ) advertises Default USB power and is detected as vRd-USB 0.25 V-0.61 V; 22 kΩ "
                "advertises 1.5 A, detected as vRd-1.5 0.70 V-1.16 V; 10 kΩ advertises 3.0 A, detected as vRd-3.0 "
                "1.31 V-2.04 V; 4.7 kΩ (electronically marked cable) is detected at 2.45 V-3.05 V.",
         anchors=["table:4-24", "table:4-25"]),
    dict(qid="T4", category="T", content_type="multipage_table", docs=["pd"], author="ws3-owner",
         question="What are the limits of tFirstSourceCap and tTypeCSinkWaitCap?",
         answer="tFirstSourceCap is 100 ms minimum and 200 ms maximum; tTypeCSinkWaitCap is 310 ms minimum and "
                "620 ms maximum (Table 6.8).",
         anchors=["table:6.8"]),

    # --- figure / diagram ------------------------------------------------- #
    dict(qid="F1", category="F", content_type="state_machine", docs=["pd"], author="ws1-owner",
         question="In the Source Policy Engine state diagram, which state follows PE_SRC_Send_Capabilities when a "
                  "Request is received, and which timer runs while the Source waits for it?",
         answer="PE_SRC_Negotiate_Capability follows PE_SRC_Send_Capabilities when a Request is received, and "
                "tSenderResponse runs while the Source waits for the exchange to complete.",
         anchors=["figure:F-1"]),
    dict(qid="F2", category="F", content_type="state_machine", docs=["typec"], author="ws2-owner",
         question="In the Source connection state diagram, which conditions move a port from Unattached.SRC to "
                  "AttachWait.SRC and then to Attached.SRC?",
         answer="Unattached.SRC moves to AttachWait.SRC when a Sink termination is detected on CC1 or CC2; "
                "AttachWait.SRC moves to Attached.SRC once the CC voltage stays inside the advertised range for "
                "tCCDebounce, after which VBUS is enabled.",
         anchors=["figure:4-12"]),
    dict(qid="F3", category="F", content_type="timing", docs=["tech2"], author="ws1-owner",
         question="During Tech 2 power transfer, how does the Control Error Packet drive the Power Transmitter's "
                  "operating point?",
         answer="The Power Receiver measures the received power and sends a Control Error packet; a positive error "
                "asks the Transmitter to increase the operating point and a negative one to decrease it. The "
                "Transmitter feeds the error into a proportional-integral controller and moves its operating "
                "frequency and duty cycle, with a limit on how fast the operating point may change.",
         anchors=["figure:4@book4", "tech2:control_error"]),
    dict(qid="F4", category="F", content_type="state_machine", docs=["pd"], author="ws4-owner",
         question="In the Sink Policy Engine state diagram, which states does the Sink pass through between "
                  "PE_SNK_Wait_for_Capabilities and PE_SNK_Transition_Sink?",
         answer="The Sink moves from PE_SNK_Wait_for_Capabilities to PE_SNK_Select_Capability when "
                "Source_Capabilities arrives, and reaches PE_SNK_Transition_Sink after the Source's Accept; PS_RDY "
                "returns it to PE_SNK_Select_Capability for a later operating point.",
         anchors=["figure:F-2"]),

    # --- cross-document multi-hop ---------------------------------------- #
    dict(qid="X1", category="X", content_type="test_procedure", docs=["pd_cts", "pd"], author="ws4-owner",
         question="Which USB PD compliance test checks a Source's response time to a Request message, and which "
                  "timer and limit from the PD specification does it check against?",
         answer="TEST.PD.PROT.SRC.3. Its checks #1 and #2 compare the measured time from the Request to Accept "
                "against tSenderResponse (minimum 24 ms, maximum 30 ms in PD Table 6.8), and its further checks use "
                "tPSTransition (450-550 ms).",
         anchors=["cts:src3", "table:5-3", "table:6.8"]),
    dict(qid="X2", category="X", content_type="test_procedure", docs=["typec_fts", "typec"], author="ws4-owner",
         question="Which Type-C functional tests exercise Try.SRC or Try.SNK behaviour, and which sections of the "
                  "Type-C specification define those states?",
         answer="TEST.TYPEC.STATE.TRYSRC.1 and TEST.TYPEC.STATE.TRYSNK.1. Try.SRC is defined in Type-C §4.5.2.2.5 "
                "and Try.SNK in §4.5.2.1.5; the tests check tDRPTryWait.",
         anchors=["fts:try_tests", "table:4-2", "typec:try_states"]),
    dict(qid="X3", category="X", content_type="test_procedure", docs=["pd_cts", "pd"], author="ws3-owner",
         question="Which check of TEST.PD.PROT.SRC.3 verifies the Source's response to an unsupported request, and "
                  "which state must the Source be in for that check?",
         answer="Check #4 verifies that a Request for an unsupported PDO is refused with Reject or Not_Supported "
                "within tSenderResponse (30 ms maximum). The measured response time depends on the Source being in "
                "PE_SRC_Ready, as stated for check #3 of the same test.",
         anchors=["table:5-3", "cts:src3", "pd:8:ready"]),
    dict(qid="X4", category="X", content_type="prose", docs=["scc_cts", "typec"], author="ws4-owner",
         question="The Shared Capacity Charger test specification refers to Type-C section 4.8.6. What does that "
                  "section cover, and which of the test's verifications depend on it?",
         answer="Type-C §4.8.6 covers Shared Capacity Charging: the Source must keep total delivered power inside "
                "its shared capacity, renegotiate a port's contract when the budget changes, and not drop a port "
                "below its contract without renegotiating. V.1 to V.5 all reference it, and V.3 and V.4 are the "
                "verifications that depend on the renegotiation requirement.",
         anchors=["typec:4.8.6:shared_capacity", "table:3-2", "scc:v3"]),
    dict(qid="X5", category="X", content_type="prose", docs=["pd", "typec", "usb4"], author="ws2-owner",
         question="Which USB PD message is used to enter USB4 operation, and which sections of the USB Type-C and "
                  "USB4 specifications describe the resulting discovery, entry and link?",
         answer="Enter_USB. The PD specification defines the message (§6.4), the Type-C specification defines USB4 "
                "Discovery and Entry (§4.9 and §6.9) and the USB4 specification defines router-level entry (§6.5) "
                "and the tunnels (Chapter 7).",
         anchors=["enter_usb_role", "typec:4.9:usb4_entry", "usb4:entry", "table:6-1"]),

    # --- comparison ------------------------------------------------------- #
    dict(qid="C1", category="C", content_type="prose", docs=["pd"], author="ws3-owner",
         question="Compare SPR PPS and EPR AVS: voltage ranges, step sizes, and how a Sink requests each.",
         answer="PPS (SPR) covers 3.3 V to 21 V with a 20 mV step size and up to 5 A in 50 mA steps, requested with "
                "a PPS Request Data Object. AVS (EPR) covers 15 V to 48 V with a 100 mV step size, requested with "
                "an EPR_Request whose data object follows the AVS layout after EPR_Mode has been exchanged.",
         anchors=["pd:7.3:pps", "pd:7.5:epr", "table:7.7"]),
    dict(qid="C2", category="C", content_type="prose", docs=["tech2"], author="ws2-owner",
         question="Compare how a Power Receiver reaches power transfer under the Tech 2 Baseline Power Profile and "
                  "the Extended Power Profile. Which phases differ?",
         answer="BPP uses Ping, Configuration and Power Transfer only; EPP inserts a Negotiation phase between "
                "Configuration and Power Transfer, during which the guaranteed power level is agreed and confirmed "
                "with an End Negotiation packet.",
         anchors=["table:3@book4", "tech2:profiles"]),
    dict(qid="C3", category="C", content_type="prose", docs=["pd", "typec"], author="ws1-owner",
         question="How do the PD and Type-C specifications share responsibility for the tEnterUSB handover?",
         answer="PD defines the Enter_USB message and the tEnterUSB timer that bounds it (Table 6.8), while Type-C "
                "defines the USB4 Discovery and Entry handover that follows the accepted message (§4.9, §6.9), "
                "after which the USB4 specification takes over at the router level (§6.5).",
         anchors=["table:6.8", "enter_usb_role", "typec:6.9:discovery"]),

    # --- graph traversal -------------------------------------------------- #
    dict(qid="G1", category="G", content_type="prose", docs=["pd"], author="ws3-owner",
         question="Which messages may a Sink send in reply to Source_Capabilities in this corpus, and what does each "
                  "achieve?",
         answer="Request selects one advertised capability, Reject refuses the capabilities, and Not_Supported "
                "reports that the request cannot be served; the Sink keeps waiting in PE_SNK_Wait_for_Capabilities "
                "until Source_Capabilities is sent again after Wait or Not_Supported.",
         anchors=["table:6.1", "table:6.2", "pd:6.4.1:exchange"]),
    dict(qid="G2", category="G", content_type="multipage_table", docs=["pd", "pd_cts"], author="ws3-owner",
         question="Which timers appear both in the PD specification's timer table and in the PD compliance test "
                  "specification's timer traceability table?",
         answer="tSenderResponse (TEST.PD.PROT.SRC.3 and TEST.PD.PROT.SNK.1), tPSTransition (TEST.PD.PROT.SRC.3), "
                "tFirstSourceCap (TEST.PD.PE.SRC.2), tGoodCRC (TEST.PD.PROT.SRC.1) and tPSSourceOff "
                "(TEST.PD.PWR.SRC.1) appear in both.",
         anchors=["table:6.8", "table:5-9"]),

    # --- revision aware --------------------------------------------------- #
    dict(qid="V1", category="V", content_type="multipage_table", docs=["pd"], author="ws5-owner",
         question="According to the PD revision history, which Revision 3.1 versions incorporated EPR-related ECNs, "
                  "and what were those ECNs?",
         answer="R3.1 V1.2 incorporated ECN EPR_Source_Capabilities and ECN EPR_Sink_Capabilities; V1.4 "
                "incorporated ECN EPR_Mode_Timing; V1.5 incorporated ECN EPR_AVS_Voltage_Range and ECN "
                "EPR_KeepAlive; V1.6 incorporated ECN EPR_AVS_Step_Size and ECN EPR_Request_Reject; V1.7 "
                "incorporated ECN EPR_PPS_Extended_Range; V1.8 incorporated ECN EPR_Chunking_Not_Supported. The "
                "table spans two pages.",
         anchors=["table:A.1"]),
    dict(qid="V2", category="V", content_type="prose", docs=["typec_fts", "typec"], author="ws5-owner",
         question="The Type-C Functional Test Specification points to \"Spec Table 4-23\" for the USB Type-C "
                  "configuration matrix. What is Table 4-23 in Type-C Release 2.5, and what should the system tell "
                  "the engineer?",
         answer="In Release 2.5 the configuration matrix moved to Table 4-9; the reference is stale. The engineer "
                "should be told that Table 4-23 no longer holds the configuration matrix, that the number changed "
                "from Release 2.4 to Release 2.5 (appendix B), and that the current matrix is Table 4-9.",
         anchors=["table:B-1", "table:4-9", "fts:td_4_2_4"]),
    dict(qid="V3", category="V", content_type="prose", docs=["pd_cts", "pd"], author="ws5-owner",
         question="The PD Compliance Test Specification describes coverage in terms of chapters of older PD "
                  "revisions. Where do message definitions and state-machine diagrams live in PD R3.2 V1.2?",
         answer="In R3.2 V1.2 the message definitions live in Chapter 6 (Protocol Layer Messages) and the state "
                "machines in Appendix F (State Diagrams). The test specification's coverage section still refers to "
                "Chapter 5 of R3.0 for message definitions, so the reference is stale; the table for "
                "TEST.PD.PWR.SRC.1 also still cites Chapter 5.6.2 of R3.0.",
         anchors=["cts:coverage", "table:5-9", "pd:8:source_sm", "sec:6"]),

    # --- unanswerable / false premise ------------------------------------ #
    dict(qid="U1", category="U", content_type="prose", docs=["pd"], author="ws4-owner", answerable=False,
         question="What is the EPR Fixed Supply PDO voltage for 60 V operation?",
         answer="The premise does not hold: there is no 60 V EPR fixed supply contract in this corpus. The highest "
                "fixed EPR voltage is 48 V; EPR fixed voltages are 28 V, 36 V and 48 V, and the AVS range ends at "
                "48 V. The system should decline and show what it searched.",
         anchors=["pd:7.5:epr", "table:7.7"]),
    dict(qid="U2", category="U", content_type="prose", docs=[], author="ws4-owner", answerable=False,
         question="What does IEEE 802.11be (Wi-Fi 7) require for multi-link operation?",
         answer="The corpus does not cover IEEE 802.11be. The USB4 specification mentions Wi-Fi only for "
                "coexistence signalling; nothing in the seven documents specifies multi-link operation, so the "
                "system must decline.",
         anchors=["usb4:wireless"]),
    dict(qid="U3", category="U", content_type="prose", docs=["typec"], author="ws5-owner", answerable=False,
         question="Which Type-C release first allowed a 240 W cable assembly, and in which year was that release "
                  "published?",
         answer="The corpus does not state the release history of cable power ratings. Type-C Release 2.5 gives "
                "current ratings and shared capacity behaviour but no 240 W cable rating history, so the system must "
                "decline rather than answer from the model's memory.",
         anchors=["typec:4.8:intro", "typec:emarker"]),

    # --- harder table / procedure questions ------------------------------ #
    dict(qid="T5", category="T", content_type="test_procedure", docs=["pd_cts"], author="ws2-owner",
         question="List the steps of TD 4.2.4 with their tags and expected results.",
         answer="V.1 attach the reference Sink (VBUS within tVBUSON), V.2 wait for the first Source_Capabilities "
                "(within tFirstSourceCap), V.3 request the first Fixed Supply PDO (Accept within tSenderResponse), "
                "V.4 measure Accept to PS_RDY (within tPSTransition), V.5 measure VBUS (within 5 % of requested), "
                "V.6 request an unsupported PDO (Reject or Not_Supported within tSenderResponse).",
         anchors=["table:4-4"]),
    dict(qid="T6", category="T", content_type="bitfield_table", docs=["pd"], author="ws1-owner",
         question="Which bits of a Programmable Power Supply APDO carry the maximum and minimum voltage, and in "
                  "what units?",
         answer="B23..17 carry the maximum programmable voltage and B16..9 the minimum programmable voltage, both "
                "in 100 mV units; B8..0 carry the maximum current in 50 mA units.",
         anchors=["table:7.4"]),
    dict(qid="F5", category="F", content_type="state_machine", docs=["usb4"], author="ws1-owner",
         question="In the USB4 architecture figure, which adapters hang off the Lane Adapter and what does each "
                  "terminate?",
         answer="The USB3 Adapter, the DP Adapter and the PCIe Adapter hang off the Lane Adapter; each terminates "
                "one tunnel to the peer router, and the Lane Adapter terminates the physical lanes.",
         anchors=["figure:4-1", "usb4:routers"]),
    dict(qid="X6", category="X", content_type="test_procedure", docs=["scc_cts", "typec", "pd"], author="ws4-owner",
         question="Verification V.4 of the Shared Capacity Charger tests measures a renegotiation time. Which "
                  "timers does it depend on, and where is each defined?",
         answer="V.4 depends on tSenderResponse (PD Table 6.8, 24-30 ms) plus the supply transition time "
                "tPSTransition (450-550 ms), and on the shared capacity renegotiation requirement of Type-C §4.8.6 "
                "that gives the contract change its timing context.",
         anchors=["table:3-2", "table:6.8", "scc:v3"]),
    dict(qid="G3", category="G", content_type="prose", docs=["pd"], author="ws3-owner",
         question="Which states start or stop the timer tSenderResponse in the PD specification's Source Policy "
                  "Engine description?",
         answer="PE_SRC_Negotiate_Capability starts tSenderResponse when a Request is received, and leaving the "
                "negotiation (to PE_SRC_Transition_Supply after Accept) stops it; PE_SRC_Ready starts it again for "
                "a new Request.",
         anchors=["pd:8:source_sm", "figure:F-1"]),
]


#: The full set: 34 questions from this module plus 26 from ``gold_extra`` (60 total,
#: 12 per team member).  ``HELD_OUT_IDS`` marks the reviewer subset.
ALL_TEMPLATES: list[dict[str, Any]] = QUESTION_TEMPLATES + EXTRA_TEMPLATES


def build_gold_questions(corpus: Any) -> list[dict[str, Any]]:
    """Resolve manifest anchors into pages and return gold question records."""
    gold: list[dict[str, Any]] = []
    for tmpl in ALL_TEMPLATES:
        evidence: list[dict[str, Any]] = []
        docs: set[str] = set(tmpl.get("docs") or [])
        for anchor in tmpl.get("anchors", []):
            entry = corpus.entry(anchor)
            if entry is None:
                continue
            page = int(entry["page"])
            docs.add(entry["doc_id"])
            evidence.append({
                "doc_id": entry["doc_id"],
                "page": page,
                "element_id": f"{entry['doc_id']}:{entry['type']}:{anchor}",
                "anchor": anchor,
                "region": _region_of(entry),
                "quote": _quote_of(entry),
            })
        gold.append({
            "qid": tmpl["qid"],
            "question": tmpl["question"],
            "category": tmpl["category"],
            "gold_answer": tmpl["answer"],
            "gold_evidence": evidence,
            "author": tmpl.get("author", "synthetic-generator"),
            "verifier": "synthetic-generator (page-verified by construction)",
            "content_type": tmpl.get("content_type", "prose"),
            "docs": sorted(docs),
            "answerable": bool(tmpl.get("answerable", True)),
            "split": "held_out" if tmpl["qid"] in HELD_OUT_IDS else "dev",
            "notes": "generated against the synthetic stand-in corpus",
            "gold_citations": [f"[{e['doc_id']}, §{_section(entry_of(corpus, e['anchor']))}, p.{e['page']}"
                               f", {e['region']}]" for e in evidence],
        })
    return gold


def entry_of(corpus: Any, anchor: str) -> dict[str, Any]:
    return corpus.entry(anchor) or {}


def _section(entry: dict[str, Any]) -> str:
    return str(entry.get("section") or "—")


def _region_of(entry: dict[str, Any]) -> str:
    t = entry.get("type")
    if t == "table":
        return f"Table {entry.get('number')}"
    if t == "figure":
        return f"Figure {entry.get('number')}"
    if t == "section_heading":
        return f"section {entry.get('number')}"
    return "paragraph"


def _quote_of(entry: dict[str, Any]) -> str:
    text = str(entry.get("text") or entry.get("caption") or entry.get("title") or "")
    return text[:280]


# --------------------------------------------------------------------------- #
# Appendix A seed questions (for the real corpus; answers left to the team)
# --------------------------------------------------------------------------- #
SEED_QUESTIONS: list[dict[str, Any]] = [
    dict(qid="SEED-S1", category="S", content_type="prose",
         question="What does the PS_RDY message signal, and which port sends it during an explicit contract "
                  "negotiation?", docs=["pd"]),
    dict(qid="SEED-S2", category="S", content_type="prose",
         question="Under what conditions must a USB Type-C cable be electronically marked?", docs=["typec"]),
    dict(qid="SEED-S3", category="S", content_type="test_procedure",
         question="What does the Shared Capacity Charger interoperability test verify (V.1-V.5), and under which "
                  "device conditions is it run?", docs=["scc_cts"]),
    dict(qid="SEED-S4", category="S", content_type="prose",
         question="What happens during the Tech 2 Ping phase, and what does the Power Receiver send in response to "
                  "a digital ping?", docs=["tech2"]),
    dict(qid="SEED-S5", category="S", content_type="prose",
         question="Which protocols can USB4 tunnel, and which chapter of the USB4 specification defines each?",
         docs=["usb4"]),
    dict(qid="SEED-T1", category="T", content_type="bitfield_table",
         question="List the fields of a Fixed Supply PDO sent by a Source, with their bit positions.", docs=["pd"]),
    dict(qid="SEED-T2", category="T", content_type="multipage_table",
         question="What are the minimum and maximum values of tSenderResponse, and in which table are they defined?",
         docs=["pd"]),
    dict(qid="SEED-T3", category="T", content_type="bitfield_table",
         question="Which Rp values advertise Default USB power, 1.5 A and 3.0 A, and which CC voltage ranges does a "
                  "Sink use to detect each?", docs=["typec"]),
    dict(qid="SEED-F1", category="F", content_type="state_machine",
         question="In the Source Policy Engine state diagram, which state follows PE_SRC_Send_Capabilities when a "
                  "Request is received, and which timer runs while the Source waits for it?",
         docs=["pd"], hint="figure"),
    dict(qid="SEED-F2", category="F", content_type="state_machine",
         question="In the Source connection state diagram, which conditions move a port from Unattached.SRC to "
                  "AttachWait.SRC and then to Attached.SRC?", docs=["typec"], hint="figure"),
    dict(qid="SEED-F3", category="F", content_type="timing",
         question="During Tech 2 power transfer, how does the Control Error Packet drive the Power Transmitter's "
                  "operating point? Use the PID control-algorithm figure.", docs=["tech2"], hint="figure"),
    dict(qid="SEED-X1", category="X", content_type="test_procedure",
         question="Which USB PD compliance test(s) check a Source's response time to a Request message, and which "
                  "timer and limit from the PD specification do they check against?", docs=["pd_cts", "pd"]),
    dict(qid="SEED-X2", category="X", content_type="test_procedure",
         question="Which Type-C functional tests exercise Try.SRC or Try.SNK behaviour, and which sections of the "
                  "Type-C specification define those states?", docs=["typec_fts", "typec"]),
    dict(qid="SEED-X3", category="X", content_type="test_procedure",
         question="Pick one PD compliance test on Source Capabilities. List every message, state and timer it "
                  "depends on, and where each is defined.", docs=["pd_cts", "pd"]),
    dict(qid="SEED-X4", category="X", content_type="prose",
         question="The Shared Capacity Charger test specification refers to Type-C Section 4.8.6. What does that "
                  "section cover in Release 2.5, and which of the test's verifications depend on it?",
         docs=["scc_cts", "typec"]),
    dict(qid="SEED-X5", category="X", content_type="prose",
         question="Which USB PD message is used to enter USB4 operation, and which sections of the USB Type-C and "
                  "USB4 specifications describe USB4 discovery, entry and the resulting link?",
         docs=["pd", "typec", "usb4"]),
    dict(qid="SEED-C1", category="C", content_type="prose",
         question="Compare SPR PPS and EPR AVS: voltage ranges, step sizes, and how a Sink requests each.",
         docs=["pd"]),
    dict(qid="SEED-C2", category="C", content_type="prose",
         question="Compare how a Power Receiver reaches power transfer under the Tech 2 Baseline Power Profile and "
                  "the Extended Power Profile. Which phases differ?", docs=["tech2"]),
    dict(qid="SEED-G1", category="G", content_type="prose",
         question="List every message a Sink may send in response to Source_Capabilities, and the Sink Policy "
                  "Engine state it enters after each.", docs=["pd"]),
    dict(qid="SEED-G2", category="G", content_type="multipage_table",
         question="Which timers appear in both the USB PD specification and the PD compliance test specification, "
                  "and in which test cases?", docs=["pd", "pd_cts"]),
    dict(qid="SEED-V1", category="V", content_type="multipage_table",
         question="According to the PD revision history, which Revision 3.1 versions incorporated EPR-related "
                  "ECNs, and what were those ECNs? (The table spans two pages.)", docs=["pd"]),
    dict(qid="SEED-V2", category="V", content_type="prose",
         question="The Type-C Functional Test Specification points to \"Spec Table 4-23\" for the USB Type-C "
                  "configuration matrix. What is Table 4-23 in Type-C Release 2.5, and what should the system tell "
                  "the engineer?", docs=["typec_fts", "typec"]),
    dict(qid="SEED-V3", category="V", content_type="prose",
         question="The PD Compliance Test Specification describes its coverage in terms of chapters of older PD "
                  "revisions. Where do message definitions and state-machine diagrams live in PD R3.2 V1.2?",
         docs=["pd_cts", "pd"]),
    dict(qid="SEED-U1", category="U", content_type="prose", answerable=False,
         question="What is the EPR Fixed Supply PDO voltage for 60 V operation? (Check whether the premise holds.)",
         docs=["pd"]),
    dict(qid="SEED-U2", category="U", content_type="prose", answerable=False,
         question="What does IEEE 802.11be (Wi-Fi 7) require for multi-link operation? (Not in this corpus: the "
                  "system should decline, even though the USB4 specification mentions Wi-Fi.)", docs=[]),
]


def build_seed_questions() -> list[dict[str, Any]]:
    """Appendix A seed questions with the gold fields left for a human author."""
    out: list[dict[str, Any]] = []
    for q in SEED_QUESTIONS:
        out.append({
            "qid": q["qid"],
            "question": q["question"],
            "category": q["category"],
            "gold_answer": "",                      # to be written from the documents
            "gold_evidence": [],                    # doc, page, element_id to be recorded
            "author": "assignment-appendix-a (unassigned)",
            "verifier": "",
            "content_type": q.get("content_type", "prose"),
            "docs": q.get("docs", []),
            "answerable": bool(q.get("answerable", True)),
            "notes": "seed question from assignment Appendix A; assign an author and fill the gold answer and "
                     "evidence before use",
            "gold_citations": [],
        })
    return out
