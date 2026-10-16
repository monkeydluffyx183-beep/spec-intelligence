"""Second half of the gold question set (the first is in :mod:`specintel.synthetic.gold`).

The assignment asks for at least 60 questions with at least 12 per team member; the
first module carries 34 and this module adds 26, bringing the set to 60 with 12 per
member and a balanced spread over the Task 5 content types.  Every question names
manifest anchors, so the evidence page is resolved from the corpus that was really
rendered — never typed by hand.

Question ids here start with ``A`` (additional) and stay stable so that evaluation
reports from earlier runs remain comparable.
"""

from __future__ import annotations

from typing import Any

EXTRA_TEMPLATES: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # ws1 — ingestion and table/figure grounding
    # ------------------------------------------------------------------ #
    dict(qid="A01", category="T", content_type="multipage_table", docs=["pd"], author="ws1-owner",
         question="Which counters does the USB PD specification define, and what does each count?",
         answer="nCapsCount (1 to 8 messages, the number of Source_Capabilities messages sent), "
                "nDiscoverIdentityCount (1 to 3, Discover Identity retries), nHardResetCount (2, the number "
                "of Hard Resets before a protocol error is reported) and nRetryCount (1 to 3, retries of a "
                "message that received no GoodCRC). All four are defined in the counter table next to the "
                "timer table (Counter definitions, PD §6.6).",
         anchors=["table:6.9"]),
    dict(qid="A02", category="F", content_type="timing", docs=["pd"], author="ws1-owner",
         question="In the EPR mode entry timing diagram, which exchanges take place between the mode request "
                  "and the first AVS operating point, and which message completes the change?",
         answer="The ports first exchange EPR_Mode, then the Sink sends an EPR_Request for an AVS operating "
                "point, and the Source confirms with PS_RDY before the new voltage takes effect (Figure 7-2). "
                "PS_RDY is the message that completes the change.",
         anchors=["figure:7-2", "pd:7.5:epr"]),

    # ------------------------------------------------------------------ #
    # ws2 — retrieval: bit-field tables, test-procedure tables, figures
    # ------------------------------------------------------------------ #
    dict(qid="A03", category="T", content_type="bitfield_table", docs=["pd"], author="ws2-owner",
         question="Which bits of a Programmable Power Supply APDO carry the minimum and maximum voltage, and "
                  "in what units are they expressed?",
         answer="B23..17 carry the maximum programmable voltage in 100 mV units above 5 V and B16..9 carry the "
                "minimum programmable voltage in 100 mV units; B8..0 carry the maximum current in 50 mA units "
                "(Table 7.4). B29..28 identify the APDO subtype as Programmable Power Supply.",
         anchors=["table:7.4", "pd:7.3:pps"]),
    dict(qid="A04", category="T", content_type="bitfield_table", docs=["pd"], author="ws2-owner",
         question="How is an Adjustable Voltage Supply APDO recognised, and how are its voltage limits encoded?",
         answer="An AVS APDO has APDO Type 11b and APDO Subtype 01b in B31..28; B19..10 hold the maximum voltage "
                "and B9..0 the minimum voltage, both in 100 mV units above 15 V, with peak current in B25..20 "
                "(Table 7.6).",
         anchors=["table:7.6", "pd:7.5:epr"]),
    dict(qid="A05", category="T", content_type="test_procedure", docs=["pd_cts"], author="ws2-owner",
         question="List the steps of TD 4.2.4 with their tags and expected results.",
         answer="V.1 attach the reference Sink (VBUS within tVBUSON), V.2 wait for Source_Capabilities (within "
                "tFirstSourceCap), V.3 request the first Fixed Supply PDO (Accept within tSenderResponse), "
                "V.4 measure Accept to PS_RDY (within tPSTransition), V.5 measure VBUS after PS_RDY (within 5 % "
                "of the requested voltage) and V.6 request an unsupported PDO (Reject or Not_Supported within "
                "tSenderResponse).",
         anchors=["table:4-4"]),
    dict(qid="A06", category="T", content_type="test_procedure", docs=["pd_cts"], author="ws2-owner",
         question="Which procedure does TEST.PD.PROT.SRC.3 follow, and what is the pass criterion after "
                  "repeating the Request ten times?",
         answer="The procedure runs TD 4.2.4 with a protocol analyser capturing every message, repeats the "
                "Request ten times from PE_SRC_Ready and reports the maximum measured response time; the pass "
                "criterion is that every Accept arrives inside the tSenderResponse limits and the maximum "
                "measured value is below 30 ms.",
         anchors=["table:5-4", "table:5-9"]),
    dict(qid="A07", category="F", content_type="state_machine", docs=["tech2"], author="ws2-owner",
         question="In the Tech 2 phase diagram, which packet ends each phase transition?",
         answer="The Ping phase hands over to Configuration after the Signal Strength packet, Configuration "
                "moves to Negotiation with the Configuration packet, and Negotiation reaches Power Transfer "
                "when the End Negotiation packet arrives (Figure 2, Book 1).",
         anchors=["figure:2@book1", "table:3@book1"]),
    dict(qid="A08", category="S", content_type="prose", docs=["tech2"], author="ws2-owner",
         question="Which packets does a Tech 2 Power Receiver send, and during which phase is each sent?",
         answer="Signal Strength (SIG.STR) during Ping, Control Error (CTL.ERR) and End Power Transfer (EOP) "
                "during Power Transfer, Configuration (CFG) during Configuration and End Negotiation (EON) "
                "during Negotiation.",
         anchors=["table:3@book6", "tech2:packets"]),

    # ------------------------------------------------------------------ #
    # ws3 — knowledge graph: multi-hop and cross-document paths
    # ------------------------------------------------------------------ #
    dict(qid="A09", category="G", content_type="multipage_table", docs=["pd_cts", "pd"], author="ws3-owner",
         question="Which PD timers are checked by which compliance test cases, and which of those rows cite a "
                  "superseded base specification?",
         answer="TEST.PD.PROT.SRC.1 checks tGoodCRC, TEST.PD.PROT.SRC.3 checks tSenderResponse and "
                "tPSTransition, TEST.PD.PROT.SNK.1 checks tSenderResponse, TEST.PD.PWR.SRC.1 checks "
                "tPSSourceOff and TEST.PD.PE.SRC.2 checks tFirstSourceCap. The rows for TEST.PD.PROT.SNK.1 and "
                "TEST.PD.PWR.SRC.1 cite PD R3.0 V1.3 (Table 6-8 and Chapter 5.6.2), a superseded revision, so "
                "they are stale references; the rest cite PD R3.2 V1.2 Table 6.8.",
         anchors=["table:5-9", "table:6.8", "table:3-1"]),
    dict(qid="A10", category="G", content_type="prose", docs=["scc_cts", "typec"], author="ws3-owner",
         question="Every verification in the Shared Capacity Charger test specification cites the same base "
                  "specification section. Which one is it, and which requirement does the section add?",
         answer="All five verifications cite Type-C §4.8.6 (Shared Capacity Charging). That section requires "
                "the Source to keep total delivered power inside its shared capacity, to renegotiate or reduce "
                "a port's contract when a new port attaches, and never to drop a port below its accepted "
                "minimum contract without renegotiating first — the requirement V.3 and V.4 exercise.",
         anchors=["table:3-2", "typec:4.8.6:shared_capacity", "scc:v3"]),
    dict(qid="A11", category="G", content_type="prose", docs=["usb4", "typec", "pd"], author="ws3-owner",
         question="Trace a single connection from the USB PD message to an established USB4 tunnel: which "
                  "document defines each hop?",
         answer="The USB PD message is Enter_USB (PD §6.4.4, message table and Enter_USB paragraph); the "
                "discovery and entry handover that follows is defined by the Type-C specification (§4.9 and "
                "§6.9); router-level entry and enumeration are defined by the USB4 specification (§6.5); the "
                "tunnels themselves are defined in USB4 Chapter 7 (USB 3.2 §7.2, DisplayPort §7.4, PCIe §7.6).",
         anchors=["table:6-1", "enter_usb_role", "typec:6.9:discovery", "table:7-2"]),
    dict(qid="A12", category="X", content_type="prose", docs=["typec_fts", "typec"], author="ws3-owner",
         question="Which requirement identifiers in the Type-C Functional Test Specification assertion table "
                  "reference USB4 entry or USB4 discovery, and which test case exercises each?",
         answer="R-USB4-0007 (Enter_USB handover to USB4 discovery) is exercised by TEST.TYPEC.USB4.ENTRY.1 and "
                "cites Type-C §6.9 together with Table 4-23. R-PWR-0031 (a Source keeps the shared capacity "
                "budget) is exercised by TEST.TYPEC.PWR.SHARED.1 and cites Type-C §4.8.6.",
         anchors=["table:2-1", "typec:6.9:discovery"]),
    dict(qid="A13", category="X", content_type="test_procedure", docs=["pd_cts", "typec_fts"], author="ws3-owner",
         question="Both test documents have a TD 4.2.4. Are they the same test, and what does each check?",
         answer="They share the identifier but not the content. In the PD compliance test specification TD "
                "4.2.4 drives a Source through attach, capability advertisement, Request, PS_RDY and VBUS "
                "measurement ([TD.4.2.4.V.1] to [TD.4.2.4.V.6]). In the Type-C functional test specification "
                "TD 4.2.4 walks the configuration matrix, checking attach time, advertised current, reported "
                "data capability and the e-marker requirement ([TD.4.2.4.V.1] to [TD.4.2.4.V.4]).",
         anchors=["table:4-4", "table:3-4", "fts:td_4_2_4"]),
    dict(qid="A14", category="C", content_type="prose", docs=["pd_cts", "pd"], author="ws3-owner",
         question="Compare how the Source-side and Sink-side protocol tests use tSenderResponse.",
         answer="TEST.PD.PROT.SRC.3 measures the Source's response to a Request against the tSenderResponse "
                "minimum (24 ms) and maximum (30 ms): check #1 checks the minimum, check #2 the maximum, check "
                "#3 the maximum again while the Source is already in PE_SRC_Ready and check #4 the refusal of "
                "an unsupported PDO. TEST.PD.PROT.SNK.1 measures the mirror case — the Sink must send its "
                "Request within tSenderResponse of receiving Source_Capabilities — so the same timer bounds "
                "both directions of the exchange.",
         anchors=["cts:src3", "cts:other_tests", "table:6.8"]),

    # ------------------------------------------------------------------ #
    # ws4 — agent behaviour: state, resets, stale references
    # ------------------------------------------------------------------ #
    dict(qid="A15", category="X", content_type="prose", docs=["pd", "pd_cts"], author="ws4-owner",
         question="What does Soft_Reset do, and which shared procedure verifies retry behaviour?",
         answer="Soft_Reset returns the protocol layer to its default state without interrupting VBUS: both "
                "ports reset their message counters and the policy engines return to the state that "
                "corresponds to the current contract. Retry behaviour and the counter limits are verified by "
                "the shared procedure COMMON.CHECK.PD.5.",
         anchors=["pd:6.3:soft_reset", "table:6-1"]),
    dict(qid="A16", category="X", content_type="prose", docs=["pd", "pd_cts"], author="ws4-owner",
         question="What triggers a hard reset, what happens to VBUS, and which counter limits the retries "
                  "before it is reported?",
         answer="A hard reset is initiated by the Protocol Layer when a message cannot be delivered within its "
                "retry budget; both policy engines return to their startup states and VBUS is removed for at "
                "least tHardResetComplete before the Sink may renegotiate. The retry budget is bounded by "
                "nRetryCount (1 to 3) and nHardResetCount (2), and the VBUS removal time is checked by "
                "TEST.PD.PWR.SRC.1 against tPSSourceOff.",
         anchors=["pd:8:hard_reset", "table:6.9", "cts:other_tests"]),
    dict(qid="A17", category="V", content_type="multipage_table", docs=["pd_cts", "pd"], author="ws4-owner",
         question="The compliance test specification describes its coverage in terms of older PD revisions. "
                  "Which of the timer traceability rows are stale, and what should the agent tell an engineer "
                  "who follows them?",
         answer="Coverage is claimed against the chapters of the revision the document was written against "
                "(Chapter 5 held message definitions, Chapter 6 held protocol layer behaviour in PD R3.0). "
                "Two rows of the timer traceability table cite PD R3.0 V1.3 — TEST.PD.PROT.SNK.1 (Table 6-8) "
                "and TEST.PD.PWR.SRC.1 (Chapter 5.6.2). In PD R3.2 V1.2 the timers live in Table 6.8 under "
                "§6.6 and the message definitions are in Chapter 6, so the references must be mapped before "
                "they are followed, not followed blindly.",
         anchors=["table:5-9", "cts:coverage", "sec:6"]),
    dict(qid="A18", category="F", content_type="state_machine", docs=["pd"], author="ws4-owner",
         question="Which transitions in the Sink Policy Engine state diagram return the Sink to "
                  "PE_SNK_Select_Capability, and what does the Sink do in that state?",
         answer="PE_SNK_Transition_Sink returns to PE_SNK_Select_Capability once PS_RDY is received, and the "
                "Sink waits there for the next Source_Capabilities so that it can select a new operating "
                "point; PE_SNK_Select_Capability is also the state the Sink goes back to when the Source sends "
                "new capabilities.",
         anchors=["figure:F-2", "pd:8:ready"]),

    # ------------------------------------------------------------------ #
    # ws5 — evaluation, revision awareness, cross-document facts
    # ------------------------------------------------------------------ #
    dict(qid="A19", category="S", content_type="prose", docs=["typec"], author="ws5-owner",
         question="Which terminations do a Source and a Sink present on the Configuration Channel, and what do "
                  "they determine?",
         answer="A Source presents Rp on CC1 and CC2 and a Sink presents Rd. The terminations determine attach "
                "state, port orientation, current advertisement and the initial power role.",
         anchors=["typec:4.5:intro"]),
    dict(qid="A20", category="S", content_type="prose", docs=["typec"], author="ws5-owner",
         question="How does a Sink work out how much current it may draw from the CC pin?",
         answer="It attaches Rd and measures the CC voltage against the detection thresholds: vRd-USB "
                "(0.25 V-0.61 V) for Default USB power, vRd-1.5 (0.70 V-1.16 V) for 1.5 A, vRd-3.0 "
                "(1.31 V-2.04 V) for 3.0 A and vRd-3.0 at 2.45 V-3.05 V for 3.0 A with an electronically "
                "marked cable. A voltage outside every range leaves the port unattached.",
         anchors=["table:4-25", "typec:4.5:rd"]),
    dict(qid="A21", category="T", content_type="multipage_table", docs=["pd"], author="ws5-owner",
         question="Which EPR contracts exist, and what is the maximum power of each?",
         answer="EPR fixed contracts are 28 V (140 W), 36 V (180 W) and 48 V (240 W), all at 5 A, and the EPR "
                "AVS range covers 15 V-48 V at up to 5 A (up to 240 W). The highest SPR fixed contract is "
                "20 V at 5 A (100 W).",
         anchors=["table:7.7"]),
    dict(qid="A22", category="T", content_type="test_procedure", docs=["scc_cts"], author="ws5-owner",
         question="Under which device conditions is the Shared Capacity Charger interoperability test run?",
         answer="The device under test is a Source with two or more ports that share a total capacity; it is "
                "run when the charger advertises a shared capacity, when at least two Sinks of different "
                "capability are attached (one able to request a full single-port contract), at 25 °C ± 5 °C, "
                "with electronically marked cable assemblies rated above 3 A.",
         anchors=["table:2-1", "scc:conditions"]),
    dict(qid="A23", category="T", content_type="multipage_table", docs=["typec"], author="ws5-owner",
         question="What current may each row of the USB Type-C configuration matrix carry?",
         answer="A Source limited to USB 2.0 advertises 1.5 A, or 3.0 A with an e-marked cable; a Source "
                "supporting USB 3.2 or USB4 advertises 3.0 A up to 5 A with an e-marked cable; a USB 2.0 Sink "
                "may draw up to 3.0 A; a dual-role port supporting USB4 may carry up to 5 A with an e-marked "
                "cable.",
         anchors=["table:4-9", "table:4-24"]),
    dict(qid="A24", category="F", content_type="state_machine", docs=["typec", "typec_fts"], author="ws5-owner",
         question="Which timer separates a Try.SRC port from one that takes the source role immediately, and "
                  "which test verifies it?",
         answer="tDRPTryWait. A port in Try.SRC waits in AttachWait.SRC for tDRPTryWait before it takes the "
                "source role when it is attached to another dual-role port, and TEST.TYPEC.STATE.TRYSRC.1 "
                "verifies this against Type-C §4.5.2.2.5 (TEST.TYPEC.STATE.TRYSNK.1 is the mirror test).",
         anchors=["figure:4-12", "typec:try_states", "table:4-2"]),
    dict(qid="A25", category="X", content_type="prose", docs=["pd_cts", "pd"], author="ws5-owner",
         question="Which test case prefix covers the policy engine area, and which timers do those tests "
                  "depend on?",
         answer="Policy engine coverage uses the TEST.PD.PE prefix (the coverage table lists Protocol "
                "TEST.PD.PROT, Power TEST.PD.PWR, Policy engine TEST.PD.PE, Alternate modes TEST.PD.AM and "
                "Common COMMON.CHECK.PD). TEST.PD.PE.SRC.2 exercises tFirstSourceCap, whose limits (100 ms "
                "minimum, 200 ms maximum) are in PD Table 6.8.",
         anchors=["table:3-1", "table:5-9", "table:6.8"]),
    dict(qid="A26", category="S", content_type="prose", docs=["tech2"], author="ws5-owner",
         question="What ends power transfer in Tech 2, and what does the Power Transmitter do afterwards?",
         answer="Power transfer ends on an End Power Transfer packet, on a control error that stays outside the "
                "allowed range, or on loss of communication with the Receiver. The Transmitter then returns to "
                "Ping and may start a new detection cycle.",
         anchors=["tech2:power_transfer_end", "table:3@book6"]),
]

#: 20 of the 60 questions are held out for the reviewer pass: they are not used while
#: tuning the pipeline and are reported separately (see ``eval/results/held_out.json``).
HELD_OUT_IDS = {
    "S3", "S6", "T1", "T2", "T4", "T5", "F2", "F5", "X1", "X5",
    "C1", "G3", "A01", "A04", "A07", "A10", "A13", "A17", "A21", "A24",
}
