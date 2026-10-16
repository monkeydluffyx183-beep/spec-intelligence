# Knowledge graph ontology (WS3)

Generated from `src/specintel/kg/schema.py` — do not edit by hand.
Every edge carries provenance: `doc_id`, `revision`, `section_number`, `page`,
`element_id`, `extraction_method` (rule | llm | vlm), `confidence`.

## Node types

| Node type | Meaning |
| --- | --- |
| `Document` | one of the seven source documents, with its revision |
| `Section` | a numbered section or appendix |
| `Table` | a table, including stitched multi-page tables |
| `Figure` | a figure or diagram, with its structured JSON when available |
| `Message` | USB PD control/data/extended message (Source_Capabilities, PS_RDY, Enter_USB …) |
| `DataObject` | PDO/APDO/RDO/VDO type (Fixed Supply, PPS APDO, AVS APDO …) |
| `Field` | a bit field of a data object (B31..30 PDO Type …) |
| `State` | a policy-engine, protocol-layer or Type-C connection state |
| `Timer` | t… timer with min/max/unit when the document states them |
| `Counter` | n… counter with min/max |
| `Parameter` | voltage, current or power limit such as vRd-1.5 |
| `Signal` | VBUS, VCONN and other electrical signals |
| `Pin` | CC1, CC2 and other connector pins |
| `Role` | Source, Sink, DRP, cable plug, eMarker, Tech 2 transmitter/receiver |
| `TestCase` | a compliance test case or test description (TEST.PD.*, TD 4.2.4, PTX.CPX.*) |
| `Check` | one check inside a test case ([TEST.PD.PROT.SRC.3#2], [TD.4.2.4.V.3]) |
| `Assertion` | a base-specification requirement identifier or a "shall" statement |
| `VIFField` | an applicability condition used to select tests |
| `PowerProfile` | BPP, EPP, MPP and other power profiles |
| `Tech2Phase` | Ping, Configuration, Negotiation, Power Transfer |
| `Tech2Packet` | Signal Strength, Control Error, End Power Transfer packets |
| `Router` | USB4 Host Router or Device Router |
| `Adapter` | USB4 Lane Adapter or Protocol Adapter |
| `Tunnel` | a tunnelled protocol path |
| `Protocol` |  |
| `ECN` | an engineering change notice named in the revision history |
| `Revision` | a document revision (R3.1 V1.6, Release 2.4 …) |

## Edge types

| Relation | From | To | Meaning | Edge properties |
| --- | --- | --- | --- | --- |
| `DEFINED_IN` | any | Section, Table, Figure | the entity is defined in this section, table or figure | — |
| `REFERENCES` | any | Section, Table, Figure, Document, ECN, Revision | a cross-reference, possibly to another document or an older revision | — |
| `SENT_BY` | Message, Tech2Packet | Role, State, Document | which role or state sends the message | direction, condition |
| `RESPONDS_TO` | Message, Tech2Packet | Message, Tech2Packet | this message is the answer to that message | within_timer |
| `CONTAINS` | any | Field, DataObject, Message, Section, Tunnel, Adapter | payload or hierarchy containment | — |
| `TRANSITIONS_TO` | State | State | state machine transition | trigger, condition, actions |
| `STARTS_TIMER` | State, TestCase, Section | Timer | this state or test starts the timer | condition |
| `STOPS_TIMER` | State, TestCase, Section | Timer | this state or test stops the timer | condition |
| `HAS_FIELD` | DataObject, Message, Tech2Packet, Table | Field | the data object is laid out with this field | — |
| `HAS_VALUE` | Field, Timer, Counter, Parameter, PowerProfile | Parameter, Timer, Counter, Field | the entity holds this value, limit or unit | value, min, max, unit |
| `TESTS` | TestCase | Assertion, Requirement, Message, State, Timer, Parameter, DataObject, PowerProfile, Tech2Packet | the test case exercises this requirement or entity | — |
| `VERIFIES` | Check, TestCase | Assertion, Requirement, Timer, State, Message, Parameter, PowerProfile | the check verifies this requirement with a stated limit | limit, condition |
| `APPLIES_WHEN` | TestCase, Check, Assertion | VIFField | the test only applies under this condition | value |
| `TUNNELS` | Adapter, Tunnel | Protocol, Adapter, Document | this USB4 adapter tunnels that protocol | section, adapter_id |
| `SUPERSEDES` | Revision, Document, ECN | Revision, Document, ECN | the newer revision replaces the older one | reason |
| `HAS_REVISION` | Document | Revision | the document has this revision | — |
| `PART_OF` | Section, Table, Figure, Field, Check | Section, Document, Table, DataObject, TestCase | hierarchy: this element belongs to that one | — |
| `DEPENDS_ON` | TestCase, Check, Message, State | Message, State, Timer, Section, DataObject | the entity cannot work without that one | reason |
| `HAS_STATE` | Document, Section, Figure | State | the state machine contains this state | — |

## Validation

* an unknown node or edge type is rejected;
* an edge whose endpoints are not in the declared types is rejected
  (for example `TRANSITIONS_TO` from a `State` to a `Timer`);
* an edge without provenance, or with a confidence outside `[0, 1]`, is rejected;
* every rejection is written to `data/graph/rejections.jsonl` with its reason and
  counted in the build report.
