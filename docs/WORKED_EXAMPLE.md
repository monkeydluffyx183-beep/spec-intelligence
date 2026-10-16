# Worked example — from a PDF page to a cited answer

One question, followed from the bytes of the source PDF to the citation a reviewer
can open. Everything below is the real output of the committed code on the
current corpus (synthetic stand-in with the same layout traps as the licensed
specifications), captured with:

```bash
PYTHONPATH=src python3 -m specintel.agent.cli ask \
  "What are the limits of tSenderResponse in Table 6.8?" --trace
```

---

## Step 1 — the page exists, and the ingester read it

The question names a table that exists in the Power Delivery book. The ingest
stage recorded exactly what it took from each PDF page (`event=parse.file`,
`event=tables.file`, `event=pages.render`):

```
event=parse.file    doc=pd duration_s=0.088 elements=223 file=USB_PD_R3.2_V1.2_2026_08_19.pdf status=ok
event=tables.file   doc=pd duration_s=0.673 parts=12 stitched=2 tables=10
event=tables.stitched doc=pd table=6.8 pages=[3,4] rows=14 cols=5
```

Table 6.8 is one of the two **stitched** tables: it starts on page 3 and its last
rows are on page 4, so the ingester emits a single `Table` object with
`pages=[3, 4]` and repeats the header row, rather than two unrelated fragments:

```
| Timer | Min | Max | Unit | Definition |
| --- | --- | --- | --- | --- |
| tSenderResponse | 24 | 30 | ms | Maximum time a Sender waits before it starts to send a response to a received message |
| tReceiverResponse | 0 | 15 | ms | Time a Receiver has to respond with GoodCRC |
| tPSTransition | 450 | 550 | ms | Time for the power supply to reach the new operating point after Accept |
...
```

Its element ids are namespaced by document (`pd:table:tbl_e94857a647da`), because
`Table 6.8` also exists in Type-C and in Tech 2 with different content.

## Step 2 — the chunk lands in the index, and the graph knows what defines what

`data/index/build_report.json` records `pd` as 182 chunks (10 tables, 3 figures,
16 parent sections); `data/graph/edges.jsonl` records the relation the rules
extracted from the same elements:

```
Timer:tsenderresponse  -[DEFINED_IN]->  Table:pd_table_6_8   pd §6.6 p.3
Timer:tsenderresponse  -[DEFINED_IN]->  Section:pd_6_6        pd §6.6 p.3
State:pe_src_negotiate_capability -[STARTS_TIMER]-> Timer:tsenderresponse  pd §8 p.6
```

Every edge carries document, section, page, extraction method and confidence; the
graph refuses an edge without them (5 rejections in the current build, all
`DEFINED_IN cannot end at Document`).

## Step 3 — the agent's run, step by step

```
ANSWERED
Timer: tSenderResponse; Min: 24; Max: 30; Unit: ms; Definition: Maximum time a
Sender waits before it starts to send a response to a received message
[pd, §6.6, p.3, Table 6.8] …

  cited sources
    [pd, §6.6, p.3, Table 6.8]

  trace
    plan      : type=table | sub-questions=1 | expected_documents=any
                step 1: search_text (find the table that holds the value)
                step 2: get_table (read the whole table, including any continuation page)
                step 3: search_text (surrounding prose for context)
    search_text      8 hits top=3.352 0.006s  [1] pd_cts §5 p.3 (table, score 3.352)
                                              Table 5-9. Test cases that reference PD timers
    search_text      8 hits top=1.126 0.005s  [1] pd_cts §5 p.3 (table, score 1.126)
    reflect          0 hits 0.000s            the question needs a table but no table was read
    search_text      8 hits top=1.126 0.005s  [1] pd_cts §5 p.3 (table, score 1.126)
    resolve_entity   0 hits 0.005s            Timer: tSenderResponse (pd §6.6 p.3)
    graph_query     12 hits top=1.000 0.010s  70 node(s), 25 edge(s)
                                              Timer: tSenderResponse (pd §6.6 p.3)
                                              Table: Table 6.8 (pd §6.6 p.3)
                                              Section: §6.6
    get_table        1 hits top=1.000 0.000s  | Timer | Min | Max | Unit | Definition |
                                              | tSenderResponse | 24 | 30 | ms | M…
    generate         0 hits 2037 tok 0.000s   Timer: tSenderResponse; Min: 24; Max: 30; …

    tokens: 2037   latency: 0.04s   model: extractive   tools: 6
```

What the trace shows, and why each part is there:

1. **Text search alone gets the wrong table first.** `search_text` ranks `pd_cts`
   Table 5-9 — a *test* table that lists `tSenderResponse` among five test cases —
   above the table that defines it, because that row mentions the timer twice and
   the reference `PD R3.2 V1.2 Table 6.8` as well. This is the superseded-revision
   trap in the assignment: Table 5-9's Sink row points at **PD R3.0 V1.3 Table
   6-8**, an older revision of the same table. A pipeline that trusted the first
   hit would answer from the wrong document and cite an obsolete revision.
2. **Reflection notices the gap** — "the question needs a table but no table was
   read" — because the planner classified the question as a table question and no
   `get_table` call has produced rows yet.
3. **The graph resolves the identifier**: `resolve_entity` finds
   `Timer:tsenderresponse` in `pd §6.6 p.3`, and `graph_query` returns the
   `DEFINED_IN` edges to `Table 6.8`, to `§6.6` and to `§8` — with provenance.
4. **The follow-up read turns the relation into content**: the agent calls
   `get_table(doc_id="pd", number="6.8")` and gets the stitched grid, header
   repeated, pages 3–4. This is the step that makes the answer *specific*: the
   graph can say which table defines a timer, only the table can say 24 ms to
   30 ms.
5. **The composer writes from that table**, and the citation verifier checks two
   things before the answer is returned: that `pd §6.6 p.3` is a page the tools
   actually returned, and that every number in the sentence (`24`, `30`) occurs in
   the cited table. Had the numbers not been there, the claim — not the
   citation — would have been dropped.

The supporting claims that follow in the answer come from the test
specification (`pd_cts`), which is correct and useful: they say which test case
verifies the timer (`TEST.PD.PROT.SRC.3`) and how it is checked. Where such a
claim cites an older revision, the graph has it flagged:

```
Section:pd_cts_3 -[REFERENCES]-> Section:pd_5   pd_cts p.2   stale=true
Section:pd_cts_3 -[REFERENCES]-> Section:pd_6   pd_cts p.2   stale=true
```

(`Coverage is expressed in terms of the chapters of the revision that this
document was written against…` — the CTS predates the specification it tests, so
its chapter references point at older revisions. They are marked `stale=true`
rather than followed silently, and `SPECINTEL_ABLATE_STALE=1` switches the
flagging off for the ablation that shows what a reviewer would otherwise get.)

## Step 4 — the same answer over HTTP

```
$ curl -s localhost:8000/api/ask -H 'content-type: application/json' \
    -d '{"question":"What are the limits of tSenderResponse in Table 6.8?"}' | jq '.citations, .tool_calls, .latency_s'
[
  "[pd, R3.2 V1.2 (19 Aug 2026), §6.6, p.3, Table 6.8]",
  "[pd_cts, Revision Q3 2026 (31 May 2026), §5, p.3, Table 5-9]"
]
6
0.039
```

## Step 5 — the log lines that make it auditable

```
event=parse.file     doc=pd elements=223 file=USB_PD_R3.2_V1.2_2026_08_19.pdf status=ok
event=tables.stitched doc=pd table=6.8 pages=[3,4] rows=14 cols=5
event=graph.doc      doc=pd nodes=145 edges=364 tables=10 figures=3
event=index.text     chunks=478 bm25_terms=1131 dense_dims=384 embedder=hashing reranker=lexical
event=tool.call      tool=search_text n_hits=8  top_score=3.352 duration_s=0.006
event=tool.call      tool=graph_query n_hits=12 top_score=1.000 duration_s=0.010
event=tool.call      tool=get_table   n_hits=1  top_score=1.000 duration_s=0.000
event=agent.answer   citations=1 evidence=29 tool_calls=6 duration_s=0.039 declined=False
event=cli.done       command=ask status=ok duration_s=0.262
```

Same format on every line (`INTERFACES.md §4`), with the run id on the agent
events, so the whole path — PDF page → element → chunk → graph edge → tool call →
citation — can be replayed from the log without the demo.

## What this example does *not* show

* **A real licensed PDF.** The corpus here is the generated stand-in
  (`README.md § Reproducibility`); the same commands run unchanged against
  `data/raw/<doc_id>/*.pdf` once the licensed files are present.
* **A GPU model.** The composer is the deterministic extractive backend, so the
  wording is assembled from evidence sentences rather than generated. The
  citation and verification path is identical for a local model
  (`get_llm(config, "ollama")`), which is the point of the adapters.
* **A raster figure.** `tech2 F4` has no text layer, so its structure scores
  `None` until an OCR/VLM pass runs — `docs/OCR_BENCHMARK.md` is where that gap is
  measured.
