# DIAGRAMS.md — how the system is put together

Seven diagrams, one per layer, plus the worked example trace. They are Mermaid,
so they render on GitHub and in most editors, and they were kept in the same
naming as the code so a reader can jump from a box to a module.

---

## 1. System overview — five workstreams, one data ladder

```mermaid
flowchart TB
  subgraph WS1["WS1 · Document AI & OCR"]
    RAW[("data/raw/<doc_id>/*.pdf<br/>licensed, git-ignored")]
    PARSER["ingest/parser.py<br/>typed elements + page provenance"]
    TABLES["ingest/tables.py<br/>cell grid + multi-page stitching"]
    FIGURES["ingest/figures.py<br/>vector draw-model / raster crop"]
    OCR["ingest/ocr.py<br/>pymupdf-text · rapidocr · Unlimited-OCR"]
    VLM["ingest/vlm.py<br/>HeuristicVLM · Ollama · OpenAI-compatible"]
    RAW --> PARSER --> TABLES --> FIGURES
    OCR -. "no text layer" .-> PARSER
    VLM -. "figure → description" .-> FIGURES
  end

  subgraph WS2["WS2 · Multimodal retrieval"]
    CHUNKER["index/chunker.py<br/>parent section → child chunks"]
    BM25["index/bm25.py"]
    DENSE["index/dense.py + index/embeddings.py"]
    PAGES["index/page_index.py<br/>page-image vectors"]
    RRF["index/search.py<br/>reciprocal rank fusion"]
    RERANK["index/reranker.py"]
    CHUNKER --> BM25
    CHUNKER --> DENSE
    PAGES --> RRF
    BM25 --> RRF
    DENSE --> RRF
    RRF --> RERANK
  end

  subgraph WS3["WS3 · Knowledge graph"]
    RULES["kg/rules.py<br/>test IDs, timers, states, cross-refs"]
    LLMEX["kg/llm_extractor.py<br/>optional, off by default"]
    SCHEMA["kg/schema.py<br/>ontology + validate_edge"]
    MERGE["kg/resolve.py<br/>entity merge 94 merges"]
    STORE["kg/store.py<br/>nodes · edges · provenance"]
    RULES --> SCHEMA --> MERGE --> STORE
    LLMEX -.-> MERGE
  end

  subgraph WS4["WS4 · Agentic RAG"]
    PLANNER["agent/planner.py<br/>classify · decompose · plan steps"]
    TOOLS["agent/tools.py<br/>7 tools + MockToolkit"]
    LOOP["agent/loop.py<br/>act → reflect → verify → answer/decline"]
    VERIFY["agent/verifier.py<br/>locator + support check"]
    PROMPTS["agent/prompts.py"]
    CLI["agent/cli.py<br/>specqa ask|…"]
    SERVER["server/app.py + static/index.html<br/>demo UI & HTTP API"]
    PLANNER --> LOOP
    LOOP --> TOOLS --> VERIFY --> LOOP
    PROMPTS --> LOOP
    LOOP --> CLI
    LOOP --> SERVER
  end

  subgraph WS5["WS5 · Evaluation & integration"]
    GOLD[("eval/gold.jsonl<br/>60 questions, 12 per member")]
    HARNESS["eval/run_eval.py<br/>A–F × content type"]
    METRICS["eval/metrics.py"]
    STRUCT["eval/structure.py<br/>table cells / figure structure"]
    KGP["eval/kg_precision.py<br/>triple audit"]
    REPORT[("eval/results/report.md<br/>summary · matrix · error analysis")]
    GOLD --> HARNESS --> METRICS --> REPORT
    STRUCT --> REPORT
    KGP --> REPORT
  end

  PARSER -- "elements.jsonl / tables.json / figures.json / pages" --> CHUNKER
  PARSER -- "typed elements" --> RULES
  TABLES -- "requirement rows, test steps" --> RULES
  FIGURES -- "states, transitions, timers" --> RULES
  RERANK -- "SearchHit" --> TOOLS
  STORE -- "GraphResult" --> TOOLS
  HARNESS -- "same agents as `specqa ask`" --> LOOP

  classDef store fill:#eef,stroke:#447
  class RAW,GOLD,REPORT store
```

Everything between the workstreams is one of the dataclasses in
`common/models.py`; nothing reaches across a boundary by importing an internal
function of another workstream (see `INTERFACES.md §1`).

---

## 2. WS1 — from PDF bytes to typed elements with page provenance

```mermaid
flowchart LR
  A["PDF page"] --> B{"text layer?"}
  B -- yes --> C["pymupdf blocks + spans<br/>bbox, font, size"]
  B -- "no (scan / raster figure)" --> D["rasterise 220 dpi"] --> E["OCR backend<br/>pymupdf-text | rapidocr | Unlimited-OCR"]
  C --> F["block classifier<br/>heading / paragraph / bullet /<br/>table / figure / caption / note"]
  E --> F
  F --> G["boilerplate stripper<br/>running headers, footers, page numbers"]
  G --> H{"caption regex<br/>'Table 6.8.' / 'Table 4-27' / 'Table 3'"}
  H -- table --> I["cell grid reconstruction"]
  I --> J{"continues on next page?"}
  J -- yes --> K["stitch: one Table, pages=[p, p+1],<br/>header repeated"]
  J -- no --> L["Table"]
  H -- figure --> M["vector draw-model<br/>scattered text → labels"]
  M --> N["HeuristicVLM structure:<br/>states, transitions, signals, timers"]
  N --> O["Figure"]
  G --> P["Element JSONL<br/>element_id = doc_id:type:n"]
  K --> P
  L --> P
  O --> P
  P --> Q["render page PNGs<br/>PageImage[] for the vision path"]
```

Namespacing note: numbering differs per book (PD writes `Table 6.8.`, Type-C
`Table 4-27`, Tech 2 restarts at `Table 1` per book), so every id is prefixed
with the document — `pd:table:tbl_3`, `tech2:figure:2`. Nothing downstream may
assume a number is unique across documents.

---

## 3. WS2 — hybrid retrieval and the page (vision) path

```mermaid
flowchart TB
  E["elements.jsonl"] --> CH["chunker<br/>parent section + packed children"]
  CH --> CI[("chunks.jsonl")]
  CI --> BM["BM25Index<br/>1131 terms"]
  CI --> DE["DenseIndex<br/>HashingEmbedding(384) or ST model"]
  PG["pages/*.png"] --> PE["page embedder<br/>hashing-pages or ColQwen2.5-style MaxSim"]
  PE --> PI[("index/pages")]
  Q["query"] --> BM
  Q --> DE
  Q --> PI
  BM -- "lexical scores" --> RRF["reciprocal rank fusion"]
  DE -- "semantic scores" --> RRF
  PI -- "page patches" --> RRF
  RRF --> FLT["Filters<br/>docs · tables_only · figures_only"]
  FLT --> RR["reranker<br/>identifier match · coverage · proximity"]
  RR --> HIT["SearchHit[]<br/>doc · revision · §section · page(s) · element_ids"]
```

The page path exists because a waveform or a state diagram is not a sentence:
`search_pages` returns the page image (and its patch boxes) so the agent can
answer from layout rather than from a description of layout. Config C is that
path alone, which is the capstone-04 vision-first approach; the interesting
result is that it is *much* weaker than fusion on this corpus (`eval/results/`).

---

## 4. WS3 — requirements, timers, states and test cases in one graph

```mermaid
flowchart LR
  subgraph sources["what the rules read"]
    S1["Prose: 'the Source shall …'"]
    S2["Bit-field tables"]
    S3["Multi-page parameter tables"]
    S4["Test-procedure tables<br/>TEST.PD.PROT.SRC.3 [.. #2]"]
    S5["State-machine figures<br/>PE_SRC_* → PE_SRC_*"]
    S6["Timing figures<br/>tSenderResponse"]
  end
  S1 --> RC["Requirement + Check"]
  S2 --> RC
  S3 --> RC
  S4 --> TC["TestCase<br/>TEST.PD.PROT.SRC.3"]
  S5 --> ST["State + TRANSITIONS_TO"]
  S5 --> TM["Timer / Counter<br/>STARTS_TIMER"]
  S6 --> TM
  RC --> V["validate_edge()<br/>src/dst type from KG_SCHEMA"]
  TC --> V
  ST --> V
  TM --> V
  V -- "5 rejected" --> REJ[("rejections.jsonl<br/>with the reason")]
  V -- "kept" --> M["EntityResolver<br/>alias + similarity merge · 94 merges"]
  M --> G[("graph.sqlite + nodes.jsonl + edges.jsonl")]
  G --> VZ["graph.graphml, graph.cypher,<br/>subgraph_*.mmd (Mermaid)"]
  G --> XD{"cross-document?"}
  XD -- "134 edges" --> STALE{"points at an older revision?"}
  STALE -- yes --> FL["Edge.stale = True<br/>flagged, never followed silently"]
```

Every edge carries `doc_id`, `page`, `section_number`, `extraction_method` and
`confidence`; `evaluate_edge_provenance` scores that coverage and
`audit_triples` scores schema validity over `triple_sample.jsonl`. The
test-spec-versus-base-spec trap (the CTS revisions are older than the
specifications they test) is exactly what `stale` is for.

---

## 5. WS4 — the agent loop, and where an answer can be refused

```mermaid
flowchart TB
  Q["question"] --> P["planner<br/>type: single_hop | table | figure | comparison |<br/>cross_document | test_procedure | timeline"]
  P --> ST["steps: search_text → graph_query → get_table /<br/>get_figure → search_pages"]
  ST --> ACT["act: run the tool<br/>log tool.call"]
  ACT --> GAP{"gaps?<br/>uncovered terms · missing table ·<br/>sub-question unanswered"}
  GAP -- "yes, rounds < reflect_rounds" --> RF["reflection: reformulate,<br/>try search_pages / graph"]
  RF --> ACT
  GAP -- no --> DEC{"decline decision<br/>no evidence · identifier absent from the whole corpus ·<br/>false premise contradicted · nothing on topic"}
  DEC -- decline --> REF["INSUFFICIENT + reason<br/>log agent.decline, exit code 3"]
  DEC -- answer --> CMP["compose from evidence<br/>ExtractiveLLM / Ollama"]
  CMP --> VER["CitationVerifier<br/>1 locator must be evidence the tools returned<br/>2 sentence tokens, ids and numbers in the cited hit"]
  VER --> OUT["cited answer + AgentTrace<br/>log agent.answer"]
  OUT --> UI["specqa ask / /api/ask / the worked example"]
```

Two properties matter more than accuracy here:

1. **A decline is a first-class outcome.** `--trace` shows the reason; the CLI
   exits `3`; the evaluation scores it as correct when the question is
   unanswerable.
2. **The verifier can only remove claims, never add them.** If verification
   empties the answer, the run becomes a decline rather than an uncited answer —
   the guard that caught the citation-marker bug described in `TEAM.md`.

---

## 6. WS5 — what is measured, against what

```mermaid
flowchart LR
  GQ[("eval/gold.jsonl<br/>60 q · 41 dev / 19 held-out<br/>categories S T F X C G V U")]
  for each of ["A naive dense", "B +hybrid+rerank", "C vision only", "D fused + tables/figures", "E +KG tool", "F +agent loop"]: CFG["config"]
  CFG --> RUN["run_eval.py --configs A,…,F<br/>same gold set, same corpus"]
  GQ --> RUN
  RUN --> M1["retrieval<br/>recall@5/@10 · MRR · nDCG@5"]
  RUN --> M2["answer<br/>correctness · faithfulness · citation accuracy"]
  RUN --> M3["honesty<br/>refusal accuracy · false answers · over-refusals"]
  RUN --> M4["structure<br/>table cell acc. · figure structure acc."]
  RUN --> M5["graph<br/>triple precision · edge provenance"]
  M1 --> AGG["aggregate by category, by content type,<br/>per config, dev vs held-out"]
  M2 --> AGG
  M3 --> AGG
  M4 --> AGG
  M5 --> AGG
  AGG --> REP[("eval/results/<br/>report.md · summary.json · matrix.json<br/>answers_*.jsonl · traces_*.jsonl · error_analysis.json")]
  AGG --> ABL["scripts/ablate.py → ablation.json<br/>7 single-component switch-offs"]
  REP --> ERR["error_analysis.py stages:<br/>retrieval miss → wrong evidence → unsupported claim →<br/>citation miss → over-refusal → false answer"]
```

`a gold question the corpus cannot answer` is part of the gold set on purpose
(category U): a system that never declines cannot score 1.000 on refusal
accuracy, and one that declines everything cannot answer the other 57.

---

## 7. Deployment — one process, two entry points

```mermaid
flowchart LR
  subgraph host["one machine, no GPU needed"]
    CLI2["specqa ask …<br/>CLI, exit 0/2/3"]
    API["specqa serve<br/>ThreadingHTTPServer on 0.0.0.0:8000"]
    UI2["server/static/index.html<br/>question · answer · citations · trace · evidence"]
    API --> UI2
  end
  IDX[("data/index")]
  GR[("data/graph")]
  PR[("data/processed")]
  IDX --> CLI2
  GR --> CLI2
  PR --> CLI2
  IDX --> API
  GR --> API
  PR --> API
  LLM{"LLM backend<br/>extractive (default) · Ollama · OpenAI-compatible"}
  CLI2 -.-> LLM
  API -.-> LLM
  API -. "reads" .-> SUM[("eval/results/summary.json<br/>shown in /api/stats")]
```

The server binds `0.0.0.0`, performs no Host/Origin check and sends no
framing headers, so it also works behind a preview proxy. It is stateless
apart from the lazily built index/graph/agent, and the whole thing is
one `python3 -m specintel.agent.cli serve` process.

---

## 8. The worked example, as a sequence

Question: *"What are the limits of tSenderResponse in Table 6.8?"* — the trace
below is the one captured from `/api/ask` and reproduced in
`docs/WORKED_EXAMPLE.md`.

```mermaid
sequenceDiagram
  autonumber
  participant U as user
  participant A as SpecAgent (agent/loop.py)
  participant P as Planner
  participant T as Toolkit (agent/tools.py)
  participant I as SearchIndex (WS2)
  participant G as GraphStore (WS3)
  participant V as CitationVerifier

  U->>A: ask("What are the limits of tSenderResponse in Table 6.8?")
  A->>P: plan(question)
  P-->>A: type=table · identifiers=[tSenderResponse] · steps=[search_text, graph_query, get_table]
  A->>T: search_text("tSenderResponse", k=8)
  T->>I: BM25 ⊕ dense → RRF → rerank
  I-->>T: hits: pd §6.6 p.4 Table 6.8 (table row), pd §6.6 p.3, …
  T-->>A: ToolResult(hits=8)
  A->>T: graph_query(kind=neighbors, name="tSenderResponse")
  T->>G: neighbors("Timer:tsenderresponse", depth=1)
  G-->>T: DEFINED_IN → Table 6.8, Section 6.6, Section 8; STARTS_TIMER ← PE_SRC_NegotiateCapability
  T-->>A: GraphResult(text="Timer: tSenderResponse (pd §6 p.4)")
  A->>T: get_table("pd:table:tbl_3")
  T-->>A: full Markdown grid, pages=[3,4]
  A->>A: dedup evidence · compose from the table rows
  A->>V: verify(answer, evidence, citations)
  V-->>A: 1 claim, locator ok, numbers 24 and 30 present in the cited table
  A-->>U: "Table 6.8 gives tSenderResponse a minimum of 24 ms and a maximum of 30 ms. [pd, §6.6, p.4, Table 6.8]"
```

Every hop in that diagram emits exactly one structured log line (`tool.call`,
`agent.step`, `agent.answer`), which is what makes the PDF page → answer path
auditable after the fact rather than only in a demo.
