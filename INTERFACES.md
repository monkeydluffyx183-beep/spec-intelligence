# INTERFACES.md — the contracts

Every workstream talks to the others only through the objects in
`src/specintel/common/models.py` and the files listed here. This document is
written **before** the code that consumes it and is the tie-breaker whenever two
modules disagree: the dataclass field names, the artefact paths and the log line
below are the interface. `tests/test_contracts.py` fails the build if a field or
the log format drifts.

---

## 1. The data ladder

```
raw PDF ──WS1──▶ Element / Table / Figure / PageImage ──WS2──▶ Chunk ──▶ SearchHit
                            │                                                  │
                            └──────────────WS3──▶ Node / Edge ◀────────────────┘
                                                       │
                                        WS4 Citation ◀─┴─ ToolResult / AgentTrace
                                                       │
                                              WS5 GoldQuestion → QuestionResult → EvalSummary
```

| stage | produced by | consumed by | where it lands |
|---|---|---|---|
| `Element[]` | WS1 `ingest/parser.py` | WS2 chunker, WS3 rules | `data/processed/<doc>/elements.jsonl` |
| `Table[]` | WS1 `ingest/tables.py` | WS2 table rows, WS3 requirements, WS5 cell scoring | `tables.json` + `tables.md` |
| `Figure[]` | WS1 `ingest/figures.py` | WS2 figure text, WS3 states/timers, WS5 figure scoring | `figures.json` (+ `figures/*.png`) |
| `PageImage[]` | WS1 `ingest/pipeline.py` | WS2 page index (vision path) | `pages.jsonl` + `pages/*.png` |
| `Chunk[]` | WS2 `index/chunker.py` | WS4 tools, WS5 retrieval metrics | `data/index/chunks.jsonl` |
| index files | WS2 `index/build.py` | `specqa chunks`, `/api/search` | `bm25.json`, `dense.npy`, `dense.json`, `pages/` |
| `Node`/`Edge` | WS3 `kg/build.py` | WS4 `graph_query`, WS5 triple audit | `data/graph/{nodes,edges}.jsonl`, `graph.sqlite`, `graph.graphml`, `graph.cypher` |
| `Citation` | WS4 `agent/tools.py` | WS4 verifier, WS5 citation metrics, the demo UI | inside `AgentTrace` and `AgentAnswer` |
| `AgentTrace` | WS4 `agent/loop.py` | WS5 traces, the PDF-page→answer worked example | `eval/results/traces_<config>.jsonl` |
| `GoldQuestion` | WS5 (authored by every member) | WS5 harness | `eval/gold.jsonl` |

`data/` is git-ignored (licensed material). Only the gold set, the config, the
benchmark summaries and the small `samples/` excerpts are committed.

---

## 2. Field-level contracts

Names below are exact; `ContractMixin` gives every dataclass `to_dict()`,
`to_json()`, `from_dict()` and (for the JSONL types) `to_jsonl()`/`read_jsonl()`.

* **`Element`** — `element_id, doc_id, revision, element_type, page, text=…,
  markdown=…, section_number=…, section_title=…, section_path=[…], bbox=…,
  image_path=None, extractor="pymupdf", order=0, table_id=…, figure_id=…,
  caption=…, page_count=1, parent_element_id=…, meta={}`.
  `element_type ∈ {heading, paragraph, bullet, table, table_row, figure,
  caption, note, code, formula, page_image}` (`ELEMENT_TYPES`).
  `element_id` is namespaced by document: `<doc_id>:<type>:<n>` — the same table
  number exists in more than one book, so the id is never global.
* **`Table`** — `table_id, doc_id, revision, number, caption, pages=[…],
  headers=[[…]], rows=[[…]], section_number=…, bbox=…, image_path=…,
  element_ids=[…], stitched=False, extractor=…`. `pages` is a **list**: a table
  that continues on the next page is one object with several pages.
  `markdown` is a property (headers repeated), not a field.
* **`Figure`** — `figure_id, doc_id, revision, number, caption, page,
  image_path=…, bbox=…, kind="unknown", description=…, structure=…,
  section_number=…, references=[…], extractor=…, hallucination_notes=[…]`.
  `kind ∈ {state_machine, timing, block, photo, chart, unknown}`.
* **`Chunk`** — `chunk_id, doc_id, revision, element_type, text, page,
  page_end=0, section_number=…, section_title=…, section_path=[…],
  element_ids=[…], parent_chunk_id=None, child_chunk_ids=[…], table_id=…,
  figure_id=…, bbox=…, n_tokens=0, meta={}`. Parent (section) chunks carry
  `meta.kind="parent_section"`; children point back through `parent_chunk_id`.
* **`SearchHit`** — `chunk_id, score, doc_id, page, page_end=0,
  section_number=…, element_type=…, text=…, source="text", bbox=…,
  element_ids=[…], table_id=…, figure_id=…, component_scores={}, meta={}`.
  `source ∈ {text, pages, graph, table, figure, fused}`. The document revision
  travels in `meta["revision"]` (kept out of the constructor so the retrieval
  contract stays small); the citation builder reads it from there.
* **`Node`** — `node_id, node_type, name, canonical_name=…, doc_id=…,
  revision=…, section_number=…, page=0, aliases=[…], properties={},
  element_ids=[…]`. `node_id = "<Type>:<slug>"` (e.g. `Timer:tsenderresponse`).
* **`Edge`** — `edge_id, src_id, dst_id, edge_type, doc_id=…, revision=…,
  section_number=…, page=0, element_id=…, extraction_method="rule",
  confidence=1.0, evidence=…, stale=False, properties={}`. **Provenance is
  mandatory**: `validate_edge(...)` rejects an edge without document, section or
  page. `extraction_method ∈ {rule, llm, vlm}`; `stale=True` marks a
  cross-reference that points at an older revision than the target document's.
* **`GraphQuery`** — `kind, name=…, node_id=…, src=…, dst=…, edge_types=(…),
  node_types=(…), direction="both", depth=1, limit=25`;
  `kind ∈ {neighbors, path, lookup, subgraph, stats}`.
  `GraphResult(query, nodes=[…], edges=[…], paths=[…], notes=[…], stats={})`.
* **`Citation`** — `doc_id, revision, section_number, page, region=…,
  page_end=0, element_ids=[…], quote=…, verified=None, verification_note=…`.
  `to_marker()` renders the required format, the revision shown only when known:

  ```
  [pd, R3.2 V1.2 (19 Aug 2026), §6.6, p.4, Table 6.8]
  [typec, §4.7, p.2, prose]
  ```

* **`ToolResult`** — `tool, hits=[…], text=…, graph=…, table=…, figure=…,
  notes=[…], error=None, latency_s=0.0`; `.ok` is `error is None`,
  `.top_score()` is the best hit score or `None`.
* **`TraceStep`** — `step, kind, name=…, args={}, n_hits=0, top_score=…,
  observation=…, duration_s=0.0, tokens=0, decision=…, citations=[…],
  error=…`; `kind ∈ {plan, tool, reflect, answer, decline, note}`.
* **`AgentTrace`** — `run_id, question, question_category=…, plan=[…],
  steps=[…], answer=…, citations=[…], declined=False, decline_reason=…,
  confidence=0.0, token_usage={}, latency_s=0.0, config_name="F",
  model="extractive", started_at=…, hardware={}, warnings=[…]`.
* **`GoldQuestion`** — `qid, question, category, gold_answer, gold_evidence=[…],
  author=…, verifier=…, content_type="prose", docs=[…], answerable=True,
  notes=…, gold_citations=[…], split="dev"`.
  `gold_evidence` entries are `{doc_id, page, element_id, quote}`.
  `split ∈ {dev, held_out}` — `held_out` questions are scored, never tuned on.

---

## 3. Artefact paths

| key (`config.path(...)`) | default | contents |
|---|---|---|
| `paths.raw` | `data/raw` | `<doc_id>/*.pdf` — the licensed originals (never committed) |
| `paths.processed` | `data/processed` | `<doc_id>/{elements.jsonl,tables.json,tables.md,figures.json,pages.jsonl,ingest_stats.json,pages/*.png,figures/*}` |
| `paths.index` | `data/index` | `chunks.jsonl`, `bm25.json`, `dense.npy`, `dense.json`, `build_report.json`, `pages/` |
| `paths.graph` | `data/graph` | `graph.sqlite`, `nodes.jsonl`, `edges.jsonl`, `graph.graphml`, `graph.cypher`, `rejections.jsonl`, `stats.json`, `triple_sample.jsonl`, `subgraph_*.mmd` |
| `paths.synthetic` | `data/synthetic` | generated stand-in corpus + `manifest.json` (structure ground truth) |
| `paths.runs` | `data/runs` | per-run agent traces and benchmark scratch |
| `paths.logs` | `data/logs` | `specintel.log` |

There is **no `paths.pages` key**: page PNGs live under
`paths.processed/<doc_id>/pages/`. Page-image retrieval is indexed to
`paths.index/pages/`.

Evaluation and benchmark outputs are not addressed by `config.path`; they sit at
fixed relative paths: `eval/results/` and `benchmarks/ocr/results/`.

---

## 4. Structured logging — one format for the whole system

Every module logs through `specintel.common.logging`; nothing calls `print()` on
the answer path. One line per event:

```
<ts> <level> <logger> event=<name> key=value key=value …
```

* `<ts>` is UTC, `YYYY-MM-DDTHH:MM:SSZ`; `<level>` is the standard name padded to
  5 characters; `<logger>` is `specintel.{ws1…ws5}.<module>`.
* `event` is a dotted name (`graph.done`, `tool.call`, `eval.question`).
* Values are rendered `key=value`; strings with whitespace are quoted. Keys are
  sorted, so two lines with the same event diff cleanly.
* `LOG_LINE_RE` in `common/logging.py` is the machine-readable form of that line
  and `tests/test_contracts.py` asserts every emitted line matches it.
* `log_event(logger, "name", **kv)` emits; `log_exception(logger, "name", exc,
  **kv)` adds `error=…` and `error_type=…` and a traceback to the file handler.
  Duplicate keyword names are an error, not a silent overwrite.
* `SPECINTEL_LOG_JSON=1` switches the formatter to JSON lines without changing
  event names or fields, so log analysis works either way.

What the mandatory events carry (this is what the assignment asks to be able to
trace):

| event | module | carries |
|---|---|---|
| `ingest.start` / `ingest.done` | WS1 | `doc`, `pages`, `elements`, `tables`, `figures`, `duration_s`, `rss_mb` |
| `parse.file`, `tables.file`, `figures.file`, `pages.render` | WS1 | per-document counts, drops, headers stripped |
| `index.text`, `index.pages`, `index.report` | WS2 | `chunks`, `bm25_terms`, `dense_dims`, `pages_indexed`, timing |
| `graph.doc`, `graph.done`, `graph.rejected`, `graph.resolve` | WS3 | `nodes`, `edges`, `merges`, `rejected`, `cross_doc`, `stale`, rejection reasons |
| `tool.call` | WS4 | `tool`, `n_hits`, `top_score`, `duration_s`, `ok` |
| `agent.step` | WS4 | `run_id`, `step`, `tool`, `hits`, `evidence` |
| `agent.answer` / `agent.decline` | WS4 | `run_id`, `question`, `citations`, `evidence`, `tool_calls`, `duration_s`, `reason` |
| `eval.question` | WS5 | `config`, `qid`, `recall_at_10`, `correctness`, `declined`, `duration_s` |
| `ocr.bench.crop` | WS1 bench | `backend`, `crop`, `cer`, `wer`, `latency_s` |

The **PDF page → answer** trace is the union of `parse.file` (page count of the
PDF), `index.text` (chunk that holds the page), `tool.call`/`agent.step` (the
retrieval that found it) and `agent.answer` with its citations — all sharing the
same `doc_id`, page and `run_id`. See `DIAGRAMS.md §5` and
`docs/WORKED_EXAMPLE.md`.

---

## 5. Tool signatures (WS4)

`specqa tools` prints exactly these; a caller may rely on the names and keyword
arguments.

| tool | signature | returns |
|---|---|---|
| `search_text` | `search_text(query: str, k: int = 8, docs: list[str] \| None = None)` | `list[SearchHit]` over the text/table/figure index |
| `search_pages` | `search_pages(query: str, k: int = 5, docs: list[str] \| None = None)` | `list[SearchHit]` (page images) with `meta.image_path` and patch boxes |
| `graph_query` | `graph_query(kind: str, name='', node_id='', src='', dst='', edge_types=None, depth=1, direction='both')` | `GraphResult` |
| `get_section` | `get_section(doc_id: str, section_number: str)` | the section text plus its tables and figures |
| `get_table` | `get_table(table_id: str)` | the whole stitched `Table` (Markdown + pages) |
| `get_figure` | `get_figure(figure_id: str)` | the `Figure` with structure and description |
| `resolve_entity` | `resolve_entity(name: str)` | graph `Node`s the name resolves to, with aliases |

`MockToolkit` implements the same seven from one JSON file
(`write_mock_fixtures(path)` writes the default set), so WS5 and the UI can be
developed before the index exists. The dispatcher's own first parameter is
`tool_name`, never `name`, so a tool's `name=` keyword cannot collide with it.

## 6. CLI and HTTP surface

```
specqa ask "<question>" [--docs pd,typec,usb4,tech2] [--trace] [--json] [--quiet]
                         [--top-k N] [--max-steps N] [--budget S] [--type T] [--no-verify]
specqa tools | ingest | build-index | build-graph | eval | graph | chunks | doctor | serve
```

* exit `0` answered, `3` declined (an honest decline is a successful run),
  `2` a required artefact is missing, `1` anything else.
* `SPECINTEL_CONFIG` (or `--config`) selects the config file;
  `SPECINTEL__SECTION__KEY` overrides any config value.

HTTP (the demo UI and API, `specqa serve --host 0.0.0.0 --port 8000`):

| method | path | body / result |
|---|---|---|
| `GET` | `/` | the demo page (`server/static/index.html`) |
| `GET` | `/api/health` | status, documents ingested, model, hardware, uptime |
| `GET` | `/api/stats` | index, elements, graph and last-evaluation summary |
| `GET` | `/api/docs` | the document registry with an `ingested` flag |
| `POST` | `/api/ask` | `{question, docs?, top_k?, trace?}` → answer, citations, evidence, trace |
| `POST` | `/api/search` | `{query, k?, docs?, tables_only?, figures_only?}` → hits |
| `POST` | `/api/graph` | `{kind, name?, src?, dst?, depth?, limit?}` → nodes, edges, text |
| `OPTIONS` | any | CORS preflight |

`X-Specintel-Token` is enforced on `POST` when the server is started with a
token; `GET /api/health` stays open so a probe can see the service is alive.
The server binds `0.0.0.0`, adds no host allowlist and embeds no framing
headers, so it works behind the sandbox preview proxy unchanged.

## 7. Environment switches

| variable | effect |
|---|---|
| `SPECINTEL_CONFIG` | config file path |
| `SPECINTEL__<SECTION>__<KEY>` | override any config value, e.g. `SPECINTEL__AGENT__MAX_STEPS=12` |
| `SPECINTEL_LOG_JSON` | JSON log lines |
| `SPECINTEL_SCORE_FLOORS` | ablation: one relevance floor for every retriever (config A's naive floor) |
| `SPECINTEL_ABLATE_{BM25,RERANKER,TABLE_ROWS,FIGURE_TEXT,MULTIPART,STALE}` | switch off one component (see `docs/ABLATIONS.md`) |
| `SPECINTEL_UNLIMITED_OCR_ENDPOINT` / `_MODEL` | point the Unlimited-OCR adapter at a served GPU |
| `SPECINTEL_TUNE_{MAX_SENTENCES,PER_ITEM,FOCUS,GATE}` | composer tuning hooks used by `scripts/tune_composer.py` |

---

## 8. Models and reproducibility

Local-first, open-weight only; no hosted API is called anywhere on the
`ingest → build-graph → eval` path. Every AI component sits behind an adapter
with a deterministic CPU fallback, so a clean clone with no GPU, no network and
no model server still reproduces the numbers below:

| component | default (no model server) | optional local model |
|---|---|---|
| embeddings | `HashingEmbedding(dim=384)` — deterministic hashed bag of words | `sentence-transformers` / Ollama embeddings via `get_embedder(config, backend=…)` |
| page embeddings | `hashing-pages` over rendered page patches | ColQwen2.5-style MaxSim page vectors (capstone 04 approach) via the same adapter |
| answer composer | `ExtractiveLLM` — sentence selection with citation markers | Ollama / OpenAI-compatible / vLLM (`get_llm(config, "ollama"\|"openai")`) |
| figure structure | rule-based vector-text reconstruction | VLM adapter (`ingest/vlm.py`) |
| raster OCR | `pymupdf-text` (text layer) | Unlimited-OCR / RapidOCR adapters (`ingest/ocr.py`) |

Recorded versions live in `requirements.lock.txt` and in the `hardware` block of
every report; each run stamps `model`, `hardware`, `llm_backend` and the config
name into `summary.json`, `build_report.json` and the traces.
