# TEAM.md — who owns what, and the decisions log

## 1. Workstream ownership

Five workstreams, five owners. The names below are the roles the artefacts are
attributed to (`eval/gold.jsonl` carries an `author` and a `verifier` per
question, and the gold set is split 12 questions per owner).

| workstream | owner | owns | artefacts |
|---|---|---|---|
| WS1 Document AI & OCR (Task 2, steps 1–4) | `ws1-owner` | PDF parsing, typed elements, page provenance, table stitching, figure structure, OCR/VLM adapters, the OCR benchmark | `src/specintel/ingest/*`, `benchmarks/ocr/*`, `docs/OCR_BENCHMARK.md` |
| WS2 Multimodal retrieval (Task 2, steps 5–7) | `ws2-owner` | chunking, BM25 + dense + page indexes, fusion, reranking, filters | `src/specintel/index/*` |
| WS3 Knowledge graph (Task 3) | `ws3-owner` | ontology, rule/LLM extraction, entity resolution, provenance, exports | `src/specintel/kg/*`, `docs/KG_SCHEMA.md` |
| WS4 Agentic RAG (Task 4) | `ws4-owner` | planner, seven tools, reflection, citation verification, declines, CLI, demo API | `src/specintel/agent/*`, `src/specintel/server/*` |
| WS5 Evaluation & integration (Tasks 5–6) | `ws5-owner` | gold set, metrics, structure and triple scoring, ablations, error analysis, report, docs | `src/specintel/eval/*`, `eval/*`, `scripts/*`, `INTERFACES.md`, `DIAGRAMS.md`, this file |

Nobody edits another workstream's module to make their own metric pass. A change
that needs a field, a path or an event name goes through `INTERFACES.md` first,
as a change to the contract, and only then into the two modules on either side.

### How reviews were done here

The assignment requires every PR to be reviewed by another member. In this
environment the work was executed by one agent on one branch, so the review is
mechanised rather than social, and it is honest to say so:

* **The contract test is the reviewer of first resort.** `tests/test_contracts.py`
  fails on a changed field, a changed log format or a missing artefact path, so a
  change that would have broken another workstream's assumptions cannot merge.
* **Every workstream ships its own test module** (`test_ingest.py`,
  `test_index.py`, `test_kg.py`, `test_agent.py`, `test_eval.py`,
  `test_server.py`) and the suite runs on the fixture corpus, not on the
  licensed one, so it is a real gate.
* **Numbers are re-derived, not quoted.** The evaluation is re-run from the
  scratch corpus in the same commit that changes ingestion or retrieval
  (`TEAM.md §3, D-14`), which is what caught the stale-results problem below.
* **The checklist a human reviewer applies** is §4 — it is written as questions
  with a file to look at for each, so a second member can work through it.

If this work is taken onto a real team, each row of §1 becomes "opened by X,
reviewed by the next workstream owner": WS1↔WS2 (does the chunker rely on any
field the parser does not promise?), WS2↔WS4 (are hits self-describing enough to
cite?), WS3↔WS5 (is provenance enough to audit?), WS4↔WS5 (does the harness
score what the agent actually does?).

## 2. The five deliverables and where they live

| brief deliverable | file / command |
|---|---|
| cited answers to held-out questions | `specqa ask "…" [--trace] [--docs …]` |
| multimodal ingestion with page provenance | `src/specintel/ingest/`, `data/processed/<doc>/*` |
| OCR benchmark including Unlimited-OCR | `benchmarks/ocr/run_bench.py` → `docs/OCR_BENCHMARK.md` |
| text vs vision retrieval comparison | `eval/results/report.md` (configs B/C/D) |
| knowledge graph with provenance on every edge | `src/specintel/kg/`, `data/graph/`, `docs/KG_SCHEMA.md` |
| gold set, metrics, content-type × approach matrix, ablations, error analysis | `eval/gold.jsonl`, `src/specintel/eval/`, `scripts/ablate.py`, `eval/results/` |
| architecture, diagrams, logs, worked example | `README.md`, `DIAGRAMS.md`, `INTERFACES.md §4`, `docs/WORKED_EXAMPLE.md` |

## 3. Decisions log

D-* are technical decisions; each records the alternative that was rejected and
the evidence that settled it.

**D-01 — Real documents first, synthetic stand-ins only where the real file is
missing.** *(user decision)* The seven licensed specifications cannot be
downloaded in this environment (usb.org/WPC are unreachable), and they may not be
committed anyway. So `data/raw/<doc_id>/` stays the primary input path, and
`scripts/make_synthetic_corpus.py` generates structurally faithful stand-ins into
`data/synthetic/` so `ingest → build-graph → eval` runs from a clean clone. The
corpus generator reproduces the traps that matter (per-book numbering, stitched
tests, raster figures, superseded revisions); when the real PDFs are dropped into
`data/raw/`, the same commands ingest them unchanged.

**D-02 — All five workstreams end-to-end.** *(user decision)* No workstream is a
stub. Every deliverable in the brief has a runnable implementation and a test.

**D-03 — Everything behind an adapter with a deterministic CPU fallback.**
*(user decision)* Hashing embeddings, a rule-based extractor, an extractive
composer and a heuristic figure reader mean the whole pipeline runs with no GPU,
no model server and no network, and the same code paths accept a real local model
through `get_llm` / `get_embedder` / `get_ocr_backend` / `get_vlm`. Consequence:
the reported numbers are *reproducible*, and a GPU is an upgrade rather than a
requirement. See `INTERFACES.md §8`.

**D-04 — Local-first, no hosted API on the critical path.** The assignment
forbids paid APIs unless documented with cost; nothing on the ingest/graph/eval
path calls one. The Ollama and OpenAI-compatible clients exist for the container
that has a model server, and both are opt-in.

**D-05 — Contracts before code.** `common/models.py` + `INTERFACES.md` were
written first, `tests/test_contracts.py` enforces them, and the log format
(`<ts> <level> <logger> event=<name> key=value …`) is part of the contract rather
than a convention. Evidence: the contract test caught four field mismatches
(`Element.revision`, `Table.pages`, `Chunk.element_type`, `Edge.edge_id`) before
they reached a consumer.

**D-06 — Every id is namespaced by document.** PD writes `Table 6.8.`, Type-C
`Table 4-27`, Tech 2 restarts numbering per book; test ids come in four dialects
(`TEST.PD.PROT.SRC.3`, `[TEST.PD.PROT.SRC.3#2]`, `TD 4.2.4` / `[TD.4.2.4.V.3]`,
`PTX.CPX.CFG.S02.IDX.002`). Rejected: a global `<type>:<number>` scheme, which
silently merges tables from different books. Settled by the structure check
scoring `tech2` tables against `pd` tables of the same number.

**D-07 — A multi-page table is one object with `pages=[…]`, not N objects.**
Rejected: emitting one table per page and joining at query time, which loses the
header repetition and the "continues" evidence. The stitcher is scored directly
(2 stitched tables in `pd`, 1 in `typec_fts`).

**D-08 — Provenance is mandatory on every graph edge.** `validate_edge()` rejects
an edge without `doc_id`, `page` and a section or element, and the rejection
reason is written to `rejections.jsonl`. Rejected: making provenance optional and
filtering later — the assignment's "provenance on every edge" is a hard
requirement, and a rejected edge is more useful than an unprovable one.

**D-09 — Stale cross-references are flagged, never silently followed.** The CTS
revisions predate the specifications they test, so `Edge.stale` marks a
cross-reference to an older revision than the target document's current one, and
`SPECINTEL_ABLATE_STALE=1` switches the flagging off so the ablation can show
what a reviewer would get without it. (A test spec pointing at a superseded
requirement is a *finding*, not a retrieval problem.)

**D-10 — A backend that cannot run is reported unavailable, never scored.**
`available: false` with `null` metrics and exclusion from the averages. This rule
came from a real failure: recording "dependency missing" as `0.0` would have made
Unlimited-OCR look like the worst backend instead of an unmeasured one. The same
rule applies to figures that need OCR (`tech2 F4` scores `None`, not `0`).

**D-11 — The citation verifier may remove claims but never add them, and an
emptied answer becomes a decline.** This is why the marker-parsing bug below was
loud instead of silent: a correct answer that lost all its citations turned into
an honest "INSUFFICIENT" rather than an uncited answer.

**D-12 — An honest decline is a successful run.** CLI exit `3`, one refusal
metric (`refusal_accuracy`) with the two failure modes (over-refusal, false
answer) reported separately, and three unanswerable questions in the gold set on
purpose. Rejected: scoring a decline as a zero — that makes a system that
guesses look better than one that admits ignorance.

**D-13 — Config A gets an explicit naive relevance floor.** The "naive dense RAG"
baseline must be naive, not subtly tuned: `SPECINTEL_SCORE_FLOORS` sets one floor
for every retriever, and config A then declines what it cannot ground instead of
answering from a weak top-1 hit. The floor is recorded in the ablation output so
the comparison cannot be mistaken for a tuned baseline.

**D-14 — A change to ingestion or retrieval invalidates every downstream
number.** After the layout fix (D-15) the index, the graph, the structure check
and the whole evaluation were rebuilt in the same commit before any number was
quoted. This is the rule that keeps `eval/results/` trustworthy; a statistic from
a previous corpus is treated as a bug, not as a baseline.

**D-15 — The table bug was in the generator, not the extractor.** Symptom:
`tables=0` in every document. Two hypotheses were wrong (the caption regex, the
cell extractor) before the real cause was found in
`synthetic/layout.py::_column_widths`: it added fit slack measured in *weight*
units to widths already in *points*, so the first column was ~470 pt wide, the
rest fell off the page, and the cells were never written into the PDF at all.
Lesson recorded here because it generalises: **when a detector finds nothing,
check that the thing was ever there.**

**D-16 — `docs` on a query is a promise, not a hint.** `specqa ask --docs` and
`/api/ask {"docs": …}` restrict the evidence list itself (after the graph step,
before reflection), so a filtered answer cannot cite an excluded document. The
filter logs `agent.docs_filter` with the number of items removed.

**D-17 — The demo server must survive its own concurrency.** `DemoState._lock` is
an `RLock`: building the agent builds the toolkit, and both take the lock, so a
plain `Lock` deadlocked the first `/api/ask`. `demo/app.py` logs every request
with its duration and counts internal errors separately from bad requests.

**D-18 — Two log levels of detail, one format.** One line per event, key=value,
parsed by `LOG_LINE_RE`; `SPECINTEL_LOG_JSON=1` switches to JSON without changing
event names, so an operator can grep the file and a script can parse it. The
test asserts the shape, so a new `print()` on the answer path fails the suite.

## 4. Reviewer checklist

Answer these before approving a change; each one names the file that settles it.

1. **Contract drift** — did any field, path or event name change? (diff
   `common/models.py` and `INTERFACES.md`; `tests/test_contracts.py` must pass)
2. **Provenance** — does every new `Element`/`Table`/`Figure`/`Chunk`/`Edge` carry
   `doc_id` and `page`, and does every `Citation` name revision, section, page and
   region? (`specqa ask … --json` and look at `citations`)
3. **Honesty** — can the change turn a decline into a guess? Does a new metric
   treat "could not run" as zero? (`eval/metrics.py`, `benchmarks/ocr/run_bench.py`)
4. **Staleness** — if ingestion or retrieval changed, were the index, the graph
   and the evaluation rebuilt before any number was quoted? (`data/index/build_report.json`
   timestamp vs. the change; D-14)
5. **Determinism** — does the change depend on a model server, a network call or a
   random seed that is not recorded? (`INTERFACES.md §8`, the `hardware` block of
   `eval/results/summary.json`)
6. **Licences** — is any licensed PDF, ZIP or bulk extracted text about to be
   committed? (`.gitignore` covers `data/`, `*.pdf`, `*.zip`; only
   `samples/**` and the gold set are allowed through)
7. **Reproducibility** — does `make_synthetic_corpus → ingest → build_index →
   build_graph → run_eval` still run from a clean clone with no arguments?
   (`README.md § Quickstart`)
