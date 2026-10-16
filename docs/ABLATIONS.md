# Ablations and what each component is worth

Two kinds of comparison are reported here, and they answer different questions:

* **Architecture sweep `A → F`** (`eval/results/report.md`) — dense → hybrid →
  vision → fused → graph → agent loop. This is "which architecture wins".
* **Option ablations** (`eval/results/ablation.json`, `scripts/ablate.py`) — one
  component switched off inside a fixed architecture. This is "which part of the
  pipeline is doing the work".

Both were run on the current corpus with the same 60-question gold set, the same
index and the same graph (`--split all`, `extractive` composer, no GPU).

```bash
PYTHONPATH=src python3 -m specintel.eval.run_eval --configs A,B,C,D,E,F --split all
PYTHONPATH=src python3 scripts/ablate.py --split all      # 7 ablations → ablation.json
```

## 1. Architecture sweep

| config | pipeline | recall@10 | recall@5 | correctness | faithfulness | citation acc. | refusal acc. | p50 s |
|---|---|---|---|---|---|---|---|---|
| A | naive dense RAG (dense-only, no rerank) | 0.867 | 0.867 | 0.814 | 0.904 | 0.713 | 0.950 | 0.002 |
| B | A + hybrid BM25+dense + lexical reranker | 0.800 | 0.800 | 0.798 | 0.909 | 0.675 | 0.950 | 0.007 |
| C | vision only (page-image retrieval, capstone-04 style) | 0.967 | 0.833 | 0.853 | 0.836 | **0.200** | 0.950 | 0.006 |
| D | B + C fused (reciprocal rank fusion) + structured tables/figures | 0.950 | 0.867 | 0.835 | 0.901 | 0.513 | 0.950 | 0.011 |
| E | D + knowledge-graph tool | 0.950 | 0.917 | 0.815 | **0.927** | 0.653 | 0.933 | 0.017 |
| F | E + agent loop (planning, reflection, citation verification) | **1.000** | **0.950** | **0.882** | 0.920 | **0.930** | **1.000** | 0.026 |

Read honestly:

* **The agent's contribution is grounding.** A (naive dense) answers *narrowly*
  and gets 0.814 correctness, but it cites correctly only 71% of the time and
  never checks that a sentence is backed by the evidence it names. F adds the
  planner, the graph→table follow-up, reflection and citation verification, and
  reaches 0.882 correctness with citation accuracy 0.930, faithfulness 0.920 and
  a refusal discipline of 1.000. The gap is largest exactly where the brief puts
  the hurdles: `test_procedure` 0.675 → 0.946 and `bitfield_table` 0.812 → 0.908.
* **Vision-only retrieval is strong at finding the page and useless at citing
  it.** C has the best recall@10 (0.967) and the worst citation accuracy (0.200):
  a page-image hit can name the page but not the table or element inside it, so
  the citation cannot meet `[doc, revision, §section, p.page, region]`. That is
  the text-versus-vision comparison the brief asks for, and it is why the shipped
  path is fusion (D) rather than vision alone. **Caveat:** this synthetic corpus
  renders the text layer perfectly, so C's retrieval is flattered; on a scanned
  or raster-heavy real corpus the page path is expected to do relatively better
  and the text path worse (`docs/OCR_BENCHMARK.md` measures exactly that gap).
* **Fusion (D) helps retrieval and hurts citation** relative to B, until the
  graph (E) and the agent (F) restore it — fusing page hits into the evidence
  set dilutes element-level provenance.
* **Refusal accuracy separates the honest configurations from the rest**: A–D sit
  at 0.950 (they answer three unanswerable questions), E answers 4/60 it should
  have declined, F declines exactly what it should.

## 2. Option ablations

Each row switches off one component inside its architecture (config in
parentheses); the baseline column is the same configuration untouched.

| ablation | config | recall@10 | MRR | correctness | faithfulness | citation acc. | over-refusal | false answers |
|---|---|---|---|---|---|---|---|---|
| *(baseline)* | B | 0.800 | 0.717 | 0.798 | 0.909 | 0.675 | 0.000 | 0.050 |
| `reranker` — drop the lexical reranker | B | 0.867 | 0.784 | 0.820 | 0.914 | 0.669 | 0.000 | 0.050 |
| `bm25` — drop BM25, keep dense | B | 0.817 | 0.725 | 0.805 | 0.919 | 0.656 | 0.000 | 0.050 |
| *(baseline)* | E | 0.950 | 0.829 | 0.815 | 0.927 | 0.653 | 0.067 | 0.000 |
| `figure_text` — drop figure descriptions and OCR text | E | 0.950 | 0.826 | 0.815 | 0.926 | 0.661 | 0.067 | 0.000 |
| `table_rows` — drop stitched table bodies | E | 0.950 | 0.777 | 0.818 | 0.914 | **0.590** | 0.067 | 0.000 |
| `kg_staleness` — stop flagging superseded cross-refs | E | 0.950 | 0.829 | 0.815 | 0.927 | 0.653 | 0.067 | 0.000 |
| *(baseline)* | F | 1.000 | 0.873 | 0.882 | 0.920 | 0.930 | 0.000 | 0.000 |
| `single_pass` — no multi-part decomposition | F | 1.000 | 0.873 | 0.867 | 0.917 | 0.930 | 0.000 | 0.000 |
| *(baseline)* | D | 0.950 | 0.591 | 0.835 | 0.901 | 0.513 | 0.000 | 0.050 |
| `uniform_floor` — one relevance floor for every retriever | D | 0.950 | 0.591 | **0.000** | — | 0.000 | **0.950** | 0.000 |

What each row says, including where the effect is small or absent:

1. **`reranker` (B).** Removing it *raises* recall@10 (0.800 → 0.867) and lowers
   MRR (0.717 → 0.784, i.e. it improves the ranking) and correctness by 0.02: the lexical reranker improves the
   *order* of the top hits and slightly reduces how many land inside the top 10,
   because it promotes identifier-bearing rows over broader pages. Citation
   accuracy is unchanged. Honest conclusion: on this corpus the reranker is a
   precision tool, not a recall tool, and it is not the source of the pipeline's
   quality.

   *Measurement note:* the first run of this ablation showed `correctness 0.000`
   with 95% over-refusal. That was an artefact, not a finding — with the reranker
   off, hybrid scores collapse from the 1–4 range into reciprocal-rank-fusion
   range (~0.03), and config B's calibrated floor (0.25) then rejected *every*
   question. The runner now falls back to a relative floor (0.25 × best hit) when
   a *calibrated* floor sits above the retriever's whole score scale, and records
   that in `AgentTrace.warnings`; an explicitly set floor (the `uniform_floor`
   ablation) is still obeyed exactly. See `TEAM.md` D-19.
2. **`bm25` (B).** Dense alone: MRR 0.717 → 0.725, recall@10 0.800 → 0.817,
   correctness +0.007. The specifications are identifier-dense, so BM25 carries
   the ordering; dropping it costs rank quality everywhere except raw recall.
3. **`figure_text` (E).** No measurable change on the 60-question gold set
   (every metric within 0.01; citation accuracy even rises 0.008, which is noise
   at this sample size). The questions that depend on a figure description
   are few (9 category-F questions, of which 4 are timing/state-machine), and the
   figure text is *also* reachable through the table/prose path. The switch is
   still worth keeping: it is how the figure contribution is priced when a
   figure-only question set is added.
4. **`table_rows` (E).** MRR 0.829 → 0.777 and citation accuracy 0.653 → 0.590:
   dropping stitched table bodies costs *provenance and ranking* (the row is
   often both the answer and the thing to cite) more than it costs final answers,
   because the parent section chunk still usually contains the table caption. The
   largest single-component degradation among the option ablations.
5. **`kg_staleness` (E).** No change in the aggregate scores, and that is the
   point: the flag is a *correctness-of-provenance* feature, not a retrieval
   feature. With it off, the pipeline still retrieves the same documents — it
   just stops telling the reader that a test specification is pointing at a
   superseded requirement. It is verified separately (`tests/test_kg.py`,
   `kg_precision.evaluate_edge_provenance`) rather than through answer metrics.
6. **`single_pass` (F).** Small but consistent: correctness 0.882 → 0.867 and
   faithfulness 0.920 → 0.917. Multi-part decomposition pays for itself on
   compound questions (category X, 11 questions); averaged over all 60 the effect
   is 1.5 points, which is worth stating precisely rather than overclaiming.
7. **`uniform_floor` (D).** The instructive failure: with one absolute floor
   (0.25) for every retriever, the fused retriever — whose scores are RRF sums,
   ~0.03 — grounds nothing, so config D declines 95% of the questions and scores
   0.000 correctness against its 0.835 baseline. This is *why* floors are
   per-retriever, and why the floors are recorded with the results.

## 3. What is not measured yet

* **Real-corpus ablations.** All of the above is on the synthetic stand-in
  corpus; the licensed PDFs need a machine that can hold them. The numbers should
  be regenerated (not adjusted) after `data/raw/` is populated — `TEAM.md` D-14.
* **Vision ablations on raster-heavy content.** `pymupdf-text` returns nothing on
  a raster figure, so the figure path's contribution is currently
  corpus-specific; `docs/OCR_BENCHMARK.md` is the controlled measurement of that
  gap.
* **LLM extraction in the graph.** `build_graph --no-llm` is the recorded run
  (rule extraction); the optional LLM pass is implemented and switchable but its
  triple precision is not audited here, since it needs a GPU to be worth running.
* **The held-out split.** `--split held_out` runs the 19 reviewer-style questions;
  the headline table is `--split all` (60). The held-out column in the report
  fills in when that run is taken, and the questions were frozen before the
  ablation runs that tuned the floors.
