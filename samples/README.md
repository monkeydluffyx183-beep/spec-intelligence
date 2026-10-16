# samples/

Small, committable excerpts of the derived artefacts, so a reviewer can see the
shapes without the licensed corpus (`data/` is git-ignored).

| file | what it is |
|---|---|
| `elements_pd_sample.jsonl` | six typed elements from the Power Delivery book with page provenance and bounding boxes |
| `table_stitched_sample.json` | Table 6.8 as one stitched object with `pages=[3,4]`, headers and cell grid |
| `graph_triples_sample.jsonl` | eight graph edges with document, section, page, extraction method and confidence |
| `gold_questions_sample.jsonl` | two gold questions with their gold evidence (doc, page, element id, quote) |
| `agent_trace_sample.json` | the full agent trace of the worked example in `docs/WORKED_EXAMPLE.md` |

Regenerate any of them with the commands in `README.md § Quickstart`; the
complete artefacts stay in `data/` and `eval/results/`.
