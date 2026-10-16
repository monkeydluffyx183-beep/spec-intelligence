# Spec Intelligence

### Multimodal Ingestion, Knowledge Graphs & Agentic RAG for Technical Specifications

Spec Intelligence is an AI-powered system designed to process technical specifications and answer engineering questions using **multimodal document understanding, hybrid retrieval, knowledge graphs, and agentic Retrieval-Augmented Generation (RAG)**.

The project focuses on USB Power Delivery (USB-PD), USB Type-C, USB4, and Tech 2 specifications. It aims to transform complex technical documents into searchable, interconnected knowledge and provide answers grounded in supporting evidence, with page-level citations and verifiable references.

```console
$ specqa ask "What are the limits of tSenderResponse in Table 6.8?"
ANSWERED
Timer: tSenderResponse; Min: 24; Max: 30; Unit: ms; Definition: Maximum time a
Sender waits before it starts to send a response to a received message
[pd, R3.2 V1.2 (19 Aug 2026), §6.6, p.3, Table 6.8] …

$ specqa ask "What is the maximum number of DisplayPort tunnels on a Meteor Lake host?"
DECLINED (exit 3) — the ingested corpus does not answer this question.
```

## 🎯 Objectives

* **Multimodal document ingestion:** Extract and process information from technical documents, including text and visual content.
* **Document intelligence:** Apply parsing and OCR techniques to handle complex document layouts and scanned pages.
* **Hybrid retrieval:** Combine text-based and vision-based retrieval to improve access to technical information.
* **Knowledge graph generation:** Represent entities and relationships extracted from specifications as interconnected knowledge.
* **Agentic RAG:** Use AI agents to retrieve relevant evidence and formulate context-aware answers.
* **Evidence-grounded responses:** Provide page-level citations and references that help users verify answers against the source documents.

## 🧠 Key Features

### 1. Multimodal Ingestion

Process technical specification documents and extract useful information from textual and visual content.

### 2. OCR and Document Parsing

Support the extraction of information from document pages where content may not be available as directly readable text.

### 3. Hybrid Search and Retrieval

Combine text retrieval and visual retrieval to locate relevant information across complex technical documents.

### 4. Knowledge Graph

Build a structured representation of entities, concepts, and relationships found in the specifications.

### 5. Agentic Retrieval-Augmented Generation

Use retrieval and agent-based workflows to answer engineering questions using relevant source evidence.

### 6. Citation and Evidence Verification

Ground generated answers in retrieved content and provide page-level citations so users can trace responses back to the source material.

## ✅ Results (configuration F — the full system)

Measured with the committed gold set: **60 questions** (41 development, 19 held out, authored 12 per workstream owner, including three the corpus cannot answer), on 2 vCPU / 3.8 GB RAM / **no GPU**.

| target | target value | measured | status |
|---|---|---|---|
| recall@10 | ≥ 0.80 | **1.000** | met |
| answer correctness | ≥ 0.75 | **0.882** | met |
| faithfulness | ≥ 0.85 | **0.920** | met |
| citation accuracy | ≥ 0.80 | **0.930** | met |
| refusal accuracy | ≥ 0.80 | **1.000** | met |
| KG triple precision | ≥ 0.80 | **0.950** | met |
| table cell accuracy | ≥ 0.80 | **0.908** | met |
| structured-figure accuracy | ≥ 0.80 | **1.000** | met |

p50 latency **0.026 s** / p95 **0.048 s**. Full per-category, per-content-type and per-configuration tables: `eval/results/report.md`; ablations: `docs/ABLATIONS.md`; the recorded hardware and model backend travel with every report.

## 🚀 Quickstart

```bash
make setup      # pinned versions (the pipeline itself is stdlib-only)
make corpus     # license-free stand-in corpus + the 60-question gold set
make build      # ingest → index → graph
make eval       # configurations A–F over the gold set → eval/results/report.md
make ask ASK="What are the limits of tSenderResponse in Table 6.8?"
make serve      # demo UI and HTTP API on 0.0.0.0:8000
make test       # 120 tests; builds its own fixture PDFs, needs no corpus
```

`specqa` is also the console entry point of the package, with the subcommands
`ask | tools | ingest | build-index | build-graph | eval | graph | chunks | doctor | serve`.

### Using the licensed specifications

The USB-IF and WPC documents are not in this repository and never will be
(`data/` is git-ignored). Put them under `data/raw/<doc_id>/` and re-run the same
commands; the document registry (ids, titles, revisions, page counts) is
`config.yaml § corpus.documents`:

```
data/raw/pd/USB_PD_R3.2_V1.2_2026_08_19.pdf
data/raw/pd_cts/USB PD3 CTS Q3 2026 OR.pdf
data/raw/typec/USB Type-C Spec R2.5 - March 2026.pdf
data/raw/typec_fts/USB Type-C Functional Test Specification.pdf
data/raw/scc_cts/USB Type-C Shared Capacity Charger Interoperability CTS v1.0.pdf
data/raw/usb4/USB4 Specification 2.0 November 2025 - CLEAN.pdf
data/raw/tech2/Tech 2 Book *.pdf
```

`data/raw/` is the primary input path; `data/synthetic/` is the fallback used
while a document is missing, so a clean clone can still run the whole pipeline.

## 🏗️ System Architecture

The implemented workflow:

```text
Technical Specification Documents  (data/raw, licensed, git-ignored)
              |
              v
      Multimodal Ingestion          typed elements · stitched tables · figures · page images
              |
              v
   Parsing, OCR & Extraction         boilerplate stripped · page provenance kept
              |
              v
       Knowledge Processing
          /          \
         v            v
  Hybrid Retrieval  Knowledge Graph  BM25 ⊕ dense ⊕ page vectors    27 node / 19 edge types
         \            /              reciprocal rank fusion          provenance on every edge
          v          v
        Agentic RAG Pipeline         plan → tools → reflect → verify citations
                |
                v
       Evidence-Grounded Answers     or an honest decline, with the reason
                |
                v
       Page-Level Citations          [doc, revision, §section, p.page, region]
```

`DIAGRAMS.md` contains the detailed Mermaid diagrams for every layer (ingestion, retrieval, graph, agent, evaluation, deployment) plus the worked-example sequence diagram.

## 🛠️ Technology Areas

* Artificial Intelligence and Large Language Models
* Retrieval-Augmented Generation (RAG)
* Multimodal AI and document understanding
* Optical Character Recognition (OCR)
* Hybrid information retrieval
* Knowledge graphs
* AI agents and tool-assisted workflows
* Evidence-grounded question answering

Local-first by design: embeddings, extraction, figure understanding and answer
composition each sit behind an adapter with a deterministic CPU fallback, so the
system runs with no GPU, no model server and no network, and the same code paths
accept a local model through Ollama / vLLM / OpenAI-compatible endpoints
(`INTERFACES.md § 8`). No hosted API is called anywhere on the
`ingest → build-graph → eval` path.

## 📚 Target Specifications

* USB Power Delivery (USB-PD)
* USB Type-C
* USB4
* Tech 2 specifications

The traps those documents contain are handled explicitly and tested: per-book
numbering (`Table 6.8.` / `Table 4-27` / `Table 1`), duplicate or superseded
revisions, test-specification references older than the base specification (flagged
`stale`, never followed silently), stitched multi-page tables, vector figures
(USB-PD) versus rasterised figures (Tech 2), repeated headers and footers, four
different test-case identifier dialects, and `__MACOSX/` stubs in the vendor ZIPs.

## 📖 Documentation

| document | contents |
|---|---|
| `INTERFACES.md` | the contracts: data ladder, field names, artefact paths, the single log format, tool signatures, CLI/HTTP surface, environment switches, model adapters |
| `DIAGRAMS.md` | eight Mermaid diagrams, layer by layer, plus the worked example as a sequence diagram |
| `docs/WORKED_EXAMPLE.md` | one question followed from the PDF page to the cited answer, with the trace and the log lines |
| `docs/ABLATIONS.md` | the architecture sweep A→F and seven single-component ablations, with what each does and does not show |
| `docs/OCR_BENCHMARK.md` | the OCR/document-AI benchmark: method, results, and how to run Unlimited-OCR on a GPU box |
| `docs/KG_SCHEMA.md` | the knowledge-graph ontology: every node and edge type |
| `TEAM.md` | workstream ownership, the decisions log (D-01…D-19), and the reviewer checklist |
| `samples/` | small committable excerpts of each derived artefact |

## 🔒 Data and Licensing

This repository is **private** because the source specifications are subject to licensing and distribution restrictions.

Licensed specification PDFs, bulk extracted specification text, credentials, API keys, and other restricted material are **not** committed: `data/` is git-ignored, as are `*.pdf` and `*.zip`. Only small derived samples (`samples/`), the gold question set, and benchmark summaries are tracked. Secrets come from environment variables (`SPECINTEL_*`); `config.yaml` holds no credentials.

## 🚀 Project Status

**Status:** implemented end to end — ingestion, retrieval, knowledge graph, agent, evaluation and documentation. All eight target metrics are met on the current corpus (`eval/results/report.md`).

The headline numbers are measured on the license-free stand-in corpus that
reproduces the layout traps of the real books; they are regenerated — never
adjusted — once `data/raw/` is populated. `README.md § Known limits` states what
is not measured yet.

## 🧪 Reproducibility

* **Deterministic by default**: hashing embeddings, rule-based extraction, an extractive composer, a heuristic figure reader — no randomness, no network, no model server. `make all` reproduces the results table from a clean clone.
* **Model and hardware versions are recorded**, not assumed: every run stamps the backend, model name, hardware profile and configuration into `eval/results/summary.json`, `data/index/build_report.json` and each agent trace. Python extras are pinned in `requirements.lock.txt`.
* **The corpus is rebuildable**: `make corpus` regenerates the stand-in PDFs, the structural manifest used as the ground truth for the table/figure scores, and the gold set from one seed.
* **A change to ingestion or retrieval invalidates every downstream number**: the index, graph and evaluation are rebuilt before any figure is quoted (`TEAM.md` D-14).

## ⚠️ Known Limits

* The reported numbers are on the **stand-in corpus**: it reproduces the layout traps but not the size or vocabulary of the real books.
* **Rasterised figures** (e.g. `tech2 F4`) have no text layer, so their structure scores `None`, never `0`; an OCR/VLM pass is required (`docs/OCR_BENCHMARK.md`). On this machine only `pymupdf-text` and the `noop` control can be scored — `rapidocr` needs an extra dependency and `unlimited-ocr` needs an NVIDIA GPU.
* **Unanswerable questions** are scored by refusal accuracy only; reporting faithfulness or citation accuracy for them would be a refusal artefact.
* The **vision-only configuration is flattered** by the synthetic pages' perfect text layer, yet remains the worst at citing (0.200) — `docs/ABLATIONS.md § 1`.
* The LLM graph-extraction pass and the Ollama/vLLM composers are implemented and switchable but are not the recorded configuration; their numbers await a GPU.

## 🗺️ Roadmap

* [x] Set up the project structure and development environment
* [x] Implement document ingestion and parsing
* [x] Add OCR and visual-content processing
* [x] Develop text and vision retrieval pipelines
* [x] Build the knowledge graph generation workflow
* [x] Integrate the agentic RAG pipeline
* [x] Add source-grounded answers and page-level citations
* [x] Evaluate retrieval quality and answer correctness
* [x] Document setup, usage, and system architecture
* [ ] Re-run ingestion, graph and evaluation against the licensed PDFs
* [ ] Score the GPU backends (Unlimited-OCR, a local VLM) on the same crops

## 🤝 Contribution

Development follows the contracts in `INTERFACES.md`, which are enforced by
`tests/test_contracts.py`. Every change is opened as a pull request and reviewed
against the checklist in `TEAM.md § 4` before merging. Contributions must respect
applicable specification licences and must not expose restricted source material.

## ⚖️ Disclaimer

This project is intended for technical research and engineering information retrieval. Answers generated by AI should be verified against the cited source specifications before being used for design decisions or compliance assessments.

---

**Spec Intelligence** — Turning complex technical specifications into searchable, connected, evidence-grounded engineering knowledge.
