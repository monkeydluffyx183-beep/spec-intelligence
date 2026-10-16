# spec-intelligence — the whole pipeline, one command per stage.
#
# Everything runs with the standard library alone; the extras in
# pyproject.toml (`pip install -e '.[dev]'`) add the PDF parser, numpy and the
# test runner.  `PYTHONPATH=src` keeps the commands working without an install.

PYTHON ?= python3
export PYTHONPATH := src

.PHONY: help setup test corpus ingest index graph build eval ablate ask serve ocr all clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:  ## install the pinned versions the reported numbers used
	$(PYTHON) -m pip install --break-system-packages --user -r requirements.lock.txt

test:  ## run the whole test suite (no corpus needed — it builds fixture PDFs)
	$(PYTHON) -m pytest tests -q

corpus:  ## generate the license-free stand-in corpus + the 60-question gold set
	$(PYTHON) scripts/make_synthetic_corpus.py

ingest:  ## parse the PDFs into typed elements, stitched tables, figures, pages
	$(PYTHON) scripts/ingest_corpus.py

index:  ## build the BM25 + dense + page-image indexes
	$(PYTHON) scripts/build_index.py

graph:  ## build the knowledge graph (rule extraction; add -- --llm for the model pass)
	$(PYTHON) scripts/build_graph.py --no-llm

build: ingest index graph  ## everything downstream of the corpus

eval:  ## run the A–F evaluation over the gold set
	$(PYTHON) -m specintel.eval.run_eval --configs A,B,C,D,E,F --split all

ablate:  ## run the seven single-component ablations
	$(PYTHON) scripts/ablate.py --split all

ask:  ## ASK="question" make ask
	$(PYTHON) -m specintel.agent.cli ask "$(ASK)" --trace

serve:  ## run the demo UI and HTTP API on 0.0.0.0:8000
	$(PYTHON) -m specintel.agent.cli serve --host 0.0.0.0 --port 8000

ocr:  ## run the OCR/document-AI benchmark
	$(PYTHON) benchmarks/ocr/run_bench.py

all: corpus build eval ablate ocr  ## the full reproduction from a clean clone

clean:  ## remove derived artefacts (never the licensed PDFs)
	rm -rf data/processed data/index data/graph data/runs data/logs eval/results
	find . -name '*.pyc' -delete
