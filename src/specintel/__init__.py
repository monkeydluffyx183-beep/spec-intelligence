"""Spec Intelligence — multimodal, agentic RAG over USB-PD, Type-C, USB4 and Tech 2 specs.

Workstream layout (see INTERFACES.md):

* :mod:`specintel.ingest`  — WS1: layout parsing, OCR, tables, figures
* :mod:`specintel.index`   — WS2: chunking, dense/BM25/page indexes, fusion
* :mod:`specintel.kg`      — WS3: ontology, extraction, entity resolution, store
* :mod:`specintel.agent`   — WS4: tools, planner, reflection, citation verifier, CLI
* :mod:`specintel.eval`    — WS5: gold set, metrics, ablations, error analysis
* :mod:`specintel.common`  — shared contracts, config, logging, resources
* :mod:`specintel.synthetic` — license-free stand-in corpus generator
"""

__version__ = "0.1.0"
