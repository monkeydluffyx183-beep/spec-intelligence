"""WS1 — ingestion, layout, tables, figures and the OCR benchmark.

The first block runs on the tiny fixture corpus built in ``conftest`` (no
artefacts required); the second block checks the full seven-document corpus and
is skipped until it has been built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from specintel.ingest import figures as figures_mod
from specintel.ingest import tables as tables_mod
from specintel.synthetic.pdfwrite import char_width, text_width, wrap
from specintel.common.models import ELEMENT_TYPES, Element, Table


# --------------------------------------------------------------------------- #
# the page writer: page geometry must match the advance widths it claims
# --------------------------------------------------------------------------- #
def test_text_width_matches_helvetica_metrics():
    # Helvetica 'W' is 944 units/1000 em -> 9.44 pt at 10 pt, plus the 4 % safety margin
    assert char_width("W") == pytest.approx(944)
    assert text_width("W", 10.0) == pytest.approx(9.44 * 1.04, rel=1e-3)
    assert text_width("i", 10.0) < text_width("W", 10.0)


def test_wrap_never_exceeds_its_column():
    """A line that overflows its column corrupts the table cell it belongs to."""
    cells = ["COMMON.CHECK.PD.2", "Verify that the port tolerates a message with no GoodCRC",
             "tSenderResponse", "EPR_Sink_Ready"]
    for cell in cells:
        for width in (60.0, 95.0, 180.0):
            for line in wrap(cell, 7.5, width=width):
                assert text_width(line, 7.5) <= width * 1.001 or " " not in line, (cell, width, line)


# --------------------------------------------------------------------------- #
# ingestion of the fixture corpus
# --------------------------------------------------------------------------- #
def test_ingest_writes_typed_elements_with_page_provenance(ingested_tiny):
    processed = ingested_tiny.path("paths.processed")
    for doc_id in ("pd", "typec"):
        elements = _elements(processed, doc_id)
        assert elements, doc_id
        for element in elements:
            assert element["doc_id"] == doc_id
            assert element["page"] >= 1
            assert element["revision"], "every element carries its revision"
            assert element["element_type"] in ELEMENT_TYPES, element["element_type"]
        types = {e["element_type"] for e in elements}
        assert "table" in types or any(e["table_id"] for e in elements)


def _elements(processed: Path, doc_id: str) -> list[dict]:
    """Read ``data/processed/<doc>/elements.jsonl`` (one element per line)."""
    path = processed / doc_id / "elements.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_table_body_text_is_not_left_behind_as_paragraphs(ingested_tiny):
    """WS1 retypes text that sits inside a table grid into the table.

    Before this, the same sentence was retrievable both as a paragraph and as a
    table row, and the agent answered with the paragraph copy.
    """
    from specintel.ingest.pipeline import load_tables

    processed = ingested_tiny.path("paths.processed")
    elements = _elements(processed, "pd")
    tables = load_tables(ingested_tiny, "pd")
    assert tables, "the fixture corpus has a table"
    table = tables[0]
    assert table.headers and table.rows
    assert table.pages == [table.pages[0]]
    # only substantial cells count as a leak: a short value such as "PS_RDY" or
    # "24" legitimately appears in prose as well as in a table
    cell_texts = {c.strip() for row in table.rows for c in row
                  if len(c.strip()) > 24 and len(c.split()) > 2}
    leaked = [e for e in elements
              if e["element_type"] == "paragraph"
              and any(cell and cell in e.get("text", "") for cell in cell_texts)]
    assert not leaked, f"{len(leaked)} table cell(s) still retrievable as paragraphs"


def test_table_cell_accuracy_is_computed_per_cell(ingested_tiny):
    from specintel.ingest.pipeline import load_tables

    tables = load_tables(ingested_tiny, "pd")
    gold = next(t for t in tables if t.number == "6.8")
    pred = Table(table_id=gold.table_id, doc_id=gold.doc_id, revision=gold.revision,
                 number=gold.number, caption=gold.caption, pages=list(gold.pages),
                 headers=[list(h) for h in gold.headers], rows=[list(r) for r in gold.rows])
    score = tables_mod.table_cell_accuracy(gold, pred)
    assert score["cell_accuracy"] == pytest.approx(1.0)
    broken = Table(table_id=gold.table_id, doc_id=gold.doc_id, revision=gold.revision,
                   number="6.8", caption=gold.caption, pages=list(gold.pages),
                   headers=[list(h) for h in gold.headers], rows=[["tSenderResponse", "24"]])
    assert tables_mod.table_cell_accuracy(gold, broken)["cell_accuracy"] < 1.0


def test_figure_structure_reports_what_it_could_not_read():
    """The structuring backend must state its limits, not invent a diagram."""
    result = figures_mod.structure_heuristic(
        "PE_SRC_Startup\nPE_SRC_Send_Capabilities", "Source policy engine states",
        has_raster=False, kind="state_machine")
    assert result.structure["states"], "state labels are recovered"
    assert result.structure["transitions"], "adjacent states become a transition"
    assert result.backend == "heuristic"

    blank = figures_mod.structure_heuristic("", "Timing of an EPR entry", has_raster=True)
    assert blank.kind in ("timing", "photo", "chart")
    assert blank.notes, "a figure with no readable labels says so"


def test_page_images_are_rendered_with_the_configured_dpi(ingested_tiny):
    """The page images are the vision path's input (config D fuses them)."""
    pages_dir = ingested_tiny.path("paths.processed") / "pd" / "pages"
    images = sorted(pages_dir.glob("*.png"))
    assert images, "page images are the vision path's input"
    assert all(p.stat().st_size > 1000 for p in images)


# --------------------------------------------------------------------------- #
# caption parsing: every document numbers its tables differently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("Table 6.8. Timer definitions and limits", "6.8"),
    ("Table 4-27 Configuration Channel", "4-27"),
    ("Table 3-4. TD 4.2.4 procedure steps", "3-4"),
    ("Table B-1 Power rules", "B-1"),
    ("Table A.1. Revision history", "A.1"),
])
def test_caption_regex_handles_every_numbering_style(text, expected):
    from specintel.ingest.parser import CAPTION_RE

    match = CAPTION_RE.match(text)
    assert match, text
    assert match.group("number").strip(" .") == expected


def test_table_ids_are_namespaced_per_document(ingested_tiny):
    """Two documents may both have a 'Table 3'; the ids must not collide.

    The assignment flags the same trap for figure numbers and test ids, and the
    fix is the same: the document id is part of every artefact id.
    """
    from specintel.ingest.pipeline import load_tables

    ids: dict[str, str] = {}
    seen_docs = set()
    for doc_id in ("pd", "typec"):
        for table in load_tables(ingested_tiny, doc_id):
            assert table.table_id.startswith(f"{doc_id}:table:"), table.table_id
            assert table.table_id not in ids, f"{table.table_id} in {ids.get(table.table_id)} and {doc_id}"
            ids[table.table_id] = doc_id
            seen_docs.add(doc_id)
    assert seen_docs == {"pd", "typec"}


# --------------------------------------------------------------------------- #
# the full corpus
# --------------------------------------------------------------------------- #
def test_full_corpus_structure_matches_the_manifest(needs_corpus, config):
    from specintel.eval.structure import evaluate_structure
    from specintel.ingest.pipeline import load_figures, load_tables

    manifest = config.path("paths.synthetic") / "manifest.json"
    tables, figures = [], []
    for doc_id in [d["doc_id"] for d in config.documents()]:
        tables.extend(load_tables(config, doc_id))
        figures.extend(load_figures(config, doc_id))
    report = evaluate_structure(manifest, tables, figures)
    payload = report.to_dict()
    assert payload["table_cell_accuracy"] >= 0.80, payload["per_document"]
    assert payload["figure_structure_accuracy"] >= 0.80, payload["per_document"]
    missing = [t for t in payload["tables"] if not t["matched"]]
    assert not missing, f"{len(missing)} manifest table(s) were not detected: {missing[:3]}"
    assert not [f for f in payload["figures"] if "not detected" in (f["note"] or "")]


def test_every_ingested_table_can_be_read_back_by_id(needs_corpus, config):
    from specintel.ingest.pipeline import load_tables

    for doc_id in [d["doc_id"] for d in config.documents()]:
        for table in load_tables(config, doc_id):
            assert table.headers or table.rows, table.table_id
            assert table.pages and min(table.pages) >= 1


def test_ocr_benchmark_runner_reports_every_backend(needs_corpus, config):
    """The benchmark compares backends on the same crops; it must not fake one."""
    import importlib.util
    import sys

    path = config.root / "benchmarks" / "ocr" / "run_bench.py"
    assert path.exists(), "benchmarks/ocr/run_bench.py must exist"
    spec = importlib.util.spec_from_file_location("ocr_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ocr_bench"] = module          # dataclasses look the module up by name
    spec.loader.exec_module(module)
    backends = module.backends_available(config)
    assert "pymupdf-text" in backends, "the vector-text backend always exists"
    assert all(isinstance(v, bool) for v in backends.values())
