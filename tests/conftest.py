"""Shared fixtures.

Two kinds of test live in this suite:

* **self-contained tests** build a tiny two-document corpus in ``tmp_path`` and run
  the real ingestion → index → graph → agent path over it, so ``pytest`` passes on
  a clean clone with no artefacts present;
* **corpus tests** exercise the full seven-document corpus and the evaluation
  numbers, and are skipped with an explanatory reason when ``data/`` has not been
  built (``make corpus && make index && make graph``).

Nothing here touches the network: the embedding backend is the deterministic
hashing embedder, the LLM is the extractive composer, and the PDFs are written by
``specintel.synthetic.pdfwrite``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:                                     # allow `pytest` without an install
    sys.path.insert(0, str(SRC))

from specintel.common.config import Config, load_config                     # noqa: E402
from specintel.synthetic import layout, pdfwrite                            # noqa: E402

MANIFEST_NAME = "manifest.json"


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def config() -> Config:
    """The repository configuration, with every path pointing inside the repo."""
    return load_config(REPO / "config.yaml", use_cache=False)


@pytest.fixture(scope="session")
def built_corpus() -> bool:
    """True when the seven-document corpus has been ingested and indexed."""
    return ((REPO / "data" / "index" / "chunks.jsonl").exists()
            and (REPO / "data" / "graph" / "graph.sqlite").exists())


@pytest.fixture(scope="session")
def needs_corpus(built_corpus: bool) -> None:
    if not built_corpus:
        pytest.skip("corpus not built: run `make corpus && make index && make graph`")


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a two-document, three-page corpus into ``tmp_path``.

    Built through the same ``synthetic`` modules the full corpus uses — the page
    writer, the layout engine and the document specs — so the fixtures stay
    honest: if the writer or the layout rules break, these tests break.
    """
    tmp_path = tmp_path_factory.mktemp("tiny")
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    licence = "Licensed material — synthetic stand-in, not redistributable."
    docs = [
        ("pd", "USB Power Delivery Specification", "R3.2 V1.2", "19 August 2026",
         [layout.Heading(number="6", title="Protocol Layer Messages", level=1, anchor="sec:6"),
          layout.Paragraph(text="All USB PD communication is carried in messages. "
                                "PS_RDY indicates that the power supply is ready.",
                           anchor="pd:6:intro"),
          layout.Paragraph(text="The Source starts tSenderResponse when it receives a Request. "
                                "A Sender that does not answer inside the limits of Table 6.8 "
                                "causes a hard reset.", anchor="pd:6:timers"),
          layout.TableBlock(number="6.8", caption="Timer definitions and limits", anchor="table:6.8",
                            headers=[["Timer", "Min", "Max", "Unit"]],
                            rows=[["tSenderResponse", "24", "30", "ms"],
                                  ["tPSTransition", "450", "550", "ms"],
                                  ["tHardResetComplete", "25", "—", "ms"]])]),
        ("typec", "USB Type-C Cable and Connector Specification", "R2.4", "1 August 2026",
         [layout.Heading(number="4.5", title="Terminations", level=1, anchor="sec:4.5"),
          layout.Paragraph(text="A Source presents Rp and a Sink presents Rd on the Configuration "
                                "Channel; the termination determines attach state, orientation and "
                                "the initial power role.", anchor="typec:4.5:term"),
          layout.TableBlock(number="4-9", caption="Termination requirements", anchor="table:4-9",
                            headers=[["Port", "Termination", "Current"]],
                            rows=[["Source", "Rp", "1.5 A"], ["Sink", "Rd", "—"]])]),
    ]
    manifest_elements: list[dict] = []
    manifest_documents: list[dict] = []
    paths: dict[str, Path] = {}
    for doc_id, title, revision, date, blocks in docs:
        writer = pdfwrite.PdfWriter(title=f"{title} ({revision}) — synthetic test stand-in")
        layout.title_page(writer, title, f"Revision {revision}", doc_id, date, licence,
                          [(b.number, b.title) for b in blocks if isinstance(b, layout.Heading)])
        flow = layout.Flow(writer, doc_id, revision, title, start_page_no=len(writer.pages))
        flow.new_page()
        for block in blocks:
            flow.add(block)
        (root / doc_id).mkdir(parents=True, exist_ok=True)
        path = root / doc_id / f"{doc_id}.pdf"
        writer.write(path)
        paths[doc_id] = path
        manifest_elements.extend(flow.manifest)
        manifest_documents.append({"doc_id": doc_id, "title": title, "revision": revision,
                                   "date": date, "files": [str(path)], "pages": flow.page_no,
                                   "elements": len(flow.manifest)})
    (root / MANIFEST_NAME).write_text(
        json.dumps({"documents": manifest_documents, "elements": manifest_elements},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def tiny_config(tiny_corpus: Path, tmp_path_factory: pytest.TempPathFactory, config: Config) -> Config:
    """A configuration whose data directories live under a temporary root."""
    import copy

    tmp_path = tiny_corpus.parent
    data = copy.deepcopy(config.data)
    data.setdefault("paths", {}).update({
        "raw": str(tmp_path / "raw"),
        "processed": str(tmp_path / "processed"),
        "index": str(tmp_path / "index"),
        "graph": str(tmp_path / "graph"),
        "pages": str(tmp_path / "pages"),
        "logs": str(tmp_path / "logs"),
        "synthetic": str(tiny_corpus),
        "runs": str(tmp_path / "runs"),
    })
    return Config(data=data, root=tmp_path, source=None)


@pytest.fixture(scope="session")
def ingested_tiny(tiny_config: Config) -> Config:
    """Run the ingestion pipeline over the fixture corpus and return the config."""
    from specintel.ingest.pipeline import ingest_corpus

    ingest_corpus(tiny_config, doc_ids=["pd", "typec"], render_pages=True, keep_images=True)
    return tiny_config
