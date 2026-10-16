# OCR / document-AI benchmark (WS1, Task 2 Step 4)

**Question this answers:** for each kind of content in these four specifications —
prose pages, bit-field tables, multi-page parameter tables, state-machine
diagrams, timing/waveform diagrams, test-procedure tables — how much text does
each extractor actually recover, and what does it cost in time and memory?

Run it:

```bash
PYTHONPATH=src python3 benchmarks/ocr/run_bench.py \
    [--docs pd,typec,usb4,tech2] [--backends pymupdf-text,unlimited-ocr] \
    [--limit 25] [--dpi 220] [--unlimited-ocr-endpoint http://gpu-box:8000]
```

Outputs: `benchmarks/ocr/results/results.json` (machine-readable, one row per
crop × backend) and `benchmarks/ocr/results/report.md` (the tables below).
`--unlimited-ocr-endpoint` may also come from `SPECINTEL_UNLIMITED_OCR_ENDPOINT`,
the model from `SPECINTEL_UNLIMITED_OCR_MODEL`.

## Method

* **Crops.** Page images and figure crops come from the ingestion pipeline's own
  output (`data/processed/<doc_id>/pages/*.png` and `figures/*.png`) at 220 dpi,
  so the benchmark measures the images the retrieval path really indexes. Each
  crop records `doc_id`, `page`, and `kind` (`page` | `figure`).
* **Reference text.** The text layer of the same crop, extracted with
  `pymupdf` and normalised (whitespace folded, hyphenation joined, running
  headers/footers and page numbers removed — the same `BOILERPLATE_RE` the
  pipeline strips with). It is a *gold* reference in the strict sense: it is what
  the document actually contains.
* **Metrics.** CER and WER (Levenshtein over characters and words), **word
  recall** (how much of the reference came back), **spurious share** (how much of
  the output is not in the reference at all), **number accuracy** (fraction of
  numeric tokens recovered exactly) and wall-clock seconds per crop, plus RSS.
* **Honesty rules.** A backend that cannot run on this machine is recorded
  `available: false` with every metric `null` and is **excluded from the
  averages** — a missing dependency is never scored as zero. `noop` is a
  deliberate control that returns nothing (CER 1.000) so the scale is anchored.

## Results — this machine

Hardware: 2 vCPU, 3.8 GB RAM, **no GPU** (`hardware` block in `results.json`).
25 crops, four documents, 220 dpi. Numbers are the run recorded in
`benchmarks/ocr/results/`; regenerate with the command above.

| backend | available | crops | CER | WER | word recall | spurious | number acc. | s/crop | RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pymupdf-text` | yes | 19/20 | 0.586 | 0.774 | 0.943 | 0.305 | 0.908 | 0.003 | 55.7 |
| `noop` (control) | yes | 19/20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 55.7 |
| `pymupdf-raster+ocr` | **no** | 0/20 | — | — | — | — | — | — | — |
| `rapidocr` | **no** | 0/20 | — | — | — | — | — | — | — |
| `unlimited-ocr` | **no** | 0/20 | — | — | — | — | — | — | — |

By content kind (CER) and by document (CER) are in
`benchmarks/ocr/results/report.md`; the per-crop rows are in `results.json`.

### How to read it

* **CER/WER punish reading order as much as missing text.** Joining a page into
  one string reorders multi-column layouts, so a page that was read perfectly but
  in the wrong order still scores badly. That is why **word recall (0.943)** and
  **spurious share (0.305)** are reported next to them: recall says almost the
  whole page came back, and the spurious share is the furniture (running headers,
  footers, page numbers) that is on the page but not part of the content — which
  is exactly what the ingest stage strips before indexing.
* **`pymupdf-text` is the baseline, not the answer.** Where a text layer exists it
  is free and exact. It returns *nothing* on a scan or on a rasterised figure —
  the score for a crop with no text layer is `null`, not `0`, and those crops are
  the work left for OCR/VLM.
* **`pymupdf-raster+ocr` and `rapidocr` need extra dependencies** (`pytesseract`,
  `rapidocr-onnxruntime`); they are reported unavailable rather than failing the
  run. Installing them turns those rows into numbers.
* **`unlimited-ocr` needs an NVIDIA GPU.** It is the reason the row exists: see
  the protocol below. Nothing is reported for it on this machine.

## Running Unlimited-OCR (expected to close the raster gap)

[`baidu/Unlimited-OCR`](https://github.com/baidu/Unlimited-OCR) (MIT, ~3B
parameters, built on DeepSeek-OCR) needs an NVIDIA GPU; the paper is
arXiv:2606.23050. It cannot be installed or run on this machine (no GPU, and the
sandbox has no access to the model host), so its row is `available: false`.

Protocol for whoever has the GPU box:

1. Serve the model with vLLM on the GPU machine, OpenAI-compatible:
   `vllm serve baidu/Unlimited-OCR --port 8000 --max-model-len 8192`.
2. Point the benchmark at it:
   `PYTHONPATH=src python3 benchmarks/ocr/run_bench.py --backends unlimited-ocr,pymupdf-text \
      --unlimited-ocr-endpoint http://<host>:8000` (or export the env var).
3. The adapter (`specintel/ingest/ocr.py::UnlimitedOCRBackend`) sends the crop as
   a data URI, asks for the document text, parses the marked block output
   (`<|ref|>…<|/ref|><|det|>bbox<|/det|>`) into typed blocks with page offsets,
   and returns `.text` for the metric code. `probe(endpoint)` is called first: an
   unreachable endpoint reports the backend as unavailable instead of scoring it.
4. Compare against the same crops: the interesting columns are **word recall on
   figure crops** and **number accuracy** — dense diagram labels and waveform
   timings are where the text-layer baseline is weakest (it scores nothing at all
   on a raster figure).
5. Record the model revision and the GPU in `results.json` (the `hardware` block
   is captured automatically) and update the table above.

## Conclusions that changed the pipeline

* Text-layer extraction is the primary path for pages that have one; the OCR
  adapters are for crops **without** a text layer (scans, rasterised figures) and
  for anything a VLM must describe rather than transcribe.
* The spurious share justified keeping the boilerplate stripper in the parser:
  without it, every page contributes its header and footer to the index and to the
  "duplicate evidence" problem the citation stage has to fix.
* Per-crop cost (0.003 s) means OCR is never the bottleneck for the page path;
  the expensive part is the VLM description of figures, which is why the figure
  path is switchable (`SPECINTEL_ABLATE_FIGURE_TEXT`) and measured separately in
  the ablations.
