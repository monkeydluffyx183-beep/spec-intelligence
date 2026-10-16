# OCR / document-AI benchmark (WS1, Task 2 Step 4)

Crops benchmarked: **20** in 4 document(s) (pd, typec, usb4, tech2), rasterised at 220 dpi.

Lower is better for CER/WER; higher is better for number accuracy.

| backend | available | crops | CER | WER | word recall | spurious | number acc. | s/crop | RSS MB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noop | yes | 19/20 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 55.711 |  |
| pymupdf-raster+ocr | **no** | 0/20 | — | — | — | — | — | — | — | dependency missing |
| pymupdf-text | yes | 19/20 | 0.586 | 0.774 | 0.943 | 0.305 | 0.908 | 0.003 | 55.695 |  |
| rapidocr | **no** | 0/20 | — | — | — | — | — | — | — | dependency missing |
| unlimited-ocr | **no** | 0/20 | — | — | — | — | — | — | — | no endpoint configured (needs an NVIDIA GPU; see README § Unlimited-OCR) |

## By content type (CER)

| backend | figure | page |
| --- | --- | --- |
| noop | 1.000 | 1.000 |
| pymupdf-raster+ocr | — | — |
| pymupdf-text | 0.573 | 0.592 |
| rapidocr | — | — |
| unlimited-ocr | — | — |

## By document (CER)

| backend | pd | typec | usb4 | tech2 |
| --- | --- | --- | --- | --- |
| noop | 1.000 | 1.000 | 1.000 | 1.000 |
| pymupdf-raster+ocr | — | — | — | — |
| pymupdf-text | 0.774 | 0.246 | 0.380 | 0.632 |
| rapidocr | — | — | — | — |
| unlimited-ocr | — | — | — | — |

## Read this table honestly

* CER and WER compare the recovered text with the reference as one string, so they punish reading order as much as missing text. **word recall** and **spurious** split those two failures apart: recall is how much of the page came back, spurious is how much of the output is not on the page at all.
* `pymupdf-text` reads the vector text layer. It is exact and free where a text layer exists (USB-PD, Type-C, USB4 as published) and returns nothing on a scan or a rasterised figure — which is why it is the baseline, not the answer. Its **spurious** share is the running headers, footers and page numbers, which are on the page but not part of the content being reconstructed; that is also why the pipeline strips them before indexing.
* `noop` is a control. Every figure it scores is a text-recovery failure the pipeline would have to cover another way.
* `unlimited-ocr` needs an NVIDIA GPU and a served model. When the row says **no**, the machine could not run it and no number is reported: see `docs/OCR_BENCHMARK.md` for the protocol to run it and what it is expected to fix (dense figure labels, where raster+OCR is weakest).
