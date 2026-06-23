# tomeforge

<p>
  <a href="https://github.com/hungryend/tomeforge/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/hungryend/tomeforge/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://github.com/hungryend/tomeforge/releases"><img alt="Release" src="https://img.shields.io/github/v/release/hungryend/tomeforge"/></a>
  <a href="https://github.com/hungryend/tomeforge/pkgs/container/tomeforge"><img alt="GHCR" src="https://img.shields.io/badge/ghcr.io-hungryend%2Ftomeforge-blue?logo=docker"/></a>
</p>

Turn PDFs (and MOBI/AZW/…) into clean, **reflowable EPUBs with a linked table of contents**.

PDFs take a Markdown detour for quality: **PDF → Markdown** (via [PyMuPDF], with font‑size +
bookmark heading detection) **→ EPUB** (via [Calibre], which embeds the figures and builds the nav
from the Markdown headings). For **scanned / image PDFs**, an optional local **LLM vision‑OCR**
(via [Ollama]) reads the pages instead of the (often garbled or absent) text layer. Other formats
are handed straight to Calibre.

## Why

Calibre's direct PDF→EPUB reflow is mediocre — weak structure, poor or missing TOC. Going through
Markdown first gives real headings (→ a proper linked TOC) and clean inline figures. And when a PDF
is just page images, tomeforge detects it and can OCR it instead of producing a useless EPUB.

## Requirements

- **Python ≥ 3.10**
- **[Calibre]** on your `PATH` (provides `ebook-convert`) — the EPUB step
- **PyMuPDF** (installed automatically) — the PDF→Markdown step
- *(optional)* an **[Ollama]** server with a vision model — only for OCR of scanned PDFs

## Install

```bash
pip install git+https://github.com/hungryend/tomeforge.git
# or, from a clone:
pip install .
```

## Usage

```bash
tomeforge book.pdf                      # → book.epub (next to the input)
tomeforge book.pdf -o ~/ebooks/book.epub
tomeforge novel.mobi                    # non-PDF formats go straight through Calibre
tomeforge book.pdf --keep-markdown      # also write the intermediate book.md
```

As a library:

```python
from tomeforge import convert
result = convert("book.pdf", "book.epub")
print(result.engine, result.scanned)    # 'heuristic' | 'ocr' | 'calibre'
```

### Scanned PDFs (optional OCR)

A born‑digital PDF needs no OCR. A **scan** (page images, no real text) does — tomeforge detects this
and, if you point it at an Ollama vision model, OCRs each page:

```bash
# 1. Start a local Ollama with an OCR model (pulls ~6.7 GB once):
docker compose --profile ocr up -d
# 2. Convert, forcing OCR:
tomeforge scanned-book.pdf --ocr always --ollama-host http://localhost:11434
```

- `--ocr auto` (default): OCR only when the PDF looks scanned **and** `--ollama-host` is set + reachable.
- `--ocr always`: force OCR (requires a reachable host). `--ocr never`: text layer only.
- `--model` picks the vision model (default `deepseek-ocr:3b`; alternatives: `qwen2.5vl:3b`,
  `minicpm-v`, `granite3.2-vision`).

**Performance:** OCR on CPU is slow — minutes per page — so a big book can take hours. For real use,
run Ollama on a **GPU** box (NVIDIA, or AMD via [ROCm]) and point `--ollama-host` at it; there
`deepseek-ocr` fits in VRAM and runs in seconds per page. Set `OLLAMA_KEEP_ALIVE` high on that host so
the model stays resident across a long book.

## Run as a service (sidecar)

Beyond the CLI, tomeforge can run as a small HTTP service so another app can offload
conversion (and its heavyweight PyMuPDF + Calibre deps) instead of bundling them:

```bash
# Prebuilt image — published to GHCR by CI on every release:
docker run -p 8400:8400 ghcr.io/hungryend/tomeforge:latest

# …or build from a clone:
docker compose up -d --build

# …or without Docker:
pip install "tomeforge[service]"          # adds fastapi + uvicorn
tomeforge serve --port 8400
```

The published image bundles Calibre + PyMuPDF and self-reports health at `/healthz`.
Tags: `:latest`, the SemVer release (`:vX.Y.Z`), and `:sha-<short>`. Scanned-PDF OCR
delegates to an external Ollama (see below) — none is bundled in the image.

```bash
# Submit a file, then poll the returned job_id until it's done.
curl -F file=@book.pdf http://localhost:8400/convert
#   → {"job_id": "…", "status": "queued", …}
curl http://localhost:8400/jobs/<job_id>          # status + phase (e.g. "OCR page 3/40")
curl -OJ http://localhost:8400/jobs/<job_id>/result   # the EPUB, once status == "done"
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/convert` | multipart `file` + optional `ocr` / `ollama_host` / `model` / `dpi` / `ocr_timeout` form fields → `{job_id}` |
| `GET` | `/jobs/{id}` | `{status, phase, engine, scanned, error}` (`phase` reports `OCR page N/M`) |
| `GET` | `/jobs/{id}/result` | the converted EPUB (`409` until done) |
| `DELETE` | `/jobs/{id}` | drop the job's temp files |
| `GET` | `/healthz` | liveness + whether Calibre is on PATH |

Scanned-PDF OCR works the same way — pass `ocr=always` (or `auto`) and an `ollama_host`
the **service** can reach (e.g. `http://ollama:11434` on the compose network). The job
registry is in-memory: the sidecar is a stateless worker, so a restart drops in-flight
jobs and the caller re-submits.

## How it works

1. **Classify** the PDF: most pages a single full‑page image with little *visible* text ⇒ a scan
   (an invisible OCR layer doesn't count).
2. **Extract Markdown** with the bundled `pdf2md` — text‑layer heuristics for born‑digital PDFs, or
   the Ollama vision model for scans (per‑page, cached + resumable).
3. **Build the EPUB** with Calibre's Markdown Input: `#`/`##`/`###` become real headings, the
   `--levelN-toc` flags build the nav, and relatively‑referenced images are embedded.

## License

**[AGPL‑3.0‑or‑later](LICENSE).** tomeforge bundles `pdf2md`, which depends on **PyMuPDF (AGPL‑3.0**,
or a commercial licence from Artifex) — so the project as a whole is AGPL. Calibre and Ollama are
invoked as separate programs.

[PyMuPDF]: https://pymupdf.readthedocs.io/
[Calibre]: https://calibre-ebook.com/
[Ollama]: https://ollama.com/
[ROCm]: https://rocm.docs.amd.com/
