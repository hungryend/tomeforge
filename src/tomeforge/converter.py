"""PDF/ebook → EPUB conversion.

PDF: detect whether it's a scan; extract Markdown + images with pdf2md (PyMuPDF's
text layer, or a local Ollama vision model for scans), then let Calibre turn the
Markdown into an EPUB — building the nav from the markdown headings. Other input
formats (MOBI/AZW/…) are handed straight to Calibre.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from tomeforge import pdf2md

DEFAULT_MODEL = "deepseek-ocr:3b"
_TIMEOUT = 900  # seconds for the Calibre step

# Calibre flags. The level*-toc XPaths build a nav TOC from <h1>/<h2>/<h3> in the
# intermediate XHTML (the `h:` prefix is Calibre's XHTML namespace).
_TOC_FLAGS = [
    "--enable-heuristics",
    "--toc-threshold", "6",
    "--max-toc-links", "50",
    "--duplicate-links-in-toc",
    "--level1-toc", "//h:h1",
    "--level2-toc", "//h:h2",
    "--level3-toc", "//h:h3",
]
# PDF → Markdown → EPUB: Calibre's Markdown Input turns #/##/### into real heading
# tags and embeds images referenced relatively from the .md's own directory.
_MD_FLAGS = ["--formatting-type", "markdown", "--paragraph-type", "off",
             "--input-encoding", "utf-8", *_TOC_FLAGS]

_CALIBRE = shutil.which("ebook-convert") or "ebook-convert"
_PDF_EXT = ".pdf"
# Other formats Calibre reads natively → convert directly.
_DIRECT_EXTS = {".mobi", ".azw", ".azw3", ".epub", ".fb2", ".lit", ".pdb", ".rtf",
                ".odt", ".docx", ".htmlz", ".cbz", ".cbr"}


class ConversionError(RuntimeError):
    """Raised on any unrecoverable conversion problem (bad input, Calibre/OCR failure)."""


@dataclass
class ConversionResult:
    output: Path
    engine: str  # 'heuristic' | 'ocr' | 'calibre'
    scanned: bool


def calibre_available() -> bool:
    return shutil.which("ebook-convert") is not None


def ollama_reachable(host: str | None, timeout: float = 5.0) -> bool:
    """True if an Ollama server answers at `host` (used to gate OCR)."""
    if not host:
        return False
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def _run_calibre(src: Path, out: Path, extra_args: list[str], timeout: int = _TIMEOUT) -> None:
    if not calibre_available():
        raise ConversionError("Calibre's `ebook-convert` was not found on PATH — install Calibre.")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_CALIBRE, str(src), str(out), "--no-default-epub-cover", *extra_args]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ConversionError(f"ebook-convert timed out after {timeout}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode(errors="replace").strip()[-500:]
        raise ConversionError(f"ebook-convert failed (exit {proc.returncode}): {tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise ConversionError("ebook-convert produced no output")


def _ocr_mostly_failed(md_path: Path) -> bool:
    """True if most OCR pages errored (pdf2md writes a '(OCR failed on page N…'
    placeholder per failed page) — so we don't emit a junk EPUB of placeholders."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    pages = text.count("<!-- page ")
    failed = text.count("OCR failed on page")
    return pages > 0 and failed >= max(1, (pages + 1) // 2)


def convert(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    ocr: str = "auto",  # "auto" | "always" | "never"
    ollama_host: str | None = None,
    model: str = DEFAULT_MODEL,
    dpi: int = 150,
    ocr_timeout: int = 600,
    num_ctx: int = 8192,
    keep_markdown: bool = False,
    quiet: bool = False,
    work_dir: str | Path | None = None,
) -> ConversionResult:
    """Convert `input_path` to an EPUB at `output_path` (default: alongside input).

    OCR (scanned PDFs) modes: ``auto`` OCRs only when the PDF looks scanned AND an
    Ollama host is set+reachable; ``always`` forces OCR (requires a reachable host);
    ``never`` uses the text layer only.

    ``work_dir`` lets a caller supply the intermediate directory (where pdf2md
    writes ``output.md`` + ``images/`` + the resumable OCR ``pages/`` cache) instead
    of a throwaway temp dir. When given it is NOT deleted — the caller owns its
    lifecycle, which also lets the caller poll ``work_dir/pages`` for OCR progress.
    """
    src = Path(input_path)
    if not src.is_file():
        raise ConversionError(f"input not found: {src}")
    out = Path(output_path) if output_path else src.with_suffix(".epub")
    ext = src.suffix.lower()

    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    # Non-PDF: Calibre handles it directly.
    if ext != _PDF_EXT:
        if ext not in _DIRECT_EXTS:
            raise ConversionError(f"unsupported input type: {ext or '(none)'}")
        log(f"converting {src.name} → {out.name} (Calibre)…")
        _run_calibre(src, out, _TOC_FLAGS)
        return ConversionResult(out, engine="calibre", scanned=False)

    # PDF: pick the extraction engine.
    scanned = pdf2md.pdf_is_scan(str(src))
    host_ok = ollama_reachable(ollama_host)
    if ocr == "always":
        if not ollama_host:
            raise ConversionError("--ocr always requires --ollama-host")
        if not host_ok:
            raise ConversionError(f"OCR server not reachable at {ollama_host}")
        engine = "ollama"
    elif ocr == "never":
        engine = "heuristic"
    else:  # auto
        engine = "ollama" if (scanned and host_ok) else "heuristic"
        if scanned and engine == "heuristic":
            log("! this looks like a scanned/image PDF and no OCR server is configured — "
                "text will be poor. Pass --ocr always --ollama-host <url> to OCR it.")

    caller_owns_work = work_dir is not None
    work = Path(work_dir) if work_dir is not None else Path(mkdtemp(prefix="tomeforge-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        opt = pdf2md.Options(
            out_dir=str(work),
            engine=engine,
            page_images="auto",
            ollama_host=ollama_host or "http://localhost:11434",
            model=model,
            dpi=dpi,
            ocr_timeout=ocr_timeout,
            ocr_num_ctx=num_ctx,
            resume=True,
        )
        log(f"extracting Markdown ({'OCR' if engine == 'ollama' else 'text layer'})…")
        md = Path(pdf2md.convert(str(src), opt, pages_spec="", toc=True, quiet=quiet))
        if not md.exists() or md.stat().st_size == 0:
            raise ConversionError("no Markdown was produced from this PDF")
        if engine == "ollama" and _ocr_mostly_failed(md):
            raise ConversionError(
                "OCR failed on most pages — the model may be out of memory or unsuitable; "
                "try a smaller --model or a host with a GPU."
            )
        log(f"building EPUB → {out}…")
        _run_calibre(md, out, _MD_FLAGS)
        if keep_markdown:
            dest = out.with_suffix(".md")
            shutil.copyfile(md, dest)
            log(f"kept Markdown → {dest}")
    finally:
        if not caller_owns_work:
            shutil.rmtree(work, ignore_errors=True)

    return ConversionResult(out, engine="ocr" if engine == "ollama" else "heuristic", scanned=scanned)
