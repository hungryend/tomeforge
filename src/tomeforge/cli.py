"""Command-line entry point: `tomeforge INPUT [-o OUT.epub] [options]`."""

from __future__ import annotations

import argparse
import sys

from tomeforge import __version__
from tomeforge.converter import ConversionError, calibre_available, convert


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    # `tomeforge serve …` runs the optional HTTP sidecar (separate arg grammar).
    if args_in and args_in[0] == "serve":
        from tomeforge.service import main as serve_main

        return serve_main(args_in[1:])

    p = argparse.ArgumentParser(
        prog="tomeforge",
        description="Convert PDFs (and MOBI/AZW/…) into clean, reflowable EPUBs with a "
        "linked table of contents. Scanned PDFs can be OCR'd by a local Ollama vision model.",
    )
    p.add_argument("input", help="source file (.pdf, .mobi, .azw, .azw3, .epub, …)")
    p.add_argument("-o", "--output", help="output .epub path (default: alongside the input)")
    p.add_argument(
        "--ocr",
        choices=("auto", "always", "never"),
        default="auto",
        help="OCR scanned PDFs via Ollama: auto (only if scanned AND --ollama-host is "
        "set + reachable), always (force; needs a reachable host), never (default: auto)",
    )
    p.add_argument("--ollama-host", metavar="URL",
                   help="Ollama base URL for OCR, e.g. http://localhost:11434")
    p.add_argument("--model", default="deepseek-ocr:3b",
                   help="Ollama vision model for OCR (default: deepseek-ocr:3b)")
    p.add_argument("--dpi", type=int, default=150, help="page render DPI for OCR (default: 150)")
    p.add_argument("--ocr-timeout", type=int, default=600,
                   help="per-page OCR timeout in seconds (default: 600)")
    p.add_argument("--keep-markdown", action="store_true",
                   help="also write the intermediate Markdown next to the EPUB")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    p.add_argument("--version", action="version", version=f"tomeforge {__version__}")
    args = p.parse_args(argv)

    if not calibre_available():
        print("error: Calibre's `ebook-convert` is required on PATH. Install Calibre "
              "(https://calibre-ebook.com/).", file=sys.stderr)
        return 2

    try:
        result = convert(
            args.input, args.output,
            ocr=args.ocr, ollama_host=args.ollama_host, model=args.model,
            dpi=args.dpi, ocr_timeout=args.ocr_timeout,
            keep_markdown=args.keep_markdown, quiet=args.quiet,
        )
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        print(f"done: {result.output}  (engine={result.engine}, scanned={result.scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
