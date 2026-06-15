"""tomeforge — turn PDFs (and MOBI/AZW/…) into clean, reflowable EPUBs with a
linked table of contents.

PDFs go PDF → Markdown (PyMuPDF, via the bundled pdf2md) → EPUB (Calibre), with an
optional local-LLM (Ollama) OCR fallback for scanned/image PDFs. Other formats are
converted by Calibre directly.
"""

from tomeforge.converter import ConversionError, ConversionResult, convert

__all__ = ["ConversionError", "ConversionResult", "convert"]
__version__ = "0.1.0"
