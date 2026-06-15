"""Tests that don't need Calibre. The PyMuPDF scan-detection test importorskips
fitz so the suite still runs on a host without the `pdf` extra; the rest stub or
avoid it. The Calibre PDF→EPUB step is exercised manually (see the README)."""

from __future__ import annotations

import pytest

from tomeforge import converter as C


def test_unsupported_input(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(C.ConversionError):
        C.convert(f)


def test_missing_input(tmp_path):
    with pytest.raises(C.ConversionError):
        C.convert(tmp_path / "nope.pdf")


def test_ollama_reachable_false():
    assert C.ollama_reachable(None) is False
    assert C.ollama_reachable("") is False
    assert C.ollama_reachable("http://127.0.0.1:1") is False  # nothing listening


def test_ocr_mostly_failed(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("<!-- page 1 -->\nReal text\n<!-- page 2 -->\nMore real text")
    assert C._ocr_mostly_failed(good) is False
    bad = tmp_path / "bad.md"
    bad.write_text(
        "<!-- page 1 -->\n*(OCR failed on page 1: x)*\n"
        "<!-- page 2 -->\n*(OCR failed on page 2: y)*"
    )
    assert C._ocr_mostly_failed(bad) is True


def test_pdf_is_scan(tmp_path):
    fitz = pytest.importorskip("fitz")
    from PIL import Image

    from tomeforge.pdf2md import pdf_is_scan

    img = tmp_path / "page.png"
    Image.new("RGB", (1000, 1400), (20, 40, 60)).save(img)

    # Scan: full-page image + an INVISIBLE OCR text layer.
    scan = tmp_path / "scan.pdf"
    d = fitz.open()
    for _ in range(3):
        pg = d.new_page(width=600, height=800)
        pg.insert_image(pg.rect, filename=str(img))
        pg.insert_text((72, 120), "ocr layer text " * 20, render_mode=3)  # invisible
    d.save(str(scan))
    d.close()
    assert pdf_is_scan(str(scan)) is True

    # Born-digital: full-page background image + real VISIBLE text.
    born = tmp_path / "text.pdf"
    d = fitz.open()
    for _ in range(3):
        pg = d.new_page(width=600, height=800)
        pg.insert_image(pg.rect, filename=str(img))
        pg.insert_text((72, 120), "Real visible body text. " * 30)
    d.save(str(born))
    d.close()
    assert pdf_is_scan(str(born)) is False
