#!/usr/bin/env python3
"""pdf2md - convert a PDF into Markdown, extracting images as detected.

Why this exists
---------------
Off-the-shelf "PDF -> Markdown" libraries (pymupdf4llm, marker, docling) either
choke on certain PDFs or need heavy ML stacks. This is a small, dependency-light
converter built directly on PyMuPDF that works well on text-based PDFs and is
smart about images.

How text becomes Markdown
-------------------------
Text is read from the PDF's text layer (no OCR) and reconstructed into a stream
of "units" (headings, paragraphs, lists, images):
  * Headings come from font *size* relative to the body text (for short blocks),
    and from bold / ALL-CAPS run-in or standalone headings that share the body
    size (common in RPG books and manuals) - see `heading_level` / `split_runin`.
  * **Bold** / *italic* come from span font flags / font names.
  * Bullet lists are detected from leading bullet glyphs, with wrapped lines
    folded back into their item.
  * Paragraphs that the PDF split into one block per visual line are reflowed
    back together using geometry (alignment + line spacing) - see the merge pass
    in `page_units`.
  * Two-column pages are read left column then right column.
  * Repeated page headers/footers and bare page numbers are stripped.

How images are handled (the interesting part)
---------------------------------------------
Many real-world PDFs (RPG books, magazines, scanned-and-OCR'd files) place a
single full-page image on every page - a parchment texture or a flattened
render - with a searchable text layer on top. Naively "extract every image"
then yields dozens of near-identical page backgrounds.

So each image is classified by how much of the page it covers:
  * Partial-page image  -> a genuine inline figure: extracted to images/ and
    linked at its position in the text.
  * Full-page image     -> treated as a page background. It is only emitted
    (as a rendered page image) when the page is *image-dominant*, i.e. it has
    little or no text - that is how maps and full-page art get captured while
    plain text pages stay clean.

Use --page-images all to render every page as an image (maximum visual
fidelity, e.g. to keep decorative art baked into text pages), or
--page-images none for text only.
"""

# Bundled with tomeforge. Requires PyMuPDF (fitz) — AGPL-3.0 (or commercial from
# Artifex). fitz is lazy-imported (see _require_fitz) so this module stays
# importable without PyMuPDF present (e.g. for `tomeforge --help`).
from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

# PyMuPDF (fitz) is lazy-imported so this module imports without PyMuPDF present.
# _require_fitz() loads it on first real use.
fitz = None


def _require_fitz():
    """Import PyMuPDF on first use; raises ImportError if the `pdf` extra is absent."""
    global fitz
    if fitz is None:
        import fitz as _fitz
        fitz = _fitz
    return fitz


# --- font flag bits (PyMuPDF span['flags']) -------------------------------
FLAG_ITALIC = 1 << 1   # 2
FLAG_BOLD = 1 << 4     # 16

# Leading glyphs that indicate a bullet list item: common Unicode bullets, the
# U+FFFD replacement char (symbol-font bullets that lost their mapping), and
# ascii fallbacks.
BULLET_RE = re.compile(r"^\s*[•‣⁃▪●◦∙·�]\s+")
ASCII_BULLET_RE = re.compile(r"^\s*[\-\*–—]\s+")


@dataclass
class Options:
    out_dir: str
    images: bool = True
    page_images: str = "auto"        # auto | all | none
    fullpage_cover: float = 0.7      # >= this fraction of page area -> "full-page"
    text_min: int = 220              # below this char count a page is image-dominant
    dpi: int = 150
    emphasis: bool = True
    split: bool = False
    rel_img_dir: str = "images"
    heading_ratio: float = 1.3       # min size/body ratio to be a (size) heading
    max_heading_words: int = 12      # blocks longer than this are never headings
    keep_furniture: bool = False     # keep repeated headers/footers + page numbers
    margin: float = 0.07             # top/bottom fraction treated as the page margin
    reflow: bool = True              # merge one-line-per-block paragraphs back together
    runin_headings: bool = True      # detect bold/ALL-CAPS run-in & standalone headings
    runin_level: int = 3
    # --- engine: "heuristic" (text layer) or "ollama" (vision OCR) ---
    engine: str = "heuristic"
    model: str = "deepseek-ocr:3b"
    ollama_host: str = "http://localhost:11434"
    ocr_prompt: str = "Convert the document to markdown."
    ocr_num_ctx: int = 8192
    ocr_timeout: int = 600
    clean_ocr: bool = True
    ocr_headings: bool = True        # promote plain ALL-CAPS titles in OCR output to headings
    bookmark_headings: bool = True   # promote bookmark section titles to headings (for TOC links)
    resume: bool = False
    merge: bool = False              # skip conversion; just merge <out>/pages/*.md


@dataclass
class Item:
    """A text block or inline image on a page, with its bounding box so we can
    sort it into reading order."""
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str            # "text" | "image"
    payload: object

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class Unit:
    """One rendered piece of Markdown, plus geometry used by the reflow pass."""
    kind: str                       # "heading" | "para" | "list" | "image"
    md: str = ""                    # final markdown (heading / list / image)
    lines: list = field(default_factory=list)  # rendered line strings (para only)
    level: int = 0
    x0: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    dom: float = 0.0
    single: bool = False            # occupies a single text line (mergeable)


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------
def _block_plain(block) -> str:
    return " ".join("".join(s["text"] for s in line["spans"]) for line in block["lines"])


def _dominant_size(block) -> int:
    sizes: Counter[int] = Counter()
    for line in block["lines"]:
        for s in line["spans"]:
            t = s["text"].strip()
            if t:
                sizes[round(s["size"])] += len(t)
    return sizes.most_common(1)[0][0] if sizes else 0


def _norm_furniture(text: str) -> str:
    """Normalise a header/footer for cross-page comparison: drop digits (page
    numbers vary) and punctuation, lower-case the rest."""
    return re.sub(r"[^a-z]+", " ", re.sub(r"\d+", "", text.lower())).strip()


def heading_level(size: float, body: float, min_ratio: float) -> int:
    """Map a font size to a heading level (1..3), or 0 for body text, using the
    ratio to the body size so it is robust across documents."""
    if body <= 0:
        return 0
    r = size / body
    if r < min_ratio:
        return 0
    if r >= 1.85:
        return 1
    if r >= 1.45:
        return 2
    return 3


def _is_caps_token(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def split_runin(plain: str):
    """Detect a bold/ALL-CAPS run-in or standalone heading at the start of a
    block. Returns (heading, rest):
      ("GOBLIN AMBUSH", "Read the following ...")   # run-in heading + body
      ("THE DUNGEON MASTER", "")                     # standalone heading
      (None, None)                                   # not a heading
    Conservative: the leading caps run must be >=2 words, or a single word of
    >=5 letters (so acronyms like DM / XP / DC are not promoted)."""
    words = plain.split()
    if not words:
        return None, None
    i = 0
    while i < len(words) and i < 8 and _is_caps_token(words[i]):
        i += 1
    if i == 0:
        return None, None
    prefix, rest = words[:i], words[i:]
    # Fold a tiny trailing remainder into the heading: it is almost always an
    # OCR-mangled caps word (e.g. "RULES TO GAME BY" read as "...GAME By").
    if len(rest) == 1 and len(rest[0]) <= 3:
        prefix, rest = prefix + rest, []
    letters = sum(c.isalpha() for w in prefix for c in w)
    if len(prefix) < 2 and letters < 5:
        return None, None
    if len(prefix) > 7:
        return None, None
    return " ".join(prefix), " ".join(rest)


# --------------------------------------------------------------------------
# Span / line -> Markdown (bold / italic)
# --------------------------------------------------------------------------
def _span_style(span) -> tuple[bool, bool]:
    flags = span.get("flags", 0)
    name = span.get("font", "").lower()
    bold = bool(flags & FLAG_BOLD) or any(k in name for k in ("bold", "black", "heavy", "semibold"))
    italic = bool(flags & FLAG_ITALIC) or any(k in name for k in ("italic", "oblique"))
    return bold, italic


def _wrap(text: str, bold: bool, italic: bool) -> str:
    if not text.strip():
        return text
    mark = "***" if (bold and italic) else "**" if bold else "*" if italic else ""
    if not mark:
        return text
    lead = " " if text[:1].isspace() else ""
    trail = " " if text[-1:].isspace() else ""
    return f"{lead}{mark}{text.strip()}{mark}{trail}"


def line_md(spans, emphasis: bool = True) -> str:
    """Join a line's spans into Markdown, merging adjacent runs of the same
    style so we get `**two words**` rather than `**two** **words**`."""
    if not emphasis:
        return "".join(s["text"] for s in spans)
    out: list[str] = []
    cur = ""
    cur_style: tuple[bool, bool] | None = None
    for s in spans:
        t = s["text"]
        if t == "":
            continue
        style = _span_style(s)
        if cur_style is None or style == cur_style:
            cur += t
            cur_style = style
        else:
            out.append(_wrap(cur, *cur_style))
            cur, cur_style = t, style
    if cur_style is not None:
        out.append(_wrap(cur, *cur_style))
    return "".join(out)


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, de-hyphenating words split across
    a line break ('Dun-' + 'geon' -> 'Dungeon')."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif re.search(r"[A-Za-z]-$", out) and re.match(r"[a-z]", ln):
            out = out[:-1] + ln
        else:
            out += " " + ln
    return out


def _render_list(rendered: list[tuple[bool, str]]) -> str:
    """rendered = [(is_bullet, line_md), ...] -> markdown list (folding wrapped
    continuation lines into the preceding item)."""
    lead: list[str] = []
    items: list[str] = []
    cur: str | None = None
    for is_bullet, md in rendered:
        if is_bullet:
            if cur is not None:
                items.append(cur)
            cur = (BULLET_RE.sub("", md, 1) if BULLET_RE.match(md)
                   else ASCII_BULLET_RE.sub("", md, 1)).strip()
        elif cur is None:
            lead.append(md)
        else:
            cur = join_lines([cur, md])
    if cur is not None:
        items.append(cur)
    parts = []
    if lead:
        parts.append(join_lines(lead))
    if items:
        parts.append("\n".join("- " + it for it in items))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Block -> Unit(s)
# --------------------------------------------------------------------------
def classify_block(block, body: float, opt: Options) -> list[Unit]:
    plain = _block_plain(block).strip()
    if not plain or sum(c.isalpha() for c in plain) == 0:
        return []  # empty, bare page numbers, rule lines, decorative numerals

    bb = block["bbox"]
    x0, y0, y1 = bb[0], bb[1], bb[3]
    dom = _dominant_size(block)
    single = bool(dom) and (y1 - y0) <= 1.6 * dom
    n_words = len(plain.split())
    n_lines = len(block["lines"])
    n_letters = sum(c.isalpha() for c in plain)

    def mk(kind, **kw):
        return Unit(kind, x0=x0, y0=y0, y1=y1, dom=dom, **kw)

    # 1) size-based heading (short block, clearly larger than body)
    level = heading_level(dom, body, opt.heading_ratio)
    if level and n_words <= opt.max_heading_words and n_lines <= 3 and n_letters >= 2:
        return [mk("heading", md="#" * level + " " + re.sub(r"\s+", " ", plain), level=level)]

    # 2) bullet list?
    rendered = []
    for line in block["lines"]:
        raw = "".join(s["text"] for s in line["spans"])
        if raw.strip():
            rendered.append((bool(BULLET_RE.match(raw) or ASCII_BULLET_RE.match(raw)),
                             line_md(line["spans"], opt.emphasis)))
    if any(b for b, _ in rendered):
        return [mk("list", md=_render_list(rendered))]

    # 3) bold/ALL-CAPS run-in or standalone heading (same size as body)
    if opt.runin_headings:
        head, rest = split_runin(plain)
        if head is not None:
            lvl = opt.runin_level
            units = [mk("heading", md="#" * lvl + " " + head, level=lvl)]
            if rest:
                units.append(mk("para", lines=[rest]))  # body kept plain
            return units

    # 4) ordinary paragraph (mergeable if it is a single visual line)
    return [mk("para", lines=[md for _, md in rendered], single=single)]


# --------------------------------------------------------------------------
# Document analysis: body font size + repeated header/footer text
# --------------------------------------------------------------------------
def analyze_document(doc, margin: float):
    size_chars: Counter[int] = Counter()
    margin_text: Counter[str] = Counter()
    n = doc.page_count
    for pno in range(n):
        page = doc[pno]
        H = page.rect.height or 1.0
        top = page.rect.y0
        seen: set[str] = set()
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    t = s["text"].strip()
                    if t:
                        size_chars[round(s["size"])] += len(t)
            in_top = (b["bbox"][3] - top) <= margin * H
            in_bottom = (b["bbox"][1] - top) >= (1 - margin) * H
            if in_top or in_bottom:
                norm = _norm_furniture(_block_plain(b).strip())
                if norm and norm not in seen:
                    margin_text[norm] += 1
                    seen.add(norm)
    body = size_chars.most_common(1)[0][0] if size_chars else 0
    thresh = max(3, n // 20)
    furniture = {t for t, c in margin_text.items() if c >= thresh}
    return body, furniture


def is_furniture(block, page, furniture: set[str], margin: float) -> bool:
    H = page.rect.height or 1.0
    top = page.rect.y0
    in_top = (block["bbox"][3] - top) <= margin * H
    in_bottom = (block["bbox"][1] - top) >= (1 - margin) * H
    if not (in_top or in_bottom):
        return False
    return _norm_furniture(_block_plain(block).strip()) in furniture


# --------------------------------------------------------------------------
# Reading order (1 or 2 columns)
# --------------------------------------------------------------------------
def detect_columns(items: list[Item], page: "fitz.Rect") -> int:
    text_items = [it for it in items if it.kind == "text"]
    if len(text_items) < 4:
        return 1
    pw = page.width
    mid = page.x0 + pw / 2.0
    left = [it for it in text_items if it.x1 <= mid + pw * 0.05]
    right = [it for it in text_items if it.x0 >= mid - pw * 0.05]
    return 2 if (len(left) >= 3 and len(right) >= 3) else 1


def reading_order(items: list[Item], page: "fitz.Rect") -> list[Item]:
    if detect_columns(items, page) == 2:
        mid = page.x0 + page.width / 2.0
        return sorted(items, key=lambda it: (0 if it.cx < mid else 1, it.y0, it.x0))
    return sorted(items, key=lambda it: (it.y0, it.x0))


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def save_inline_image(doc, xref: int, img_dir: str, stem: str) -> str | None:
    try:
        info = doc.extract_image(xref)
    except Exception:
        return None
    data = info.get("image")
    if not data:
        return None
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"{stem}.{info.get('ext', 'png')}")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def render_page_image(page, img_dir: str, stem: str, dpi: int) -> str:
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"{stem}.png")
    page.get_pixmap(dpi=dpi).save(path)
    return path


# --------------------------------------------------------------------------
# Per-page conversion
# --------------------------------------------------------------------------
def page_units(doc, pno, body, furniture, opt, img_dir, counters) -> list[Unit]:
    """Build the ordered unit stream for a page (text + inline images), then
    reflow consecutive single-line paragraph blocks into real paragraphs."""
    page = doc[pno]
    rect = page.rect
    page_area = rect.width * rect.height or 1.0

    inline_imgs, fullpage_imgs = [], []
    for info in page.get_image_info(xrefs=True):
        bbox = fitz.Rect(info["bbox"])
        info["_bbox"] = bbox
        if (bbox.width * bbox.height) / page_area >= opt.fullpage_cover:
            fullpage_imgs.append(info)
        elif info.get("xref"):
            inline_imgs.append(info)

    items: list[Item] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        if not any(s["text"].strip() for line in block["lines"] for s in line["spans"]):
            continue
        if not opt.keep_furniture and is_furniture(block, page, furniture, opt.margin):
            continue
        b = block["bbox"]
        items.append(Item(b[0], b[1], b[2], b[3], "text", block))
    if opt.images:
        for info in inline_imgs:
            b = info["_bbox"]
            items.append(Item(b.x0, b.y0, b.x1, b.y1, "image", info))

    units: list[Unit] = []
    for it in reading_order(items, rect):
        if it.kind == "text":
            units.extend(classify_block(it.payload, body, opt))
        else:
            counters["img"] = counters.get("img", 0) + 1
            stem = f"p{pno + 1:03d}-img{counters['img']:02d}"
            path = save_inline_image(doc, it.payload["xref"], img_dir, stem)
            if path:
                units.append(Unit("image", md=f"![figure]({opt.rel_img_dir}/{os.path.basename(path)})"))

    # Full-page render policy (maps / full-page art).
    want_page_image = opt.images and opt.page_images != "none" and (
        opt.page_images == "all"
        or (bool(fullpage_imgs) and len(page.get_text().strip()) < opt.text_min)
    )
    if want_page_image:
        path = render_page_image(page, img_dir, f"page-{pno + 1:03d}", opt.dpi)
        units.append(Unit("image", md=f"![Page {pno + 1}]({opt.rel_img_dir}/{os.path.basename(path)})"))

    return units


def render_units(units: list[Unit], opt: Options) -> str:
    parts: list[str] = []
    run: list[Unit] = []
    anchor_x0 = last_y1 = run_dom = 0.0

    def flush():
        if run:
            parts.append(join_lines([ln for u in run for ln in u.lines]))
            run.clear()

    for u in units:
        if u.kind == "para":
            cont = (opt.reflow and run and run[-1].single and u.single
                    and abs(u.x0 - anchor_x0) <= 12
                    and 0 <= (u.y0 - last_y1) <= 1.1 * (u.dom or 8)
                    and abs((u.dom or 0) - run_dom) <= 1)
            if cont:
                run.append(u)
                last_y1 = u.y1
            else:
                flush()
                run = [u]
                anchor_x0, last_y1, run_dom = u.x0, u.y1, u.dom
        else:
            flush()
            parts.append(u.md)
    flush()
    return "\n\n".join(p for p in parts if p.strip())


# --------------------------------------------------------------------------
# Vision-OCR engine (local Ollama)
# --------------------------------------------------------------------------
def should_render_page_image(page, opt: Options) -> bool:
    """Apply the --page-images policy: render this page as an image?"""
    if not opt.images or opt.page_images == "none":
        return False
    if opt.page_images == "all":
        return True
    area = (page.rect.width * page.rect.height) or 1.0  # auto: full-page image + little text
    has_full = any((fitz.Rect(i["bbox"]).width * fitz.Rect(i["bbox"]).height) / area
                   >= opt.fullpage_cover for i in page.get_image_info())
    return has_full and len(page.get_text().strip()) < opt.text_min


def ollama_generate(host, model, prompt, png_bytes, num_ctx=8192, timeout=600) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(png_bytes).decode()],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0},
    }).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")


def ocr_clean(text: str) -> str:
    # DeepSeek-OCR grounding tokens (emitted for detected image/figure regions).
    # The pipes may be ASCII (|) or fullwidth (｜).
    bar = r"[|｜]"
    text = re.sub(rf"<{bar}ref{bar}>.*?<{bar}/ref{bar}>", "", text, flags=re.S)
    text = re.sub(rf"<{bar}det{bar}>.*?<{bar}/det{bar}>", "", text, flags=re.S)
    text = re.sub(rf"<{bar}[^|｜]*?{bar}>", "", text)
    text = text.replace("�", "—")     # dropped em/en dashes -> the em dash
    text = re.sub(r"```(?:markdown)?\n(.*)\n```\s*$", r"\1", text.strip(), flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _looks_like_caps_heading(s: str) -> bool:
    """A standalone ALL-CAPS line that reads like a section title (not an
    abbreviation row like 'STR DEX CON' or a code like 'DC 15')."""
    s = s.strip()
    if re.search(r"\.{4,}", s):               # dot leaders -> a printed TOC/index entry
        return False
    m = re.match(r"^\d+\.\s+(.+)$", s)        # numbered location, e.g. "1. CAVE MOUTH"
    core = m.group(1) if m else s
    if any(c.islower() for c in core) or sum(c.isalpha() for c in core) < 3:
        return False
    if core.endswith((",", ";", ":")):
        return False
    words = core.split()
    return 1 <= len(words) <= 8 and not all(len(w.strip(".,&")) <= 3 for w in words)


def promote_ocr_headings(md: str, level: int = 2) -> str:
    """Promote plain ALL-CAPS section titles in OCR'd Markdown to headings.
    Idempotent; never touches tables, lists, fences, bold lines or existing
    headings."""
    out, in_fence = [], False
    for ln in md.split("\n"):
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
        if (not in_fence and s and not s.startswith(("#", "|", "-", "*", ">", "!"))
                and _looks_like_caps_heading(s)):
            out.append("#" * level + " " + s)
        else:
            out.append(ln)
    return "\n".join(out)


def run_ollama_engine(doc, page_indices, opt: Options, img_dir, quiet) -> list[tuple[int, str]]:
    """Render each page and transcribe it to Markdown with a local Ollama vision
    model. Per-page results are cached in <out>/pages so --resume can continue
    an interrupted run."""
    pages_dir = os.path.join(opt.out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    if not quiet:
        print(f"  engine=ollama  model={opt.model}  host={opt.ollama_host}  dpi={opt.dpi}")
    per_page: list[tuple[int, str]] = []
    n = len(page_indices)
    for i, pno in enumerate(page_indices, 1):
        pagefile = os.path.join(pages_dir, f"page-{pno + 1:03d}.md")
        if opt.resume and os.path.isfile(pagefile) and os.path.getsize(pagefile) > 0:
            with open(pagefile, encoding="utf-8") as fh:
                per_page.append((pno, fh.read().strip()))
            if not quiet:
                print(f"  [{i}/{n}] page {pno + 1}: cached")
            continue

        page = doc[pno]
        pix = page.get_pixmap(dpi=opt.dpi)
        png = pix.tobytes("png")
        t0 = time.time()
        try:
            text = ollama_generate(opt.ollama_host, opt.model, opt.ocr_prompt, png,
                                   opt.ocr_num_ctx, opt.ocr_timeout)
            err = ""
        except Exception as e:                       # noqa: BLE001 - keep the run going
            text, err = f"*(OCR failed on page {pno + 1}: {e})*", " [FAILED]"
        if not err:
            if opt.clean_ocr:
                text = ocr_clean(text)
            if opt.ocr_headings:
                text = promote_ocr_headings(text)

        parts = [text.strip()]
        if should_render_page_image(page, opt):
            path = os.path.join(img_dir, f"page-{pno + 1:03d}.png")
            os.makedirs(img_dir, exist_ok=True)
            pix.save(path)
            parts.append(f"![Page {pno + 1}]({opt.rel_img_dir}/{os.path.basename(path)})")
        md = "\n\n".join(p for p in parts if p.strip())

        with open(pagefile, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        per_page.append((pno, md))
        if not quiet:
            print(f"  [{i}/{n}] page {pno + 1}: {len(text)} chars, {time.time() - t0:.1f}s{err}")
    return per_page


def load_cached_pages(out_dir: str, opt: Options, quiet: bool) -> list[tuple[int, str]]:
    """Read previously generated <out_dir>/pages/page-NNN.md files (for --merge).
    Heading promotion is (idempotently) applied and the page file upgraded in
    place, so a merge brings older pages up to date."""
    pages_dir = os.path.join(out_dir, "pages")
    if not os.path.isdir(pages_dir):
        sys.exit(f"--merge: no per-page files found at {pages_dir}")
    pages: list[tuple[int, str]] = []
    upgraded = 0
    for fn in sorted(os.listdir(pages_dir)):
        m = re.match(r"page-(\d+)\.md$", fn)
        if not m:
            continue
        path = os.path.join(pages_dir, fn)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        text = promote_ocr_headings(raw) if opt.ocr_headings else raw
        if text != raw:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            upgraded += 1
        pages.append((int(m.group(1)) - 1, text))
    if not pages:
        sys.exit(f"--merge: {pages_dir} has no page-NNN.md files")
    if not quiet:
        extra = f" ({upgraded} upgraded with headings)" if upgraded else ""
        print(f"  merging {len(pages)} page file(s) from {pages_dir}{extra}")
    return pages


# --------------------------------------------------------------------------
# Document conversion
# --------------------------------------------------------------------------
def parse_page_range(spec: str, page_count: int) -> list[int]:
    """'1-5,8,20-' -> zero-based page indices."""
    if not spec:
        return list(range(page_count))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a else 1
            end = int(b) if b else page_count
        else:
            start = end = int(part)
        out.extend(range(start - 1, end))
    return [p for p in out if 0 <= p < page_count]


def gfm_slug(text: str) -> str:
    """GitHub-style heading anchor: lowercase, drop punctuation, spaces -> '-'.
    Matches the anchor IDs GitHub / VS Code / Obsidian auto-generate for headings."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text.strip().lower()))


class Slugger:
    """Reproduces GitHub's duplicate-anchor numbering: a repeated 'Foo' heading
    becomes 'foo', then 'foo-1', then 'foo-2', ..."""
    def __init__(self):
        self._seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = gfm_slug(text)
        n = self._seen.get(base, 0)
        self._seen[base] = n + 1
        return base if n == 0 else f"{base}-{n}"


_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def heading_anchor_index(per_page, title: str, include_contents: bool) -> dict:
    """Anchor slug of every body heading, in document order (so duplicate
    numbering matches the renderer). Returns {base_slug: [(anchor, page), ...]}."""
    slugger = Slugger()
    slugger.slug(title)                  # the document '# title' is the first heading
    if include_contents:
        slugger.slug("Contents")         # then '## Contents'
    index: dict[str, list[tuple[str, int]]] = {}
    for pno, md in per_page:
        for line in md.split("\n"):
            m = _HEADING_RE.match(line)
            if m:
                index.setdefault(gfm_slug(m.group(1)), []).append((slugger.slug(m.group(1)), pno + 1))
    return index


def _locate_bookmark_line(pages: dict, title: str, page: int):
    """Find the line that is the bookmark's section title, trying strict then
    loose strategies. `pages` maps 0-based page index -> list of lines, `page`
    is the bookmark's 1-based page. Returns (page_idx, line_idx, body|None)."""
    words = re.findall(r"\w+", title)
    if not words:
        return None
    # words in sequence, any non-word chars between, ending at clause punctuation
    # or end of line (so "Foo. bar" is a run-in heading but "Foobar" is not).
    rx = re.compile(r"^[\W_]*" + r"[\W_]+".join(re.escape(w) for w in words)
                    + r"(?=[.:;)—–-]|\s*$)", re.I)
    order = [p for p in (page - 1, page, page - 2) if p in pages]

    def usable(ln):
        t = ln.strip()
        return t and not t.startswith(("|", "![")) and not _HEADING_RE.match(ln)

    for pno in order:                                    # strict: exact / run-in prefix
        for li, ln in enumerate(pages[pno]):
            if not usable(ln):
                continue
            clean = re.sub(r"[*_`>]", "", ln).strip()
            if (m := rx.match(clean)):
                return pno, li, (clean[m.end():].strip(" .:;—–-") or None)
    for pno in order:                                    # loose: fuzzy on short lines
        best = None
        for li, ln in enumerate(pages[pno]):
            if not usable(ln):
                continue
            clean = re.sub(r"[*_`>#-]", "", ln).strip()
            if clean and len(clean) <= max(len(title) * 2, 36):
                r = difflib.SequenceMatcher(None, clean.lower(), title.lower()).ratio()
                if r >= 0.9 and (best is None or r > best[1]):
                    best = (li, r)
        if best:
            return pno, best[0], None
    return None


def apply_bookmark_headings(per_page, bookmarks, quiet: bool):
    """Use the PDF bookmarks (ground-truth section titles + pages) to promote
    the corresponding lines to headings, so the table of contents can link to
    them precisely. The promoted heading uses the bookmark's title text, so its
    anchor slug always matches the TOC entry."""
    pages = {pno: md.split("\n") for pno, md in per_page}
    existing = {pno + 1: {gfm_slug(m.group(1)) for ln in lines if (m := _HEADING_RE.match(ln))}
                for pno, lines in pages.items()}
    promoted = 0
    for entry in bookmarks:
        level, title, page = entry[0], re.sub(r"\s+", " ", entry[1]).strip(), entry[2]
        if not title:
            continue
        base = gfm_slug(title)
        if any(base in existing.get(p, ()) for p in (page, page + 1, page - 1)):
            continue                                     # already a heading near here
        hit = _locate_bookmark_line(pages, title, page)
        if not hit:
            continue
        pno, li, body = hit
        repl = [f"{'#' * min(level + 1, 6)} {title}"] + (["", body] if body else [])
        pages[pno][li:li + 1] = repl
        existing.setdefault(pno + 1, set()).add(base)
        promoted += 1
    if promoted and not quiet:
        print(f"  promoted {promoted} bookmarked section title(s) to headings")
    return [(pno, "\n".join(pages[pno])) for pno, _ in per_page]


def build_toc(doc, heading_index: dict | None = None,
              pages_present: set | None = None) -> str:
    """Bookmark table of contents with links. Each entry links to the matching
    in-document heading (nearest page wins); if no heading matches, it falls
    back to a `#page-N` anchor so every entry stays clickable."""
    toc = doc.get_toc(simple=True)
    if not toc:
        return ""
    lines = ["## Contents", ""]
    for level, title, page in toc:
        title = re.sub(r"\s+", " ", title).strip()     # bookmark titles can contain stray \r
        anchor = None
        if heading_index and (cands := heading_index.get(gfm_slug(title))):
            anchor = min(cands, key=lambda c: abs(c[1] - page))[0]
        elif pages_present and page in pages_present:
            anchor = f"page-{page}"
        link = f"[{title}](#{anchor})" if anchor else title
        lines.append(f"{'  ' * (max(level, 1) - 1)}- {link} *(p.{page})*")
    return "\n".join(lines)


def pdf_is_scan(pdf_path: str, *, cover: float = 0.7, text_min: int = 120,
                sample: int = 40) -> bool:
    """True when most pages are a single large image with little *visible* text —
    i.e. a scan (the page's content baked into an image), even when an invisible
    OCR text layer is present. A page is scan-like when some image covers
    >= `cover` of the page area AND it renders < `text_min` VISIBLE characters
    (text render mode 3 = invisible OCR is ignored — that's what makes a garbled
    scan-with-OCR look "text-rich" to naive checks). True when >= 60% of up to
    `sample` evenly-spaced pages are scan-like. Added for despereaux."""
    fz = _require_fitz()
    doc = fz.open(pdf_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            return False
        n = doc.page_count
        if n == 0:
            return False
        idxs = range(n) if n <= sample else [round(i * (n - 1) / (sample - 1)) for i in range(sample)]
        scanlike = checked = 0
        for i in idxs:
            page = doc[i]
            checked += 1
            area = (page.rect.width * page.rect.height) or 1.0
            full = any((fz.Rect(im["bbox"]).width * fz.Rect(im["bbox"]).height) / area >= cover
                       for im in page.get_image_info())
            if not full:
                continue
            visible = 0
            try:
                for sp in page.get_texttrace():
                    if sp.get("type") != 3:  # 3 = invisible (OCR) text layer
                        visible += len(sp.get("chars", ()))
                        if visible >= text_min:
                            break
            except Exception:
                visible = len(page.get_text().strip())  # fallback if texttrace unavailable
            if visible < text_min:
                scanlike += 1
        return checked > 0 and scanlike / checked >= 0.6
    finally:
        doc.close()


def convert(pdf_path: str, opt: Options, pages_spec: str = "", toc: bool = True,
            quiet: bool = False) -> str:
    _require_fitz()
    doc = fitz.open(pdf_path)
    if doc.is_encrypted and not doc.authenticate(""):
        raise RuntimeError("PDF is encrypted and needs a password")

    page_indices = parse_page_range(pages_spec, doc.page_count)
    img_dir = os.path.join(opt.out_dir, opt.rel_img_dir)
    os.makedirs(opt.out_dir, exist_ok=True)
    title = (doc.metadata or {}).get("title") or \
        os.path.splitext(os.path.basename(pdf_path))[0]

    if opt.merge:
        per_page = load_cached_pages(opt.out_dir, opt, quiet)
    elif opt.engine == "ollama":
        per_page = run_ollama_engine(doc, page_indices, opt, img_dir, quiet)
    else:
        body, furniture = analyze_document(doc, opt.margin)
        if not quiet:
            print(f"  {doc.page_count} pages | body font ~{body}pt | "
                  f"{len(furniture)} repeated header/footer line(s)")
        counters: dict = {}
        per_page = []
        for i, pno in enumerate(page_indices, 1):
            md = render_units(page_units(doc, pno, body, furniture, opt, img_dir, counters), opt)
            per_page.append((pno, md))
            if not quiet and (i % 10 == 0 or i == len(page_indices)):
                print(f"  ...{i}/{len(page_indices)} pages")

    if opt.bookmark_headings and toc and (bms := doc.get_toc(simple=True)):
        per_page = apply_bookmark_headings(per_page, bms, quiet)

    if opt.split:
        page_dir = os.path.join(opt.out_dir, "pages")
        os.makedirs(page_dir, exist_ok=True)
        for pno, md in per_page:
            with open(os.path.join(page_dir, f"page-{pno + 1:03d}.md"), "w", encoding="utf-8") as fh:
                fh.write(md + "\n")
        index = [f"# {title}", ""]
        if toc and (t := build_toc(doc)):
            index += [t, ""]
        index.append("## Pages\n")
        index += [f"- [Page {pno + 1}](pages/page-{pno + 1:03d}.md)" for pno, _ in per_page]
        out_path = os.path.join(opt.out_dir, "index.md")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(index) + "\n")
        return out_path

    pages_present = {pno + 1 for pno, _ in per_page}
    chunks = [f"# {title}", ""]
    if toc:
        hidx = heading_anchor_index(per_page, title, include_contents=True)
        if (t := build_toc(doc, hidx, pages_present)):
            chunks += [t, ""]
    for pno, md in per_page:
        # the <a id> is an in-document anchor target for the TOC's page fallback
        chunks.append(f'<!-- page {pno + 1} -->\n<a id="page-{pno + 1}"></a>')
        chunks.append(md if md.strip() else "*(no extractable text on this page)*")
    out_path = os.path.join(opt.out_dir, "output.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(chunks) + "\n")
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert a PDF into Markdown, extracting images as detected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf", help="path to the input PDF")
    p.add_argument("-o", "--out", help="output directory (default: <pdf name>_md next to the PDF)")
    p.add_argument("--pages", default="", metavar="RANGE",
                   help="pages to convert, e.g. '1-5,8,20-' (default: all)")
    eng = p.add_argument_group("engine")
    eng.add_argument("--engine", choices=("heuristic", "ollama"), default="heuristic",
                     help="heuristic: parse the PDF text layer (instant); "
                          "ollama: OCR each page image with a local Ollama vision model (most accurate)")
    eng.add_argument("--model", default="deepseek-ocr:3b", help="Ollama model for --engine ollama")
    eng.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server URL")
    eng.add_argument("--ocr-prompt", default="Convert the document to markdown.",
                     help="prompt sent with each page image")
    eng.add_argument("--resume", action="store_true",
                     help="skip pages already done (cached per-page in <out>/pages)")
    eng.add_argument("--no-ocr-headings", action="store_true",
                     help="do not promote plain ALL-CAPS section titles in OCR output to headings")
    p.add_argument("--page-images", choices=("auto", "all", "none"), default="auto",
                   help="auto: render only image-dominant pages (maps/art); all: every page; none: never")
    p.add_argument("--no-images", action="store_true", help="text only: skip all image output")
    p.add_argument("--dpi", type=int, default=150, help="resolution for rendered page images")
    p.add_argument("--fullpage-cover", type=float, default=0.7, metavar="F",
                   help="page-area fraction above which an image counts as full-page")
    p.add_argument("--text-min", type=int, default=220, metavar="N",
                   help="a page with a full-page image and fewer than N text chars is image-dominant")
    p.add_argument("--heading-ratio", type=float, default=1.3, metavar="R",
                   help="minimum font-size / body-size ratio for a size-based heading")
    p.add_argument("--no-emphasis", action="store_true", help="do not emit **bold** / *italic*")
    p.add_argument("--no-toc", action="store_true", help="do not prepend the PDF bookmarks as a table of contents")
    p.add_argument("--no-bookmark-headings", action="store_true",
                   help="do not promote bookmark section titles to headings for TOC linking")
    p.add_argument("--keep-furniture", action="store_true",
                   help="keep repeated page headers/footers and bare page numbers")
    p.add_argument("--no-reflow", action="store_true",
                   help="do not merge one-line-per-block paragraphs (keep raw block layout)")
    p.add_argument("--no-runin-headings", action="store_true",
                   help="do not promote bold/ALL-CAPS run-in text to headings")
    p.add_argument("--split", action="store_true", help="write one Markdown file per page plus index.md")
    p.add_argument("--merge", action="store_true",
                   help="skip conversion; just (re)assemble <out>/pages/*.md into one output.md")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    if not os.path.isfile(args.pdf):
        sys.exit(f"No such file: {args.pdf}")
    # Default output goes inside this project: <project>/output/<pdf name>/
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        os.path.splitext(os.path.basename(args.pdf))[0],
    )
    opt = Options(
        out_dir=out_dir,
        images=not args.no_images,
        page_images=args.page_images,
        fullpage_cover=args.fullpage_cover,
        text_min=args.text_min,
        dpi=args.dpi,
        emphasis=not args.no_emphasis,
        split=args.split,
        heading_ratio=args.heading_ratio,
        keep_furniture=args.keep_furniture,
        reflow=not args.no_reflow,
        runin_headings=not args.no_runin_headings,
        engine=args.engine,
        model=args.model,
        ollama_host=args.ollama_host,
        ocr_prompt=args.ocr_prompt,
        resume=args.resume,
        merge=args.merge,
        ocr_headings=not args.no_ocr_headings,
        bookmark_headings=not args.no_bookmark_headings,
    )
    if not args.quiet:
        print(f"Converting: {args.pdf}")
    out_path = convert(args.pdf, opt, pages_spec=args.pages, toc=not args.no_toc, quiet=args.quiet)
    if not args.quiet:
        print(f"Done -> {out_path}")
        if opt.images:
            print(f"Images -> {os.path.join(out_dir, opt.rel_img_dir)}")


if __name__ == "__main__":
    main()
