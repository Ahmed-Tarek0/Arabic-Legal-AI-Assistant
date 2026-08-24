# Step 1: USER Upload PDF -> PyMuPDF -> Extract Text


from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (modern import name; `fitz` alias kept for readability below)


@dataclass
class PageText:
    page_number: int   # 1-indexed, human friendly
    text: str


def _normalize_arabic(text: str) -> str:
    """Light normalization: unify unicode forms, strip control chars,
    collapse whitespace. We deliberately do NOT strip diacritics/tashkeel
    or normalize alef/hamza variants here, since legal text precision
    matters — normalization for search can be applied later at the
    embedding/query stage if needed, not destructively at extraction time.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # remove PDF artefacts: soft hyphens, zero-width chars, form feed
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\x0c", "")
    # collapse runs of whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_page_text(page: "fitz.Page") -> str:
    """Extract a single page's text using block-sorted ordering."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    if not blocks:
        return ""

    # Sort blocks by vertical position first (reading top -> bottom),
    # then by horizontal position reversed (right -> left) for RTL layout.
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), -b[0]))

    parts = [b[4] for b in blocks if b[4] and b[4].strip()]
    return "\n".join(parts)


def extract_pdf_text(pdf_path: str | Path) -> list[PageText]:
    """Extract text from every page of a PDF.

    Returns a list of PageText(page_number, text), one entry per page,
    already normalized. Empty/near-empty pages (e.g. scanned images with
    no text layer) are still returned so callers can detect and flag them
    for OCR — they are not silently dropped here.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            raw = _extract_page_text(page)
            clean = _normalize_arabic(raw)
            pages.append(PageText(page_number=i, text=clean))
    return pages


def extract_pdf_text_flagging_empty(pdf_path: str | Path) -> tuple[list[PageText], list[int]]:
    """Same as extract_pdf_text, but also returns page numbers that came back
    empty (likely scanned/image-only pages needing OCR — out of scope for
    this stage, but useful to surface to the user).
    """
    pages = extract_pdf_text(pdf_path)
    empty_pages = [p.page_number for p in pages if len(p.text) < 5]
    return pages, empty_pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    pages, empty = extract_pdf_text_flagging_empty(sys.argv[1])
    print(f"Extracted {len(pages)} pages.")
    if empty:
        print(f"Warning: pages with little/no extractable text (may need OCR): {empty}")
    for p in pages[:2]:
        print(f"\n--- Page {p.page_number} preview ---")
        print(p.text[:400])