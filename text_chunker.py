# Step 2: Extract Text -> Chunking


from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, MIN_CHUNK_WORDS
from pdf_extractor import PageText   

# Arabic + Latin sentence-ending punctuation
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?؟۔])\s+")

# Legal documents are often structured in numbered articles (المادة 1 / الفصل 2).
# We try to keep article boundaries as strong split hints when present.
_ARTICLE_HINT_RE = re.compile(r"(?=(?:المادة|الفصل|الباب|البند)\s*[\d٠-٩]+)")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_pages: list[int] = field(default_factory=list)
    doc_id: str = ""
    categories: list[str] = field(default_factory=list)  # populated by non-PDF sources (e.g. HF dataset)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # First split on strong article/section boundaries, then sentences within.
    segments = _ARTICLE_HINT_RE.split(text)
    sentences: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(seg) if s.strip())
    return sentences


def chunk_document(
    pages: list[PageText],
    doc_id: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """Build overlapping chunks across an entire document.

    We concatenate sentences (tagged with their source page) and pack them
    into ~chunk_size_words windows, sliding back by overlap_words so context
    isn't lost at chunk boundaries. This is sentence-safe: we never cut a
    sentence in half.
    """
    # Flatten to (sentence, page_number) pairs, preserving reading order.
    tagged_sentences: list[tuple[str, int]] = []
    for page in pages:
        for sent in _split_sentences(page.text):
            tagged_sentences.append((sent, page.page_number))

    chunks: list[Chunk] = []
    current_words: list[str] = []
    current_pages: set[int] = set()
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx, current_words, current_pages
        if len(current_words) >= min_chunk_words:
            chunk_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_c{chunk_idx:04d}",
                    text=" ".join(current_words),
                    source_pages=sorted(current_pages),
                    doc_id=doc_id,
                )
            )

    i = 0
    while i < len(tagged_sentences):
        sent, page_no = tagged_sentences[i]
        sent_words = sent.split()

        if len(current_words) + len(sent_words) > chunk_size_words and current_words:
            flush()
            # start new window with overlap: keep the tail of previous window
            overlap = current_words[-overlap_words:] if overlap_words else []
            current_words = list(overlap)
            # pages for the retained overlap are unknown per-word, so we just
            # keep the current page set — slight over-inclusion is fine for evidence.
            current_pages = {page_no}
        else:
            current_pages.add(page_no)

        current_words.extend(sent_words)
        i += 1

    flush()  # final chunk

    return chunks


if __name__ == "__main__":
    import sys

    from pdf_extractor import extract_pdf_text

    if len(sys.argv) != 2:
        print("Usage: python text_chunker.py <path_to_pdf>")
        sys.exit(1)

    pages = extract_pdf_text(sys.argv[1])
    chunks = chunk_document(pages, doc_id="demo_doc")
    print(f"Produced {len(chunks)} chunks from {len(pages)} pages.")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}] pages={c.source_pages} len_words={len(c.text.split())}")
        print(c.text[:300])