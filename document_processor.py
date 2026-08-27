"""
Dynamic Document Processor for Arabic Legal Contracts.
Supports PDF, DOCX, and TXT files, whether uploaded via Streamlit (bytes/stream)
or read from local disk.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, List, Tuple, Union

import docx
import faiss
import numpy as np
import pymupdf as fitz

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBED_DIM,
    MIN_CHUNK_WORDS,
)
from embedder import embed_texts
from pdf_extractor import PageText, _extract_page_text, _normalize_arabic
from retriever import LegalRetriever
from text_chunker import Chunk, chunk_document


def extract_text_from_upload(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "document.pdf",
) -> List[PageText]:
    """
    Extracts text page by page from an uploaded file or local path.
    Supports PDF, DOCX, and TXT.
    """
    suffix = Path(filename).suffix.lower()
    pages: List[PageText] = []

    # Read bytes if needed
    if isinstance(file_source, (str, Path)):
        source_path = Path(file_source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        with open(source_path, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:  # BinaryIO / Streamlit UploadedFile
        file_bytes = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(0)

    if suffix == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                raw = _extract_page_text(page)
                clean = _normalize_arabic(raw)
                pages.append(PageText(page_number=i, text=clean))

    elif suffix == ".docx":
        doc_stream = io.BytesIO(file_bytes)
        doc = docx.Document(doc_stream)
        
        # Combine paragraphs; divide into simulated pages if large
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        clean = _normalize_arabic(full_text)
        
        # If very long, split into pages of ~350 words, else single page
        words = clean.split()
        page_size = 350
        if len(words) > page_size:
            page_num = 1
            for i in range(0, len(words), page_size):
                page_slice = " ".join(words[i:i + page_size])
                pages.append(PageText(page_number=page_num, text=page_slice))
                page_num += 1
        else:
            pages.append(PageText(page_number=1, text=clean))

    elif suffix in (".txt", ".text", ".md"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("cp1256")  # Arabic Windows encoding
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore")

        clean = _normalize_arabic(text)
        words = clean.split()
        page_size = 350
        if len(words) > page_size:
            page_num = 1
            for i in range(0, len(words), page_size):
                page_slice = " ".join(words[i:i + page_size])
                pages.append(PageText(page_number=page_num, text=page_slice))
                page_num += 1
        else:
            pages.append(PageText(page_number=1, text=clean))

    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported formats: .pdf, .docx, .txt")

    # Filter out completely empty pages if total pages > 1
    non_empty = [p for p in pages if p.text.strip()]
    return non_empty if non_empty else pages


def process_contract_dynamically(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "contract.pdf",
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> Tuple[LegalRetriever, List[Chunk], List[PageText]]:
    """
    Full dynamic pipeline:
    1. Extract text from uploaded contract
    2. Chunk document preserving Arabic legal structure
    3. Generate BGE-M3 embeddings
    4. Build FAISS index in memory
    5. Return an active LegalRetriever, all chunks, and raw pages.
    """
    doc_id = Path(filename).stem
    pages = extract_text_from_upload(file_source, filename=filename)

    if not pages or all(len(p.text.strip()) == 0 for p in pages):
        raise ValueError(
            f"لم يتم العثور على نصوص قابلة للقراءة في المستند '{filename}'. "
            "تأكد أن الملف يحتوي على نصوص وليس صوراً ممسوحة ضوئياً."
        )

    # Chunking
    chunks = chunk_document(
        pages=pages,
        doc_id=doc_id,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
        min_chunk_words=min_chunk_words,
    )

    if not chunks:
        # If document is short but has text, make 1 chunk from all text
        combined_text = "\n\n".join([p.text for p in pages if p.text.strip()])
        if combined_text.strip():
            chunks = [
                Chunk(
                    chunk_id=f"{doc_id}_c0001",
                    text=combined_text.strip(),
                    source_pages=[p.page_number for p in pages if p.text.strip()],
                    doc_id=doc_id,
                )
            ]
        else:
            raise ValueError("النصوص المستخرجة غير كافية لإنشاء مقاطع للبحث.")

    # Embeddings
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    # FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Metadata
    metadata: List[dict] = []
    for row_id, chunk in enumerate(chunks):
        metadata.append({"row_id": row_id, **asdict(chunk)})

    retriever = LegalRetriever.from_index_and_metadata(index=index, metadata=metadata)
    return retriever, chunks, pages
