# Ingestion pipeline up to (but NOT including) FAISS:
#  PDF --PyMuPDF--> raw text --chunking--> chunks --BGE-M3--> embeddings --> disk


from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CHUNKS_METADATA_PATH, DATA_DIR, EMBEDDINGS_PATH
from embedder import embed_texts
from pdf_extractor import extract_pdf_text_flagging_empty
from text_chunker import Chunk, chunk_document


def ingest_pdfs(pdf_paths: list[Path]) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for pdf_path in tqdm(pdf_paths, desc="Extracting + chunking PDFs"):
        doc_id = pdf_path.stem
        pages, empty_pages = extract_pdf_text_flagging_empty(pdf_path)

        if empty_pages:
            print(
                f"  [warn] '{pdf_path.name}': pages {empty_pages} had little/no "
                f"extractable text (likely scanned images — OCR is out of scope here)."
            )

        doc_chunks = chunk_document(pages, doc_id=doc_id)
        print(f"  '{pdf_path.name}': {len(pages)} pages -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    return all_chunks


def save_embeddings(embeddings: np.ndarray, chunks: list[Chunk],
                     embeddings_path: Path = EMBEDDINGS_PATH,
                     metadata_path: Path = CHUNKS_METADATA_PATH) -> None:
    """Save embeddings + row-aligned chunk metadata to disk (no FAISS yet)."""
    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for row_id, chunk in enumerate(chunks):
            record = {"row_id": row_id, **asdict(chunk)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings (PDF -> extract -> chunk -> BGE-M3), stopping before FAISS."
    )
    parser.add_argument("pdfs", nargs="*", help="Path(s) to PDF file(s).")
    parser.add_argument("--dir", type=str, default=None, help="Process every .pdf in this directory.")
    args = parser.parse_args()

    if args.dir:
        pdf_paths = sorted(Path(args.dir).glob("*.pdf"))
    elif args.pdfs:
        pdf_paths = [Path(p) for p in args.pdfs]
    else:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found. Pass file paths, use --dir, or drop PDFs into {DATA_DIR}/")
        sys.exit(1)

    for p in pdf_paths:
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

    print(f"Ingesting {len(pdf_paths)} PDF(s)...")
    chunks = ingest_pdfs(pdf_paths)

    if not chunks:
        print("No chunks were produced — check that the PDFs contain extractable text.")
        sys.exit(1)

    print(f"\nTotal chunks: {len(chunks)}")
    print("Embedding chunks with BGE-M3 (this loads the model on first run)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    save_embeddings(embeddings, chunks)

    print(f"\nDone. Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) to:")
    print(f"  {EMBEDDINGS_PATH}")
    print(f"  {CHUNKS_METADATA_PATH}")
    print("\nFAISS indexing not run yet — that's the next stage (build_index.py).")


if __name__ == "__main__":
    main()