from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config import (
    EMBED_DIM,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    METADATA_PATH,
)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    """Load precomputed embeddings from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings array, got shape {embeddings.shape}"
        )

    if embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected embedding dimension {EMBED_DIM}, "
            f"got {embeddings.shape[1]}"
        )

    return embeddings.astype("float32", copy=False)


def load_metadata(
    path: Path = CHUNKS_METADATA_PATH,
) -> list[dict]:
    """Load chunk metadata from JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

    return records


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """Build a cosine-similarity FAISS index using inner product."""

    # Embeddings are already L2-normalized by embedder.py.
    index = faiss.IndexFlatIP(EMBED_DIM)

    index.add(embeddings)

    return index


def save_index(
    index: faiss.Index,
    path: Path = FAISS_INDEX_PATH,
) -> None:
    """Save FAISS index to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path))


def save_metadata(
    metadata: list[dict],
    path: Path = METADATA_PATH,
) -> None:
    """Save metadata in JSONL format."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_index() -> None:
    """Complete pipeline for building the FAISS index."""

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print(f"Embeddings shape: {embeddings.shape}")

    print("Loading metadata...")
    metadata = load_metadata()

    print(f"Metadata records: {len(metadata)}")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Mismatch between embeddings and metadata: "
            f"{len(embeddings)} embeddings vs "
            f"{len(metadata)} metadata records."
        )

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"FAISS index size: {index.ntotal}")

    print("Saving FAISS index...")
    save_index(index)

    print("Saving metadata...")
    save_metadata(metadata)

    print("\nDone!")
    print(f"FAISS index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()