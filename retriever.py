from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config import (
    EMBED_DIM,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K,
)
from embedder import embed_query


class LegalRetriever:
    """
    FAISS-based retriever for the legal RAG pipeline.

    Flow:
        User query
            ↓
        BGE-M3 embedding
            ↓
        FAISS similarity search
            ↓
        Metadata lookup
            ↓
        Top-K legal chunks
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = self._load_index()
        self.metadata = self._load_metadata()

        self._validate_index_and_metadata()

    def _load_index(self) -> faiss.Index:
        """Load the FAISS index from disk."""

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                "Run build_index.py first."
            )

        index = faiss.read_index(str(self.index_path))

        if index.d != EMBED_DIM:
            raise ValueError(
                f"FAISS index dimension mismatch. "
                f"Expected {EMBED_DIM}, got {index.d}."
            )

        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")

        return index

    def _load_metadata(self) -> list[dict]:
        """Load row-aligned chunk metadata from JSONL."""

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                "Run build_index.py first."
            )

        metadata: list[dict] = []

        with self.metadata_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} "
                        f"in {self.metadata_path}"
                    ) from exc

                metadata.append(record)

        if not metadata:
            raise ValueError("Metadata file is empty.")

        return metadata

    def _validate_index_and_metadata(self) -> None:
        """
        Make sure FAISS vector positions correspond to metadata rows.

        FAISS result index 0 must correspond to metadata[0],
        result index 1 to metadata[1], etc.
        """

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata size mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata records."
            )

        for row_id, record in enumerate(self.metadata):
            stored_row_id = record.get("row_id")

            if stored_row_id is not None and stored_row_id != row_id:
                raise ValueError(
                    f"Metadata row alignment error at position {row_id}: "
                    f"expected row_id={row_id}, "
                    f"found row_id={stored_row_id}."
                )

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most relevant legal chunks for a query.

        Returns a list of dictionaries containing:
            - rank
            - score
            - row_id
            - chunk metadata
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # We cannot retrieve more vectors than exist in the index.
        k = min(top_k, self.index.ntotal)

        # BGE-M3 returns a normalized vector with shape (1, 1024).
        query_embedding = embed_query(query)

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_embedding.shape != (1, EMBED_DIM):
            raise ValueError(
                f"Unexpected query embedding shape: "
                f"{query_embedding.shape}. "
                f"Expected (1, {EMBED_DIM})."
            )

        # Because both document and query embeddings are L2-normalized,
        # IndexFlatIP gives cosine similarity.
        scores, indices = self.index.search(query_embedding, k)

        results: list[dict] = []

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            # FAISS can return -1 when no valid result exists.
            if index_id < 0:
                continue

            record = dict(self.metadata[index_id])

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    **record,
                }
            )

        return results


def retrieve(
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Convenience function.

    Example:
        results = retrieve("ما شروط فسخ العقد؟")
    """

    retriever = LegalRetriever()

    return retriever.retrieve(
        query=query,
        top_k=top_k,
    )


def print_results(results: list[dict]) -> None:
    """Pretty-print retrieval results for development/testing."""

    if not results:
        print("No results found.")
        return

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")

        print(f"Chunk ID: {result.get('chunk_id', 'N/A')}")
        print(f"Document ID: {result.get('doc_id', 'N/A')}")

        pages = result.get("source_pages", [])
        print(f"Pages: {pages if pages else 'N/A'}")

        categories = result.get("categories", [])
        if categories:
            print(f"Categories: {categories}")

        text = result.get("text", "")
        print("\nText:")
        print(text)

        print("-" * 70)


def main() -> None:
    """Interactive CLI for testing the retriever."""

    print("Loading Legal Retriever...")

    retriever = LegalRetriever()

    print(f"FAISS vectors: {retriever.index.ntotal}")
    print(f"Metadata records: {len(retriever.metadata)}")

    print("\nRetriever is ready.")
    print("Type a legal question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            results = retriever.retrieve(query)
            print_results(results)

        except Exception as exc:
            print(f"\nRetrieval error: {exc}")


if __name__ == "__main__":
    main()