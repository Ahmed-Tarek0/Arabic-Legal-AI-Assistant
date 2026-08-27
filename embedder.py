"""
Ultra-Fast & Lightweight Multilingual Embedder for Arabic Legal Contracts.
Uses SentenceTransformer (cached in RAM once, <0.05s query embedding, ~120MB RAM).
"""

from __future__ import annotations

import numpy as np

from config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL_NAME

_model = None


def get_embedder():
    """Lazily load and cache the SentenceTransformer model (once)."""
    global _model
    if _model is None:
        try:
            import streamlit as st
            @st.cache_resource(show_spinner=False)
            def _load_st_model():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(EMBED_MODEL_NAME)
            _model = _load_st_model()
        except Exception:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Embed a list of texts (chunks or query) into dense L2-normalized float32 vectors."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query into a (1, DIM) float32 vector in milliseconds."""
    return embed_texts([query], batch_size=1)


if __name__ == "__main__":
    vecs = embed_texts(["ما هي شروط الفسخ في عقد الإيجار؟", "المادة الأولى: يُقصد بالعقد..."])
    print("Embedding shape:", vecs.shape)
    print("Vector norm (should be ~1.0):", float(np.linalg.norm(vecs[0])))