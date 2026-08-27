# Step 3: Chunking -> BGE-M3 -> Embeddings

from __future__ import annotations

import numpy as np

from config import EMBED_BATCH_SIZE, EMBED_MAX_LENGTH, EMBED_MODEL_NAME, USE_FP16

_model = None


def get_embedder():
    """Lazily load and cache the BGE-M3 model (load once)."""
    global _model
    if _model is None:
        try:
            import streamlit as st
            # Use Streamlit resource cache if running inside Streamlit
            @st.cache_resource(show_spinner=False)
            def _load_st_embedder():
                from FlagEmbedding import BGEM3FlagModel
                return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=False)
            _model = _load_st_embedder()
        except Exception:
            from FlagEmbedding import BGEM3FlagModel
            _model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Embed a list of texts (chunks or a single query) into dense vectors.

    Returns an (N, EMBED_DIM) float32 numpy array, L2-normalized so that
    inner product search in FAISS is equivalent to cosine similarity.
    """
    if not texts:
        return np.zeros((0, 1024), dtype="float32")

    model = get_embedder()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=EMBED_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = np.asarray(output["dense_vecs"], dtype="float32")

    # normalize to unit length for cosine-via-inner-product search
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    dense = dense / norms
    return dense


def embed_query(query: str) -> np.ndarray:
    """Convenience wrapper: embed a single question, return shape (1, DIM)."""
    return embed_texts([query], batch_size=1)


if __name__ == "__main__":
    vecs = embed_texts(["ما هي شروط الفسخ في عقد الإيجار؟", "المادة الأولى: يُقصد بالعقد..."])
    print("Embedding shape:", vecs.shape)
    print("First vector norm (should be ~1.0):", float(np.linalg.norm(vecs[0])))