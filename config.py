"""
Central configuration for the Arabic Legal RAG pipeline.
Keep every tunable value here so the rest of the code stays clean.
"""

from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                 # put source PDFs here
INDEX_DIR = BASE_DIR / "index_store"          # FAISS index + metadata live here

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "legal_index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

# Pre-FAISS embeddings stage: PDF -> extract -> chunk -> embed, saved to disk
# and stopping BEFORE vector indexing. Kept separate from index_store/ so the
# FAISS step (build_index.py) can be run later, independently, on this output.
EMBEDDINGS_DIR = BASE_DIR / "embeddings_store"
EMBEDDINGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
CHUNKS_METADATA_PATH = EMBEDDINGS_DIR / "chunks_metadata.jsonl"

# ---------- Chunking ----------
CHUNK_SIZE_WORDS = 220        # target chunk size, in words (Arabic legal text ~ dense)
CHUNK_OVERLAP_WORDS = 40      # sliding-window overlap between consecutive chunks
MIN_CHUNK_WORDS = 20          # drop chunks smaller than this (headers, page numbers, noise)

# ---------- Embedding ----------
# BGE-M3: strong multilingual embedding model with native Arabic support.
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 12
EMBED_MAX_LENGTH = 1024        # BGE-M3 supports up to 8192, 1024 is plenty for our chunk size
USE_FP16 = True                 # set False if running on CPU only

# ---------- FAISS ----------
EMBED_DIM = 1024               # BGE-M3 dense vector dimension
# We normalize embeddings and use inner product => equivalent to cosine similarity.
TOP_K = 5

# ---------- Evaluation (Phase 2) ----------
# Ground-truth corpus used ONLY to build and score the evaluation set.
# This is separate from index_store/, which holds whatever PDF(s) a real
# user uploads at query time — the eval corpus never feeds production RAG.
HF_DATASET_ID = "dataflare/egypt-legal-corpus"

EVAL_DIR = BASE_DIR / "eval_store"
EVAL_DIR.mkdir(exist_ok=True)

EVAL_INDEX_PATH = EVAL_DIR / "eval_index.faiss"
EVAL_METADATA_PATH = EVAL_DIR / "eval_metadata.jsonl"
EVAL_SET_PATH = EVAL_DIR / "eval_questions.jsonl"

EVAL_MIN_ARTICLE_WORDS = 15     # drop article fragments shorter than this
EVAL_NUM_QUESTIONS = 100        # matches the "100 Legal Questions" plan
EVAL_RANDOM_SEED = 42

# Model used to synthesize questions from ground-truth articles (LLM mode).
# Override with env var ANTHROPIC_EVAL_MODEL if you want a different one.
EVAL_LLM_MODEL = "claude-sonnet-5"

EVAL_TOP_K_VALUES = [1, 3, 5]   # Recall@K / Precision@K reported at these cutoffs