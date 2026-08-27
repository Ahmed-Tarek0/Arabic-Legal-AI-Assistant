"""
Central configuration for the Arabic Legal RAG pipeline.
Keep every tunable value here so the rest of the code stays clean.
"""

import os
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
EMBEDDINGS_DIR = BASE_DIR / "embeddings_store"
EMBEDDINGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
CHUNKS_METADATA_PATH = EMBEDDINGS_DIR / "chunks_metadata.jsonl"

# ---------- Chunking ----------
CHUNK_SIZE_WORDS = 220        # target chunk size, in words (Arabic legal text ~ dense)
CHUNK_OVERLAP_WORDS = 40      # sliding-window overlap between consecutive chunks
MIN_CHUNK_WORDS = 20          # drop chunks smaller than this (headers, page numbers, noise)

# ---------- Embedding (Fast & Lightweight Multilingual Model) ----------
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 512
USE_FP16 = False

# ---------- FAISS ----------
EMBED_DIM = 384               # MiniLM dense vector dimension
TOP_K = 3

# ---------- Evaluation (Phase 2) ----------
HF_DATASET_ID = "dataflare/egypt-legal-corpus"

EVAL_DIR = BASE_DIR / "eval_store"
EVAL_DIR.mkdir(exist_ok=True)

EVAL_INDEX_PATH = EVAL_DIR / "eval_index.faiss"
EVAL_METADATA_PATH = EVAL_DIR / "eval_metadata.jsonl"
EVAL_SET_PATH = EVAL_DIR / "eval_questions.jsonl"

EVAL_MIN_ARTICLE_WORDS = 15
EVAL_NUM_QUESTIONS = 100
EVAL_RANDOM_SEED = 42

EVAL_LLM_MODEL = "claude-sonnet-5"
EVAL_TOP_K_VALUES = [1, 3, 5]

# ---------- LLM Generation & Local .env Loader ----------
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"