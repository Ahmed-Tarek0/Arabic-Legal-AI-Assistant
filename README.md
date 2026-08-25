# Arabic-Legal-AI-Assistant

## Features

* Arabic legal document understanding
* Semantic document retrieval
* Context augmentation
* LLM-based answer generation
* Source/context-aware responses

## System Architecture

1. PDFs/Documents
2. Text Extraction
3. Chunking
4. Embeddings
5. Vector Database
6. User Question
7. Query Embedding
8. Retrieval
9. Context Augmentation
10. LLM Generation
11. Arabic Legal Answer

---

## Project Structure

* `config.py`: System configuration and path settings.
* `pdf_extractor.py`: Extracts text from legal PDF documents.
* `text_chunker.py`: Splits extracted text into smaller chunks.
* `embedder.py`: Handles vector embedding generation.
* `build_embeddings.py`: Processes documents into embeddings.
* `build_index.py`: Builds and saves the FAISS vector index.
* `retriever.py`: Retrieves relevant context based on user queries.
* `generator.py`: Connects retrieved context with LLM for final answer generation (Augmentation & Generation).

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
