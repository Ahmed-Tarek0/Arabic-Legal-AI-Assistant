# ⚖️ Arabic Legal AI Assistant

An intelligent generative AI and advanced semantic search (**RAG**) system designed specifically for analyzing and querying Arabic contracts and legal documents.

The system allows users to upload contracts in **PDF, DOCX, or TXT** format, instantly build an in-memory vector index, and ask legal questions while receiving accurate, evidence-based answers supported by relevant contract clauses and page numbers.

---

## ✨ Key Features

* **Dynamic Contract Ingestion**: Supports real-time extraction and processing of PDF, DOCX, and TXT files.
* **Real-time In-Memory RAG**: Uses `BAAI/bge-m3` embeddings and `FAISS` for accurate semantic chunking and indexing of contract clauses.
* **Interactive Streamlit UI**: Modern Arabic RTL interface with an interactive chat, suggested questions, and an intuitive user experience.
* **Evidence & Citations**: Displays relevant clause numbers, page numbers, and similarity scores supporting each answer.
* **Full Contract Risk Audit**: Automatically extracts contract parties, financial obligations, penalty clauses, and liability limitations with a single click.
* **Ready-to-Use Sample Contracts**: Includes sample Arabic contracts such as employment contracts, rental agreements, and Non-Disclosure Agreements (NDAs).
* **Multi-Backend LLM Support**: Supports Google Gemini API, OpenAI/Groq, and local Qwen 2.5 models, with an intelligent fallback mechanism for answer generation.

---

## 🚀 How to Run

### 1. Install Requirements

Activate your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

Start the web interface using:

```bash
streamlit run app.py
```

Or through your Python environment:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💻 CLI Mode

The system can also be tested directly from the terminal.

### Query a Specific Contract

```bash
python main.py --file path/to/your_contract.pdf
```

### Use the Previously Stored Index

```bash
python main.py
```

---

## 🏗️ Project Structure

```text
├── app.py                   # Interactive Streamlit user interface
├── document_processor.py    # Dynamic document processing, chunking, embedding, and indexing
├── retriever.py             # Semantic retrieval engine using FAISS + BGE-M3
├── generator.py             # LLM answer generation (Gemini / OpenAI / Groq / Qwen / Fallback)
├── augmentor.py             # Legal prompt construction and context structuring
├── rag_pipeline.py          # Unified RAG pipeline
├── sample_contracts.py      # Ready-to-use Arabic sample contracts
├── pdf_extractor.py         # PDF text extraction using PyMuPDF
├── text_chunker.py          # Contract-aware text chunking
├── embedder.py              # Embedding generation using BGE-M3
├── config.py                # System configuration and model settings
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🔄 RAG Pipeline

```text
Contract Upload
      ↓
Document Extraction
      ↓
Text Cleaning & Chunking
      ↓
BGE-M3 Embeddings
      ↓
FAISS Vector Index
      ↓
User Legal Query
      ↓
Semantic Retrieval
      ↓
Context Augmentation
      ↓
LLM Generation
      ↓
Evidence-Based Legal Answer
```

---
