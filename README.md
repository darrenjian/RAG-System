# DocBot: RAG Agent

**DocBot** is a Retrieval-Augmented Generation (RAG) system for intelligent PDF document processing and question answering. Built with FastAPI and Mistral AI, it combines semantic search, keyword matching, and advanced LLM capabilities to provide accurate, cited answers from your documents.

---

## Table of Contents
1. [General Overview](#general-overview)
2. [System Architecture](#system-architecture)
3. [Setup and Usage](#setup-and-usage)
4. [Future Improvements](#future-improvements)

---

## General Overview

**Built by**: Darren Jian || **Contact**: darrenjian28@gmail.com

### Codebase Structure

```
RAG-System/
├── app/
│   ├── main.py              # FastAPI endpoints and server configuration
│   ├── rag_pipeline.py      # Orchestrates ingestion and query workflows
│   ├── vector_db.py         # Custom vector database with cosine similarity
│   ├── mistral_client.py    # Mistral API client (OCR, embeddings, LLM)
│   ├── search.py            # Hybrid search (semantic + BM25 keyword)
│   ├── chunking.py          # Sentence-based text chunking with NLTK
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # Configuration management
├── ui/
│   └── index.html           # Chat interface for document Q&A
├── vectordb/                # Persisted vector database (created at runtime)
├── uploads/                 # Uploaded PDFs (created at runtime)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── DESIGN_DECISIONS.md      # Detailed architecture documentation
```

### Core Features

- **PDF Ingestion**: Upload PDFs with text extraction via PyPDF2
- **Hybrid Search**: Combines semantic search (embeddings) + keyword search (BM25)
- **LLM Generation**: Context-aware answer generation with citations
- **Intent Detection**: Classifies queries to optimize API usage
- **Query Transformation**: Enhances queries for better retrieval
- **Hallucination Detection**: Verifies answers against source documents
- **Confidence Filtering**: Only answers when evidence threshold is met
- **Rich Formatting**: Markdown-rendered answers with bold, lists, tables

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | Async API server with auto-docs |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| **LLM/Embeddings** | [Mistral AI](https://mistral.ai/) | Text extraction, embeddings, generation |
| **Vector Database** | Custom ([NumPy](https://numpy.org/)) | Cosine similarity search |
| **Keyword Search** | BM25 (custom) | Exact term matching |
| **Text Processing** | [NLTK](https://www.nltk.org/) | Sentence tokenization |
| **PDF Extraction** | [PyPDF2](https://pypdf2.readthedocs.io/) | Digital PDF text extraction |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) | Type-safe models |
| **Frontend** | Vanilla HTML/CSS/JS | Simple chat interface |

### Design Assumptions

This system was built with specific constraints and use cases in mind:

**Dataset Assumptions:**
- **Digital PDFs**: Documents with extractable text layers (not scanned/image-based)
- **English Language**: NLTK sentence tokenization optimized for English
- **Document Size**: Typical range 10-500 pages (~50-25,000 chunks)
- **Content Type**: Text-heavy documents (reports, papers, articles) rather than form-heavy or image-heavy PDFs

**Usage Assumptions:**
- **Scale**: Small team or personal use (<10 concurrent users)
- **Query Pattern**: Asynchronous, research-oriented queries (not real-time chat)
- **Workload**: Read-heavy (many queries per ingested document)
- **Deployment**: Single-machine deployment (no distributed system needed)

**Technical Bounds:**
- **Vector Count**: <100k vectors total (~500 documents of 200 chunks each)
- **Memory**: Entire vector database fits in RAM (typically <2GB)
- **API Limits**: Respects Mistral API rate limits (batching in groups of 100)
- **Concurrent Users**: Designed for <100 simultaneous queries

**Known Constraints:**
- **No GPU**: Runs on CPU-only machines
- **Stateless**: No conversation history across sessions
- **Single Language**: No multilingual support
- **Storage**: Local filesystem only (no cloud storage integration)

These assumptions enable simplicity and educational value. For production systems with different requirements, see [Future Improvements](#future-improvements).

---

## System Architecture

### RAG Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM PIPELINE                          │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  INGESTION PHASE                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User uploads PDF                                                   │
│      ↓                                                              │
│  [1] PDF Text Extraction (PyPDF2)                                  │
│      → Extract text page-by-page from digital PDFs                 │
│      ↓                                                              │
│  [2] Text Chunking (Sentence-based)                                │
│      → Split into sentences using NLTK                             │
│      → Preserves semantic boundaries                               │
│      ↓                                                              │
│  [3] Embedding Generation (Mistral API)                            │
│      → Convert chunks to 1024-dimensional vectors                  │
│      → Batched in groups of 100 for large documents               │
│      ↓                                                              │
│  [4] Vector Database Storage                                        │
│      → Store vectors with metadata                                 │
│      → Index for BM25 keyword search                               │
│      ↓                                                              │
│  [5] Persist to Disk                                                │
│      → Save to vectordb/ for future queries                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  QUERY PHASE                                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User asks question                                                 │
│      ↓                                                              │
│  [1] Intent Detection (Mistral LLM)                                │
│      → Classify: greeting, chitchat, factual, retrieval_needed     │
│      → Skip retrieval for greetings (faster, cheaper)              │
│      ↓                                                              │
│  [2] Query Transformation (Mistral LLM)                            │
│      → Expand abbreviations, add synonyms                          │
│      → Example: "ML" → "machine learning"                          │
│      ↓                                                              │
│  [3] Query Embedding (Mistral API)                                 │
│      → Convert enhanced query to vector                            │
│      ↓                                                              │
│  [4] Hybrid Search                                                  │
│      ├─ Semantic Search (Cosine Similarity)                        │
│      │   → Find top-10 semantically similar chunks                 │
│      │   → Score: 0.0 to 1.0                                       │
│      │                                                              │
│      ├─ Keyword Search (BM25)                                       │
│      │   → Find top-10 by keyword matching                         │
│      │   → TF-IDF based scoring                                    │
│      │                                                              │
│      └─ Combine & Re-rank                                          │
│          → Hybrid score = 0.7×semantic + 0.3×keyword               │
│          → Return top-5 results                                    │
│      ↓                                                              │
│  [5] Confidence Check                                               │
│      → If top score < 0.50: "I don't have enough evidence"         │
│      ↓                                                              │
│  [6] Answer Generation (Mistral LLM)                               │
│      → Provide retrieved context + strict constraints              │
│      → Generate answer with source citations [1][2]                │
│      ↓                                                              │
│  [7] Hallucination Detection (Optional)                            │
│      → Verify answer is supported by context                       │
│      → Override if unsupported claims detected                     │
│      ↓                                                              │
│  [8] Return Response                                                │
│      → Answer + Citations + Confidence + Processing Time           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works

#### Ingestion Phase

1. **PDF Text Extraction**: PyPDF2 extracts text from digital PDFs page-by-page
2. **Sentence-Based Chunking**: NLTK splits text into sentences, preserving semantic boundaries for precise retrieval
3. **Embedding Generation**: Mistral API converts each sentence to a 1024-dimensional vector
4. **Dual Indexing**: Stores vectors for semantic search + builds BM25 index for keyword search
5. **Persistence**: Saves to disk for fast querying

#### Query Phase

1. **Intent Detection**: LLM classifies query type to skip retrieval for greetings (saves API calls)
2. **Query Enhancement**: LLM expands abbreviations and adds synonyms for better matching
3. **Hybrid Search**: Combines semantic similarity (finds conceptually related content) with BM25 keyword matching (finds exact terms)
4. **Re-ranking**: Weighted combination (70% semantic, 30% keyword) produces final top-5 results
5. **Confidence Filtering**: Only generates answers when similarity threshold (0.5) is met
6. **Answer Generation**: LLM generates grounded response with citations using retrieved context
7. **Verification**: Optional hallucination check ensures answer is supported by source documents

### Key Design Strategies

#### Text Chunking
- **Sentence-based** (not fixed-size) to preserve semantic coherence
- Each sentence becomes a searchable unit with exact citations
- Filters out very short sentences (< 10 chars) to reduce noise

#### Hybrid Search
- **Semantic search** captures conceptual similarity (good for paraphrases)
- **Keyword search** finds exact term matches (good for specific codes/names)
- **Combination** leverages strengths of both approaches

#### Custom Vector Database
- Built with NumPy for educational purposes
- No external dependencies (Pinecone, Weaviate, etc.)
- Sufficient for small-to-medium scale (<100k vectors)

#### Batching for Large Documents
- **Batch size of 100 chunks** to prevent API timeouts and rate limits
- Enables reliable processing of large documents (1000+ chunks)
- Provides progress feedback during ingestion
- Prevents memory issues and network failures

### File Responsibilities

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app with endpoints: `/ingest`, `/query`, `/health`, `/stats`, `/reset` |
| `rag_pipeline.py` | Orchestrates ingestion workflow and query processing pipeline |
| `vector_db.py` | In-memory vector database with cosine similarity search |
| `mistral_client.py` | Wraps Mistral API for OCR, embeddings, LLM generation, intent classification |
| `search.py` | Implements BM25 keyword search and hybrid search re-ranking |
| `chunking.py` | Sentence-based text chunking using NLTK tokenizer |
| `models.py` | Pydantic models for request/response validation |
| `config.py` | Settings loaded from `.env` (API keys) and hardcoded configs |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve chat UI |
| `/health` | GET | System health and statistics |
| `/ingest` | POST | Upload and process PDF |
| `/query` | POST | Query knowledge base |
| `/stats` | GET | Detailed system statistics |
| `/reset` | POST | Clear all data (destructive) |
| `/documents` | GET | List uploaded PDFs |
| `/documents/{filename}` | DELETE | Delete specific PDF |

**Interactive API Documentation**: `http://localhost:8000/docs`

---

## Setup and Usage

### Prerequisites

- **Python 3.9+**
- **Mistral API key** ([get one here](https://console.mistral.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/darrenjian/RAG-System.git
   cd RAG-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Create environment file**
   ```bash
   cp .env.example .env
   ```

2. **Add your Mistral API key**

   Edit `.env`:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

3. **Customize model configurations** (optional)

   Edit `app/config.py` to adjust:
   - Model selection (`embedding_model`, `llm_model`, `ocr_model`)
   - RAG parameters (`chunk_size`, `top_k_results`, `similarity_threshold`)
   - Hybrid search weights (`semantic_weight`, `keyword_weight`)

### Running the Server

**Development mode** (auto-reload on file changes):
```bash
python run.py
```

**Production mode**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server starts at: `http://localhost:8000`

### Using the Chat Interface

1. Open `http://localhost:8000` in your browser
2. Click **"Upload PDF"** in the sidebar
3. Select a PDF file (digital PDFs work best)
4. Wait for processing confirmation
5. Ask questions about the document!

**Example queries:**
- "What is this document about?"
- "Summarize the key findings"
- "What were the Q4 revenue numbers?"

### Using the API Directly

**Upload a PDF:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf"
```

**Query the knowledge base:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5,
    "use_hybrid_search": true
  }'
```

**Check system health:**
```bash
curl http://localhost:8000/health
```

---

## Future Improvements

### Top 5 Planned Enhancements

1. **Adaptive Similarity Threshold**: Automatically calibrate threshold based on document characteristics and query patterns (e.g., 0.6 for technical docs, 0.4 for general content)

2. **Production-Grade Vector DB**: Integrate FAISS or Pinecone/Weaviate for scalability to millions of vectors with sub-millisecond search times

3. **Multi-Modal PDF Understanding**: Extract and query tables, charts, and images using vision models for comprehensive document analysis

4. **Conversation Memory**: Track context across queries within a session to enable multi-turn dialogue and follow-up questions

5. **Testing & Production Readiness**: Comprehensive test suite, Docker containerization, CI/CD pipeline, and monitoring infrastructure

### Known Limitations

- **Vector DB Scale**: Custom NumPy implementation is O(n) linear search; becomes slow beyond 100k vectors
- **BM25 Index Scale**: Inverted index grows with vocabulary size; for millions of documents, dedicated search engine (Elasticsearch) would be needed
- **No conversation state**: Each query is independent; no multi-turn dialogue tracking

---

**Built by Darren Jian**
