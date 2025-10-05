# RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline for intelligent PDF document processing and question answering, built with FastAPI and Mistral AI.

## Features

### Core Functionality
- **PDF Ingestion**: Upload PDFs with OCR-powered text extraction (Mistral OCR)
- **Semantic Search**: Custom vector database implementation (no external dependencies)
- **Hybrid Search**: Combines semantic (embeddings) + keyword (BM25) search
- **LLM Generation**: Context-aware answer generation with Mistral AI
- **Citation System**: Confidence thresholds and source attribution

### Additional Features
- **Intent Detection**: Automatically classifies query types (greeting, factual, etc.)
- **Query Transformation**: Enhances queries for better retrieval
- **Hallucination Detection**: Post-generation verification of answers
- **Result Re-ranking**: Improves retrieval quality through score combination
- **Chat Interface**: Beautiful, responsive UI for document Q&A

### Technical Highlights
- **FastAPI** backend with async support
- **Custom Vector DB** - no third-party vector databases used
- **BM25 Algorithm** - implemented from scratch for keyword search
- **Smart Chunking** - preserves semantic boundaries
- **Type Safety** - Pydantic models throughout

---

## Table of Contents
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Design Decisions](#design-decisions)
- [Development](#development)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/darrenjian/RAG-System.git
cd RAG-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your Mistral API key

# 5. Run the server
python -m uvicorn app.main:app --reload

# 6. Open browser to http://localhost:8000
```

The chat interface will be available at `http://localhost:8000`

---

## System Architecture

### High-Level Overview

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
│      → Extract text page-by-page                                   │
│      ↓                                                              │
│  [2] Text Chunking (Sentence-based)                                │
│      → Split into semantic units using NLTK                        │
│      → ~200 chunks for 39-page document                            │
│      ↓                                                              │
│  [3] Embedding Generation (Mistral API)                            │
│      → Convert all chunks to embeddings                            │
│      → 1024-dimensional vectors                                     │
│      ↓                                                              │
│  [4] Vector Database Storage                                        │
│      → Store vectors + metadata                                    │
│      → Index for BM25 keyword search                               │
│      ↓                                                              │
│  [5] Persist to Disk                                                │
│      → Save vectordb/ for future queries                           │
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
│      → Skip retrieval for greetings                                │
│      ↓                                                              │
│  [2] Query Transformation (Mistral LLM)                            │
│      → Expand abbreviations, add synonyms                          │
│      → "Q4 revenue" → "fourth quarter revenue earnings"            │
│      ↓                                                              │
│  [3] Query Embedding (Mistral API)                                 │
│      → Convert enhanced query to vector                            │
│      ↓                                                              │
│  [4] Hybrid Search                                                  │
│      ├─ Semantic Search (Cosine Similarity)                        │
│      │   → Find top-10 similar vectors                             │
│      │   → Score: 0.0 to 1.0                                       │
│      │                                                              │
│      ├─ Keyword Search (BM25)                                       │
│      │   → Find top-10 by keyword matching                         │
│      │   → Score: TF-IDF based                                     │
│      │                                                              │
│      └─ Combine & Re-rank                                          │
│          → Hybrid score = 0.7×semantic + 0.3×keyword               │
│          → Return top-5 results                                    │
│      ↓                                                              │
│  [5] Confidence Check                                               │
│      → If top score < 0.70: "I don't have enough evidence"         │
│      ↓                                                              │
│  [6] Answer Generation (Mistral LLM)                               │
│      → Provide context + constraints                               │
│      → Generate grounded answer with citations                     │
│      ↓                                                              │
│  [7] Hallucination Detection (Optional)                            │
│      → Verify answer against context                               │
│      → Override if unsupported claims detected                     │
│      ↓                                                              │
│  [8] Return Response                                                │
│      → Answer + Citations + Confidence + Processing Time           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works

#### 1. **Document Ingestion**

When you upload a PDF:

1. **Text Extraction**: PyPDF2 extracts text from each page
   - Fast and free for digital PDFs
   - Falls back to Mistral OCR for scanned documents

2. **Chunking**: Splits text into sentences using NLTK
   - Preserves semantic boundaries
   - Each sentence becomes a searchable unit
   - Tracks metadata (source file, chunk index)

3. **Embedding**: Converts chunks to vectors via Mistral API
   - Generates embeddings for all chunks
   - 1024-dimensional vectors capture semantic meaning

4. **Storage**: Saves to custom vector database
   - NumPy-based cosine similarity search
   - Also indexes for BM25 keyword search
   - Persists to disk for future queries

#### 2. **Query Processing**

When you ask a question:

1. **Intent Detection**: LLM classifies query intent
   - Greetings → Skip retrieval, respond directly
   - Factual queries → Proceed with search

2. **Query Enhancement**: LLM improves query
   - Expands abbreviations: "ML" → "machine learning"
   - Adds synonyms: "profit" → "profit earnings income"

3. **Hybrid Search**: Two-pronged retrieval
   - **Semantic**: Vector similarity finds conceptually similar chunks
   - **Keyword**: BM25 finds exact keyword matches
   - **Combination**: Weighted merge (70% semantic, 30% keyword)

4. **Confidence Filtering**: Checks if evidence is strong enough
   - Threshold: 0.70 similarity score
   - Below threshold → Refuse to answer

5. **Answer Generation**: LLM generates grounded response
   - Provides retrieved context as evidence
   - Enforces constraints: "Only use provided context"
   - Includes citations with source attribution

6. **Verification**: Optional hallucination check
   - Verifies answer against source documents
   - Overrides if unsupported claims detected

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | Async API server with auto-docs |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | ASGI server for FastAPI |
| **LLM/Embeddings** | [Mistral AI](https://mistral.ai/) | OCR, embeddings, answer generation |
| **Vector DB** | Custom ([NumPy](https://numpy.org/)) | Cosine similarity search |
| **Keyword Search** | BM25 (custom implementation) | Exact term matching |
| **Text Processing** | [NLTK](https://www.nltk.org/) | Sentence tokenization |
| **PDF Extraction** | [PyPDF2](https://pypdf2.readthedocs.io/) | Text extraction from PDFs |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) | Request/response models |
| **Environment** | [python-dotenv](https://github.com/theskumar/python-dotenv) | Configuration management |
| **Frontend** | HTML/CSS/JavaScript | Chat interface (vanilla, no framework) |

### Key Design Choices

1. **Why Custom Vector DB?**
   - Educational: Understand how vector search works
   - No dependencies: Complete control
   - Sufficient for small-medium scale (<100k vectors)

2. **Why Hybrid Search?**
   - Semantic search: Good for conceptual queries
   - Keyword search: Good for exact terms/codes
   - Hybrid: Best of both worlds

3. **Why Sentence-Based Chunking?**
   - Preserves semantic boundaries
   - Enables precise citations
   - Better than fixed-size (which splits mid-sentence)

4. **Why Intent Detection?**
   - Saves API costs (skip retrieval for greetings)
   - Improves UX (instant responses)
   - Reduces latency (~50ms vs ~2000ms)

### File Structure

```
RAG-System/
├── app/
│   ├── main.py              # FastAPI endpoints (/ingest, /query, /health)
│   ├── rag_pipeline.py      # Orchestrates ingestion & query flow
│   ├── vector_db.py         # Custom vector database (cosine similarity)
│   ├── mistral_client.py    # Mistral API wrapper (OCR, embeddings, LLM)
│   ├── search.py            # Hybrid search (semantic + BM25)
│   ├── chunking.py          # Text chunking (sentence-based)
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # Environment variable configuration
├── ui/
│   └── index.html           # Chat interface (upload PDFs, ask questions)
├── vectordb/                # Persisted vector database (created at runtime)
└── uploads/                 # Uploaded PDFs (created at runtime)
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Mistral API key ([get one here](https://console.mistral.ai/))

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/darrenjian/RAG-System.git
   cd RAG-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

5. **Verify installation**
   ```bash
   python -c "import app; print('Installation successful!')"
   ```

---

## Usage

### Starting the Server

```bash
# Development mode (auto-reload)
python -m uvicorn app.main:app --reload

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Using the Chat Interface

1. Open `http://localhost:8000` in your browser
2. Upload a PDF document using the sidebar
3. Wait for processing (you'll see a confirmation message)
4. Ask questions about the document!

### Using the API Directly

#### Upload a PDF
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@your_document.pdf"
```

#### Query the knowledge base
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "top_k": 5,
    "use_hybrid_search": true
  }'
```

#### Health check
```bash
curl http://localhost:8000/health
```

---

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve chat UI |
| `/health` | GET | System health and statistics |
| `/ingest` | POST | Upload and process PDF |
| `/query` | POST | Query the knowledge base |
| `/stats` | GET | Detailed system statistics |
| `/reset` | POST | Clear all data (WARNING: destructive) |

### Request/Response Examples

**Ingestion Request:**
```json
POST /ingest
Content-Type: multipart/form-data

file: <PDF file>
```

**Ingestion Response:**
```json
{
  "message": "Document ingested successfully",
  "filename": "example.pdf",
  "chunks_created": 42,
  "processing_time_ms": 3542.1
}
```

**Query Request:**
```json
POST /query
{
  "query": "What is the revenue in Q4?",
  "top_k": 5,
  "use_hybrid_search": true
}
```

**Query Response:**
```json
{
  "answer": "The Q4 revenue was $1.2 billion, representing a 15% increase year-over-year [1][2]",
  "citations": [
    {
      "source_file": "annual_report.pdf",
      "page_number": 5,
      "chunk_id": "annual_report.pdf_chunk_12",
      "similarity_score": 0.89,
      "text_snippet": "Q4 revenue reached $1.2B..."
    }
  ],
  "intent": "factual",
  "confidence": 0.89,
  "processing_time_ms": 1247.3
}
```

---

## Configuration

Configuration is managed through environment variables (`.env` file):

```bash
# Mistral API
MISTRAL_API_KEY=your_key_here

# Server
HOST=0.0.0.0
PORT=8000

# RAG Parameters
CHUNK_SIZE=512              # Characters per chunk
CHUNK_OVERLAP=128           # Overlap between chunks
TOP_K_RESULTS=5             # Number of results to retrieve
SIMILARITY_THRESHOLD=0.7    # Minimum similarity for answers
MAX_CONTEXT_LENGTH=4000     # Max context for LLM

# Models
EMBEDDING_MODEL=mistral-embed
LLM_MODEL=mistral-small-latest
OCR_MODEL=mistral-ocr-latest

# Hybrid Search Weights
SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
```

---

**Key highlights:**

1. **Custom Vector Database**: Built from scratch using NumPy for educational purposes and to avoid external dependencies
2. **Hybrid Search**: Combines semantic (embeddings) and keyword (BM25) search for better retrieval
3. **Intent Detection**: Saves API calls by classifying queries before retrieval
4. **Hallucination Detection**: Verifies LLM answers against source documents
5. **Citation System**: Enforces confidence thresholds and provides source attribution

---

## Development

### Project Structure
```
RAG-System/
├── app/                    # Application code
│   ├── main.py            # FastAPI app
│   ├── rag_pipeline.py    # Main pipeline
│   ├── vector_db.py       # Vector database
│   ├── mistral_client.py  # Mistral API client
│   ├── search.py          # Hybrid search
│   ├── chunking.py        # Text chunking
│   ├── models.py          # Pydantic models
│   └── config.py          # Configuration
├── ui/                    # Frontend
│   └── index.html         # Chat interface
├── vectordb/              # Vector DB storage (created at runtime)
├── uploads/               # Uploaded PDFs (created at runtime)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── DESIGN_DECISIONS.md   # Architecture documentation
```

### Running Tests
```bash
# Unit tests (if implemented)
pytest tests/

# Manual testing
python -c "from app.vector_db import VectorDatabase; print('Vector DB OK')"
python -c "from app.search import BM25; print('BM25 OK')"
```

### Code Quality
```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

---

## Future Improvements

- Multi-modal support (images, tables, charts)
- Conversation memory and context tracking
- Advanced chunking strategies (semantic-aware)
- Query routing based on intent
- User feedback loop for continuous improvement
- Production-grade vector database integration
- Monitoring and logging
- Docker containerization
- Unit and integration tests

---

## Implementation Considerations

This section documents key design decisions and trade-offs made during implementation.

### PDF Text Extraction

**Library Choice: PyPDF2**

We use PyPDF2 for text extraction from PDFs instead of more advanced OCR solutions. Here's why:

**Advantages:**
- **Speed**: Local processing is faster than API-based OCR
- **Cost**: No API costs per page
- **Simplicity**: Mature, well-documented library with minimal dependencies
- **Sufficient for Digital PDFs**: Most modern PDFs contain extractable text

**Limitations:**
- **Scanned Documents**: Cannot extract text from image-based PDFs (scanned documents)
- **Complex Layouts**: May struggle with multi-column layouts, tables, or unusual formatting
- **Image Content**: Ignores charts, diagrams, and embedded images

**When PyPDF2 Works Best:**
- Digital PDFs (created from Word, LaTeX, etc.)
- Standard single-column text documents
- Reports, articles, research papers

**Alternative Considered: Mistral OCR API**

Mistral offers an OCR API that could handle scanned documents and complex layouts, but we chose PyPDF2 because:
- Most use cases involve digital PDFs
- Faster processing (no network latency)
- No per-page API costs
- Users can easily swap to OCR if needed

**Future Enhancement**: Add automatic fallback to Mistral OCR when PyPDF2 extracts minimal text (indicating a scanned document).

---

### Text Chunking Strategy

**Approach: Sentence-Based Chunking**

We chunk text by sentences using NLTK's sentence tokenizer rather than fixed-size chunking.

**Why Sentence-Based?**

1. **Semantic Coherence**: Each chunk is a complete thought
   - Sentences have natural boundaries
   - No mid-sentence splits that break meaning
   - Example: "The revenue was $1.2M" stays together

2. **Precise Citations**: Can point to exact sentences
   - Users see the specific sentence that answered their question
   - More useful than "characters 512-1024 of the document"

3. **Better Embeddings**: Complete sentences encode meaning better
   - Embedding models are trained on sentences
   - Fragmented text produces lower-quality embeddings

**Implementation Details:**

```python
# Using NLTK's sentence tokenizer
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)

# Filter very short sentences (likely noise)
if len(sentence) < 10:
    continue  # Skip page numbers, headers, etc.
```

**Trade-offs:**

| Aspect | Sentence-Based | Fixed-Size (512 tokens) |
|--------|---------------|------------------------|
| Semantic coherence | Perfect | Splits mid-sentence |
| Chunk count | Variable (many chunks) | Predictable |
| Citation precision | Exact sentences | Approximate ranges |
| Edge cases | Handles abbreviations (Dr., U.S.A.) | Simple to implement |

**Why Not Fixed-Size Chunking?**

Fixed-size chunking (e.g., 512 tokens with 128 overlap) is common in RAG systems:

```python
# Fixed-size approach (not used)
for i in range(0, len(tokens), chunk_size - overlap):
    chunk = tokens[i:i + chunk_size]
```

**Problems with fixed-size:**
- Splits sentences: "The Q4 revenue was $1.2 [CHUNK BREAK] million, up 15%..."
- Overlap creates duplicate information (wastes embedding API calls)
- Harder to provide meaningful citations

**When Fixed-Size is Better:**
- Very long documents where sentence-level creates too many chunks
- Non-English text where sentence tokenization is poor
- Need consistent chunk sizes for batch processing

**Handling Edge Cases:**

1. **Abbreviations**: NLTK handles "Dr. Smith", "U.S.A.", "etc." correctly
2. **Short Sentences**: Filter out sentences < 10 characters (page numbers, headers)
3. **Very Long Sentences**: Could split at commas/semicolons (not currently implemented)

---

### Chunking Considerations Summary

**Our Choices:**
- PyPDF2 for text extraction (fast, free, sufficient for digital PDFs)
- Sentence-based chunking (semantic coherence, precise citations)
- NLTK for sentence tokenization (handles edge cases)
- Filter sentences < 10 characters (remove noise)

**Trade-offs Accepted:**
- Cannot handle scanned PDFs (could add Mistral OCR fallback)
- Many small chunks for long documents (acceptable for <100k chunks)
- Requires NLTK dependency (lightweight, standard library)

**Future Improvements:**
- Add OCR fallback for scanned documents
- Implement semantic chunking (split on topic changes)
- Support multi-modal content (images, tables, charts)
- Add chunk size limits for very long sentences

---

