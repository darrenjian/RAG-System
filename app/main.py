"""FastAPI backend for the RAG system."""

import os
import time
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from app.models import (
    QueryRequest,
    QueryResponse,
    IngestionResponse,
    HealthResponse
)
from app.rag_pipeline import get_rag_pipeline
from app.config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system for PDF documents",
    version="1.0.0"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
settings = get_settings()

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize RAG pipeline
rag_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    logger.info("Starting RAG System...")
    rag_pipeline = get_rag_pipeline()
    logger.info("RAG System ready!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI"""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse("""
        <html>
            <head><title>RAG System</title></head>
            <body>
                <h1>RAG System API</h1>
                <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>Chat UI not found. Please check the ui/ directory.</p>
            </body>
        </html>
    """)


@app.get("/documents")
async def list_documents():
    """
    List all uploaded PDF documents

    Returns:
        List of uploaded document filenames with sizes
    """
    try:
        documents = []
        if UPLOAD_DIR.exists():
            for pdf_file in UPLOAD_DIR.glob("*.pdf"):
                file_size = pdf_file.stat().st_size
                documents.append({
                    "filename": pdf_file.name,
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })

        # Sort by filename
        documents.sort(key=lambda x: x['filename'])

        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "count": 0}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a PDF document from the uploads folder

    Note: This only deletes the file from disk, not from the vector database.
    To fully remove a document's data, you would need to reset the entire database.

    Args:
        filename: Name of the file to delete

    Returns:
        Success message

    Raises:
        HTTPException: If file not found or deletion fails
    """
    try:
        file_path = UPLOAD_DIR / filename

        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Check if it's a PDF
        if not filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files can be deleted")

        # Delete the file
        file_path.unlink()
        logger.info(f"Deleted file: {filename}")

        return {
            "message": f"Document '{filename}' deleted successfully",
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns system status and statistics
    """
    try:
        stats = rag_pipeline.get_stats()

        return HealthResponse(
            status="healthy",
            vector_db_initialized=True,
            total_documents=stats['vector_db']['unique_sources'],
            total_chunks=stats['vector_db']['total_vectors']
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            vector_db_initialized=False,
            total_documents=0,
            total_chunks=0
        )


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a PDF document into the knowledge base

    This endpoint:
    1. Accepts a PDF file upload
    2. Extracts text using Mistral OCR
    3. Chunks the text
    4. Generates embeddings
    5. Stores in vector database

    Args:
        file: PDF file to ingest

    Returns:
        Ingestion statistics

    Raises:
        HTTPException: If ingestion fails
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Received file: {file.filename} ({len(content)} bytes)")

        # Process the PDF
        result = await rag_pipeline.ingest_pdf(str(file_path), file.filename)

        return IngestionResponse(
            message="Document ingested successfully",
            filename=file.filename,
            chunks_created=result['chunks_created'],
            processing_time_ms=result['processing_time_ms']
        )

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        # Optionally delete uploaded file after processing
        # file_path.unlink(missing_ok=True)
        pass


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base

    This endpoint:
    1. Detects query intent
    2. Transforms the query for better retrieval
    3. Searches the knowledge base (hybrid search by default)
    4. Generates an answer using the LLM
    5. Verifies the answer for hallucinations
    6. Returns the answer with citations

    Args:
        request: Query request with question and options

    Returns:
        Answer with citations and metadata

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        response = await rag_pipeline.query(
            query=request.query,
            use_hybrid_search=request.use_hybrid_search,
            top_k=request.top_k,
            enable_hallucination_check=True
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/reset")
async def reset_database():
    """
    Reset the entire knowledge base

    WARNING: This deletes all ingested documents!

    Returns:
        Confirmation message
    """
    try:
        rag_pipeline.reset()
        return {"message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """
    Get system statistics

    Returns:
        Detailed statistics about the RAG system
    """
    try:
        stats = rag_pipeline.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
