"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Intent(str, Enum):
    """Query intent classification"""
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    FACTUAL = "factual"
    RETRIEVAL_NEEDED = "retrieval_needed"


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    use_hybrid_search: Optional[bool] = Field(True, description="Use hybrid search (semantic + keyword)")


class Citation(BaseModel):
    """Citation information for a source"""
    source_file: str
    page_number: Optional[int] = None
    chunk_id: str
    similarity_score: float
    text_snippet: str = Field(..., max_length=200)


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    citations: List[Citation] = []
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float


class IngestionRequest(BaseModel):
    """Request model for ingestion endpoint"""
    filename: str


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    message: str
    filename: str
    chunks_created: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_db_initialized: bool
    total_documents: int
    total_chunks: int
