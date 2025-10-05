"""Coordinates document ingestion and question answering."""

import os
import time
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

from app.vector_db import VectorDatabase
from app.mistral_client import MistralClient
from app.chunking import TextChunker
from app.search import HybridSearch
from app.models import Intent, Citation, QueryResponse
from app.config import get_settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Manages document ingestion and query processing."""

    def __init__(self):
        """Initialize the RAG pipeline with all components"""
        settings = get_settings()

        # Initialize components
        self.vector_db = VectorDatabase(
            dimension=1024,  # Mistral embed dimension
            storage_path="vectordb"
        )
        self.mistral_client = MistralClient()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.hybrid_search = HybridSearch(
            vector_db=self.vector_db,
            semantic_weight=settings.semantic_weight,
            keyword_weight=settings.keyword_weight
        )

        # Configuration
        self.top_k = settings.top_k_results
        self.top_n_semantic = settings.top_n_semantic
        self.top_k_keyword = settings.top_k_keyword
        self.similarity_threshold = settings.similarity_threshold

        # Load existing database if available
        loaded = self.vector_db.load()

        # Rebuild BM25 index from loaded vectors
        if loaded and self.vector_db.size() > 0:
            logger.info(f"Rebuilding BM25 index from {self.vector_db.size()} loaded vectors...")
            bm25_docs = [
                {'id': self.vector_db.ids[i], 'text': self.vector_db.texts[i], 'metadata': self.vector_db.metadata[i]}
                for i in range(self.vector_db.size())
            ]
            self.hybrid_search.index_documents(bm25_docs)
            logger.info("BM25 index rebuilt successfully")

        logger.info("Initialized RAG Pipeline")

    async def ingest_pdf(self, pdf_path: str, filename: str) -> Dict:
        """
        Ingest a PDF document into the knowledge base

        Pipeline:
        1. Extract text using Mistral OCR
        2. Chunk the text
        3. Generate embeddings
        4. Store in vector database
        5. Index for keyword search

        Args:
            pdf_path: Path to the PDF file
            filename: Original filename

        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        logger.info(f"Starting ingestion of {filename}")

        try:
            # Step 1: OCR
            logger.debug("Extracting text with Mistral OCR...")
            full_text = self.mistral_client.process_pdf_ocr(pdf_path)

            if not full_text or len(full_text.strip()) < 100:
                raise ValueError("PDF extraction failed or document too short")

            # Step 2: Chunk
            logger.debug("Chunking text...")
            metadata = {
                'source_file': filename,
                'ingestion_time': time.time()
            }
            chunks = self.chunker.chunk_with_ids(full_text, filename, metadata)

            if not chunks:
                raise ValueError("No chunks created from document")

            # Step 3: Generate embeddings (batch in groups of 100 to avoid API limits)
            logger.debug(f"Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [text for _, text, _ in chunks]

            # Batch embeddings in groups of 100
            batch_size = 100
            embeddings = []
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                batch_embeddings = await self.mistral_client.get_embeddings(batch)
                embeddings.extend(batch_embeddings)
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunk_texts) + batch_size - 1)//batch_size}")

            # Step 4: Store in vector DB
            logger.debug("Storing in vector database...")
            entries = []
            for (chunk_id, chunk_text, chunk_meta), embedding in zip(chunks, embeddings):
                entries.append((
                    chunk_id,
                    np.array(embedding, dtype=np.float32),
                    chunk_text,
                    chunk_meta
                ))

            self.vector_db.add_batch(entries)

            # Step 5: Index for BM25
            logger.debug("Indexing for keyword search...")
            bm25_docs = [
                {'id': chunk_id, 'text': chunk_text, 'metadata': chunk_meta}
                for chunk_id, chunk_text, chunk_meta in chunks
            ]
            self.hybrid_search.index_documents(bm25_docs)

            # Save database
            self.vector_db.save()

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Ingestion complete: {len(chunks)} chunks in {elapsed:.0f}ms")

            return {
                'filename': filename,
                'chunks_created': len(chunks),
                'processing_time_ms': elapsed,
                'total_chars': len(full_text)
            }

        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise

    async def query(
        self,
        query: str,
        use_hybrid_search: bool = True,
        top_k: Optional[int] = None,
        enable_hallucination_check: bool = True
    ) -> QueryResponse:
        """
        Process a user query and generate an answer

        Pipeline:
        1. Detect intent (greeting vs factual)
        2. Transform query (if needed)
        3. Generate query embedding
        4. Search (hybrid or semantic only)
        5. Check similarity threshold
        6. Generate answer
        7. Verify for hallucinations (optional)

        Args:
            query: User's question
            use_hybrid_search: Use hybrid search (semantic + keyword)
            top_k: Number of results to retrieve
            enable_hallucination_check: Run post-generation verification

        Returns:
            QueryResponse with answer, citations, and metadata
        """
        start_time = time.time()
        top_k = top_k or self.top_k

        try:
            # Step 1: Intent detection
            logger.debug("Detecting query intent...")
            intent_str = await self.mistral_client.classify_intent(query)
            intent = self._parse_intent(intent_str)

            # Handle non-retrieval intents
            if intent in [Intent.GREETING, Intent.CHITCHAT]:
                return self._create_simple_response(query, intent, start_time)

            # Step 2: Query transformation
            logger.debug("Transforming query...")
            enhanced_query = await self.mistral_client.transform_query(query)

            # Step 3: Generate embedding
            logger.debug("Generating query embedding...")
            query_embedding = await self.mistral_client.get_embedding(enhanced_query)

            # Step 4: Search
            if use_hybrid_search:
                logger.debug(f"Performing hybrid search (top_n={self.top_n_semantic} semantic, top_k={self.top_k_keyword} keyword)...")
                results = await self.hybrid_search.search(
                    query=enhanced_query,
                    query_embedding=np.array(query_embedding, dtype=np.float32),
                    top_k=top_k,
                    top_n_semantic=self.top_n_semantic,
                    top_k_keyword=self.top_k_keyword
                )
                score_key = 'hybrid_score'
            else:
                logger.debug("Performing semantic search only...")
                results = self.vector_db.search(
                    query_vector=np.array(query_embedding, dtype=np.float32),
                    top_k=top_k
                )
                score_key = 'similarity'

            # Step 5: Check confidence threshold
            if not results or results[0][score_key] < self.similarity_threshold:
                return QueryResponse(
                    answer="I don't have sufficient evidence to answer this question. The available documents may not contain relevant information.",
                    citations=[],
                    intent=intent,
                    confidence=results[0][score_key] if results else 0.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Step 6: Generate answer
            logger.debug("Generating answer with LLM...")
            # Use more chunks for summary queries to get complete context
            is_summary_query = any(word in query.lower() for word in [
                'summarize', 'summary', 'overview', 'list', 'all', 'document'
            ])
            num_chunks = 10 if is_summary_query else 5
            context_chunks = [r['text'] for r in results[:num_chunks]]  # Use top 5-10 for context
            answer = await self.mistral_client.generate_answer(query, context_chunks)

            # Build citations
            citations = self._build_citations(results[:5])

            # Step 7: Hallucination check (optional)
            confidence = results[0][score_key]

            # Detect summary/synthesis queries where hallucination check should be skipped
            is_summary_query = any(word in query.lower() for word in [
                'summarize', 'summary', 'overview', 'explain', 'describe',
                'what is this document about', 'tell me about', 'what does this document'
            ])

            if enable_hallucination_check and answer and not is_summary_query:
                logger.debug("Checking for hallucinations...")
                verification = await self.mistral_client.detect_hallucination(
                    answer, context_chunks
                )

                if verification['is_hallucination']:
                    logger.warning("Hallucination detected in answer")
                    answer = "I found relevant information, but I cannot provide a fully supported answer. Please review the source documents directly."
                    confidence *= verification['confidence']
            elif is_summary_query:
                logger.debug("Skipping hallucination check for summary/synthesis query")

            elapsed = (time.time() - start_time) * 1000

            return QueryResponse(
                answer=answer,
                citations=citations,
                intent=intent,
                confidence=confidence,
                processing_time_ms=elapsed
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            # Return a graceful error response instead of raising
            return QueryResponse(
                answer=f"I encountered an error while processing your question. This might be due to complex or ambiguous queries. Please try rephrasing your question or being more specific about which document you'd like me to reference.",
                citations=[],
                intent=Intent.RETRIEVAL_NEEDED,
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _parse_intent(self, intent_str: str) -> Intent:
        """Parse intent string to Intent enum"""
        intent_map = {
            'greeting': Intent.GREETING,
            'chitchat': Intent.CHITCHAT,
            'factual': Intent.FACTUAL,
            'retrieval_needed': Intent.RETRIEVAL_NEEDED
        }
        return intent_map.get(intent_str, Intent.RETRIEVAL_NEEDED)

    def _create_simple_response(self, query: str, intent: Intent, start_time: float) -> QueryResponse:
        """Create a response for non-retrieval queries (greetings, etc.)"""
        responses = {
            Intent.GREETING: "Hello! I'm here to help answer questions about your documents. Feel free to ask me anything!",
            Intent.CHITCHAT: "I'm a document assistant designed to help you find information in your uploaded PDFs. What would you like to know?"
        }

        return QueryResponse(
            answer=responses.get(intent, "How can I help you today?"),
            citations=[],
            intent=intent,
            confidence=1.0,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    def _build_citations(self, results: List[Dict]) -> List[Citation]:
        """Build citation objects from search results"""
        citations = []

        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            text = result.get('text', '')
            similarity = result.get('similarity', result.get('hybrid_score', 0.0))

            # Truncate text snippet to max 200 chars (including ellipsis)
            if len(text) > 197:
                text_snippet = text[:197] + '...'
            else:
                text_snippet = text

            citation = Citation(
                source_file=metadata.get('source_file', 'Unknown'),
                page_number=metadata.get('page_number'),
                chunk_id=result.get('id', f'chunk_{i}'),
                similarity_score=float(similarity),
                text_snippet=text_snippet
            )
            citations.append(citation)

        return citations

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        db_stats = self.vector_db.get_statistics()

        return {
            'vector_db': db_stats,
            'configuration': {
                'chunk_size': self.chunker.chunk_size,
                'chunk_overlap': self.chunker.chunk_overlap,
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold
            }
        }

    def reset(self) -> None:
        """Clear all data from the pipeline"""
        logger.warning("Resetting RAG pipeline - all data will be lost")
        self.vector_db.clear()
        self.vector_db.save()


# Global pipeline instance
_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
