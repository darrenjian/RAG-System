"""Splits documents into chunks for embedding and retrieval."""

import re
from typing import List, Tuple
import logging
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


class TextChunker:
    """Breaks text into sentence-level chunks for precise retrieval."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize the text chunker

        Args:
            chunk_size: Not used in sentence mode, kept for API compatibility
            chunk_overlap: Not used in sentence mode, kept for API compatibility
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("Initialized TextChunker (sentence-based mode)")

    def chunk(self, text: str, metadata: dict = None) -> List[Tuple[str, dict]]:
        """
        Split text into sentence-based chunks

        Each sentence becomes a separate chunk for precise retrieval.

        Args:
            text: Input text to chunk (markdown from Mistral OCR)
            metadata: Base metadata to attach to each chunk

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        # Clean text
        text = self._clean_text(text)

        # Split into sentences using NLTK
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)

        chunks = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:  # Skip very short sentences
                continue

            # Build chunk metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({"chunk_index": idx, "chunk_type": "sentence", "sentence_length": len(sentence)})

            chunks.append((sentence, chunk_metadata))

        logger.debug(f"Created {len(chunks)} sentence chunks from {len(text)} characters")
        return chunks

    def chunk_with_ids(self, text: str, doc_id: str, metadata: dict = None) -> List[Tuple[str, str, dict]]:
        """
        Chunk text and generate unique IDs for each chunk

        Args:
            text: Input text
            doc_id: Document identifier
            metadata: Base metadata

        Returns:
            List of (chunk_id, chunk_text, chunk_metadata) tuples
        """
        chunks = self.chunk(text, metadata)
        result = []

        for chunk_text, chunk_meta in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk_meta['chunk_index']}"
            result.append((chunk_id, chunk_text, chunk_meta))

        return result

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize text

        - Remove excessive whitespace
        - Normalize line breaks
        - Remove control characters
        """
        # Remove control characters except newlines and tabs
        text = "".join(char for char in text if char.isprintable() or char in "\n\t")

        # Normalize whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple newlines -> double newline
        text = re.sub(r" +", " ", text)  # Multiple spaces -> single space
        text = re.sub(r"\t+", " ", text)  # Tabs -> space

        return text.strip()


class SemanticChunker:
    """Groups paragraphs into chunks while respecting natural topic boundaries."""

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 128):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        logger.info(f"Initialized SemanticChunker (max={max_chunk_size}, min={min_chunk_size})")

    def chunk(self, text: str, metadata: dict = None) -> List[Tuple[str, dict]]:
        """
        Chunk text at semantic boundaries

        Algorithm:
        1. Split by paragraphs
        2. Detect headings and section breaks
        3. Group paragraphs into chunks that don't exceed max size
        4. Keep semantic units together
        """
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # Check if adding this paragraph would exceed max size
            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta["chunk_index"] = len(chunks)
                chunks.append((chunk_text, chunk_meta))

                # Start new chunk
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta["chunk_index"] = len(chunks)
            chunks.append((chunk_text, chunk_meta))

        logger.debug(f"Created {len(chunks)} semantic chunks")
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or more
        paragraphs = re.split(r"\n\s*\n+", text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]


def chunk_markdown(markdown_text: str, chunk_size: int = 512, chunk_overlap: int = 128) -> List[Tuple[str, dict]]:
    """
    Chunk markdown text while preserving structure

    This is useful for documents converted from PDFs via Mistral OCR,
    which returns markdown format.

    Args:
        markdown_text: Markdown formatted text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of (chunk_text, metadata) tuples
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(markdown_text)
