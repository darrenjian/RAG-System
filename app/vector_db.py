"""Simple vector database for semantic search using cosine similarity."""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A single entry in the vector database"""
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict


class VectorDatabase:
    """In-memory vector store with cosine similarity search."""

    def __init__(self, dimension: int = 1024, storage_path: str = "vectordb"):
        """
        Initialize the vector database

        Args:
            dimension: Dimension of the embedding vectors (Mistral embed: 1024)
            storage_path: Directory to store persisted data
        """
        self.dimension = dimension
        self.storage_path = storage_path

        # Core data structures
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        logger.info(f"Initialized VectorDatabase with dimension={dimension}")

    def add(self, id: str, vector: np.ndarray, text: str, metadata: Dict = None) -> None:
        """
        Add a single vector to the database

        Args:
            id: Unique identifier for this entry
            vector: Embedding vector (must match self.dimension)
            text: Original text that was embedded
            metadata: Additional metadata (source file, page, etc.)
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match database dimension {self.dimension}")

        # Normalize vector for cosine similarity
        normalized_vector = self._normalize(vector.reshape(1, -1))

        # Append to arrays
        self.vectors = np.vstack([self.vectors, normalized_vector])
        self.ids.append(id)
        self.texts.append(text)
        self.metadata.append(metadata or {})

        logger.debug(f"Added vector {id} to database (total: {len(self.ids)})")

    def add_batch(self, entries: List[Tuple[str, np.ndarray, str, Dict]]) -> None:
        """
        Add multiple vectors at once (more efficient than individual adds)

        Args:
            entries: List of (id, vector, text, metadata) tuples
        """
        if not entries:
            return

        ids, vectors, texts, metadatas = zip(*entries)

        # Stack and normalize vectors
        vectors_array = np.vstack(vectors)
        normalized_vectors = self._normalize(vectors_array)

        # Append to existing data
        self.vectors = np.vstack([self.vectors, normalized_vectors]) if len(self.vectors) > 0 else normalized_vectors
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

        logger.info(f"Added {len(entries)} vectors to database (total: {len(self.ids)})")

    def search(self, query_vector: np.ndarray, top_k: int = 5, filter_fn: Optional[callable] = None) -> List[Dict]:
        """
        Search for the most similar vectors to the query

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_fn: Optional function to filter results (takes metadata, returns bool)

        Returns:
            List of results with similarity scores, sorted by similarity (descending)
        """
        if len(self.vectors) == 0:
            logger.warning("Search called on empty database")
            return []

        # Normalize query vector
        query_normalized = self._normalize(query_vector.reshape(1, -1))

        # Compute cosine similarity: dot product of normalized vectors
        similarities = np.dot(self.vectors, query_normalized.T).flatten()

        # Apply filter if provided
        if filter_fn:
            valid_indices = [i for i, meta in enumerate(self.metadata) if filter_fn(meta)]
            filtered_similarities = np.full_like(similarities, -np.inf)
            filtered_similarities[valid_indices] = similarities[valid_indices]
            similarities = filtered_similarities

        # Get top-k indices
        top_k = min(top_k, len(self.ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            idx = int(idx)  # Convert numpy int to Python int
            if similarities[idx] == -np.inf:  # Skip filtered out results
                continue
            results.append({
                'id': self.ids[idx],
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })

        logger.debug(f"Search returned {len(results)} results")
        return results

    def delete(self, id: str) -> bool:
        """
        Delete a vector by ID

        Args:
            id: ID of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        if id not in self.ids:
            return False

        idx = self.ids.index(id)

        # Remove from all arrays
        self.vectors = np.delete(self.vectors, idx, axis=0)
        self.ids.pop(idx)
        self.texts.pop(idx)
        self.metadata.pop(idx)

        logger.info(f"Deleted vector {id} from database")
        return True

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get a vector entry by ID"""
        if id not in self.ids:
            return None

        idx = self.ids.index(id)
        return VectorEntry(
            id=id,
            vector=self.vectors[idx],
            text=self.texts[idx],
            metadata=self.metadata[idx]
        )

    def size(self) -> int:
        """Return the number of vectors in the database"""
        return len(self.ids)

    def clear(self) -> None:
        """Remove all vectors from the database"""
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.ids = []
        self.texts = []
        self.metadata = []
        logger.info("Cleared all vectors from database")

    def save(self, name: str = "default") -> None:
        """
        Persist the database to disk

        Args:
            name: Name of the saved database
        """
        save_path = os.path.join(self.storage_path, f"{name}.pkl")

        data = {
            'dimension': self.dimension,
            'vectors': self.vectors,
            'ids': self.ids,
            'texts': self.texts,
            'metadata': self.metadata
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved database to {save_path} ({len(self.ids)} vectors)")

    def load(self, name: str = "default") -> bool:
        """
        Load a database from disk

        Args:
            name: Name of the database to load

        Returns:
            True if loaded successfully, False if not found
        """
        load_path = os.path.join(self.storage_path, f"{name}.pkl")

        if not os.path.exists(load_path):
            logger.warning(f"Database file not found: {load_path}")
            return False

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.dimension = data['dimension']
        self.vectors = data['vectors']
        self.ids = data['ids']
        self.texts = data['texts']
        self.metadata = data['metadata']

        logger.info(f"Loaded database from {load_path} ({len(self.ids)} vectors)")
        return True

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity

        Cosine similarity of normalized vectors = dot product
        This makes search more efficient (no division needed)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_vectors': len(self.ids),
            'dimension': self.dimension,
            'storage_path': self.storage_path,
            'unique_sources': len(set(m.get('source_file', '') for m in self.metadata)),
            'memory_usage_mb': self.vectors.nbytes / (1024 * 1024)
        }
