"""Combines vector similarity and BM25 keyword search for better results."""

import math
import re
from typing import List, Dict, Set
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class BM25:
    """Keyword search using the BM25 ranking algorithm."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization (0-1)
        """
        self.k1 = k1
        self.b = b

        # Corpus statistics
        self.corpus: List[Dict] = []  # List of {id, text, tokens, metadata}
        self.doc_freqs: Counter = Counter()  # How many docs contain each term
        self.idf: Dict[str, float] = {}  # IDF scores for each term
        self.doc_len: List[int] = []  # Length of each document
        self.avgdl: float = 0.0  # Average document length

        logger.info(f"Initialized BM25 (k1={k1}, b={b})")

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the BM25 index

        Args:
            documents: List of dicts with 'id', 'text', and 'metadata'
        """
        for doc in documents:
            tokens = self._tokenize(doc['text'])
            doc_entry = {
                'id': doc['id'],
                'text': doc['text'],
                'tokens': tokens,
                'metadata': doc.get('metadata', {})
            }
            self.corpus.append(doc_entry)
            self.doc_len.append(len(tokens))

            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Compute statistics
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self._compute_idf()

        logger.info(f"Indexed {len(documents)} documents (avg length: {self.avgdl:.1f} tokens)")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents matching the query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with BM25 scores
        """
        if not self.corpus:
            logger.warning("Search called on empty BM25 index")
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            score = self._score(query_tokens, doc['tokens'], idx)
            scores.append({
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc['metadata'],
                'bm25_score': score
            })

        # Sort by score descending
        scores.sort(key=lambda x: x['bm25_score'], reverse=True)

        top_k = min(top_k, len(scores))
        results = scores[:top_k]

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def _score(self, query_tokens: List[str], doc_tokens: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a document given a query

        Args:
            query_tokens: Tokenized query
            doc_tokens: Tokenized document
            doc_idx: Document index in corpus

        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = self.doc_len[doc_idx]

        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        for token in query_tokens:
            if token not in doc_term_freqs:
                continue

            # Term frequency in document
            freq = doc_term_freqs[token]

            # IDF score
            idf = self.idf.get(token, 0.0)

            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def _compute_idf(self) -> None:
        """
        Compute IDF (Inverse Document Frequency) for all terms

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        Where:
        - N = total number of documents
        - df = number of documents containing the term
        """
        num_docs = len(self.corpus)

        for term, doc_freq in self.doc_freqs.items():
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self.idf[term] = idf

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into words

        Simple tokenization:
        - Lowercase
        - Split on non-alphanumeric
        - Remove very short tokens
        """
        # Lowercase
        text = text.lower()

        # Split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)

        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 2]

        return tokens


class HybridSearch:
    """Blends vector similarity and keyword matching with weighted scoring."""

    def __init__(
        self,
        vector_db,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid search

        Args:
            vector_db: Vector database instance
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        self.vector_db = vector_db
        self.bm25 = BM25()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        logger.info(f"Initialized HybridSearch (semantic={semantic_weight}, keyword={keyword_weight})")

    def index_documents(self, documents: List[Dict]) -> None:
        """
        Index documents for both semantic and keyword search

        Args:
            documents: List of {id, text, metadata} dicts
        """
        # Index for BM25
        self.bm25.add_documents(documents)
        logger.info(f"Indexed {len(documents)} documents for hybrid search")

    async def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        top_n_semantic: int = 10,
        top_k_keyword: int = 10
    ) -> List[Dict]:
        """
        Perform hybrid search

        Strategy:
        1. Get top_n results from semantic search (cosine similarity)
        2. Get top_k results from keyword search (BM25)
        3. Combine and re-rank using weighted scores
        4. Return final top_k results

        Args:
            query: Text query
            query_embedding: Embedding vector for the query
            top_k: Final number of results to return
            top_n_semantic: Number of results from semantic search (default: 10)
            top_k_keyword: Number of results from keyword search (default: 10)

        Returns:
            Combined and re-ranked results
        """
        logger.debug(f"Hybrid search: top_n={top_n_semantic} semantic, top_k={top_k_keyword} keyword, final={top_k}")

        # Step 1: Get top-n from semantic search (cosine similarity on embeddings)
        semantic_results = self.vector_db.search(query_embedding, top_k=top_n_semantic)

        # Step 2: Get top-k from keyword search (BM25)
        keyword_results = self.bm25.search(query, top_k=top_k_keyword)

        # Step 3: Combine and normalize scores
        combined = self._combine_results(semantic_results, keyword_results)

        # Step 4: Sort by hybrid score (weighted combination)
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Step 5: Return final top-k results
        results = combined[:top_k]

        logger.info(f"Hybrid search: retrieved {len(semantic_results)} semantic + {len(keyword_results)} keyword → {len(results)} final results")
        return results

    def _combine_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine and normalize scores from both search methods

        Args:
            semantic_results: Results from vector search
            keyword_results: Results from BM25

        Returns:
            Combined results with hybrid scores
        """
        # Normalize semantic scores (cosine similarity is already 0-1)
        semantic_map = {r['id']: r for r in semantic_results}

        # Normalize BM25 scores to 0-1
        max_bm25 = max([r['bm25_score'] for r in keyword_results], default=1.0)
        if max_bm25 > 0:
            for r in keyword_results:
                r['normalized_bm25'] = r['bm25_score'] / max_bm25
        else:
            for r in keyword_results:
                r['normalized_bm25'] = 0.0

        keyword_map = {r['id']: r for r in keyword_results}

        # Combine scores
        all_ids = set(semantic_map.keys()) | set(keyword_map.keys())
        combined = []

        for doc_id in all_ids:
            semantic_score = semantic_map[doc_id]['similarity'] if doc_id in semantic_map else 0.0
            keyword_score = keyword_map[doc_id]['normalized_bm25'] if doc_id in keyword_map else 0.0

            # Weighted combination
            hybrid_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )

            # Get document info (prefer semantic result if available)
            if doc_id in semantic_map:
                result = semantic_map[doc_id].copy()
            else:
                result = keyword_map[doc_id].copy()

            result['hybrid_score'] = hybrid_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score

            combined.append(result)

        return combined
