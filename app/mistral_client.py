"""Client for Mistral AI's OCR, embeddings, and text generation."""

import os
from typing import List, Dict, Any
from mistralai import Mistral
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)


class MistralClient:
    """Handles all interactions with Mistral AI's API."""

    def __init__(self, api_key: str = None):
        """
        Initialize Mistral client

        Args:
            api_key: Mistral API key (defaults to settings)
        """
        settings = get_settings()
        self.api_key = api_key or settings.mistral_api_key

        if not self.api_key:
            raise ValueError("Mistral API key is required. Please set MISTRAL_API_KEY in .env file")

        # Log only the first/last 4 chars of the API key for security
        if len(self.api_key) > 8:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        else:
            masked_key = "***"
        logger.info(f"Initializing MistralClient with API key: {masked_key}")

        self.client = Mistral(api_key=self.api_key)

        self.embedding_model = settings.embedding_model
        self.llm_model = settings.llm_model
        self.ocr_model = settings.ocr_model

        logger.info(f"Initialized MistralClient (embedding={self.embedding_model}, " f"llm={self.llm_model})")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(model=self.embedding_model, inputs=texts)

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []

    async def generate_answer(
        self, query: str, context_chunks: List[str], temperature: float = 0.3, max_tokens: int = 1000
    ) -> str:
        """
        Generate an answer using the LLM

        Args:
            query: User's question
            context_chunks: Retrieved context from documents
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response

        Returns:
            Generated answer
        """
        # Build prompt
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""You are a helpful assistant answering questions based on provided documents.

Context from documents:
{context}

User Question: {query}

Instructions:
1. Answer using information from the context above
2. Synthesize information across multiple context chunks when needed
3. If the context contains relevant information, provide a complete answer even if some
   details are missing
4. Only say "I don't have enough information" if the context is completely unrelated to
   the question
5. Cite your sources using [1], [2], etc. to reference the context chunks
6. Be comprehensive and informative while staying factual

FORMATTING INSTRUCTIONS:
- Use **Markdown formatting** for your response
- Use **bold** (**text**) for emphasis and important terms
- Use numbered lists (1. 2. 3.) for steps or ordered information
- Use bullet points (- item) for unordered lists
- Use headers (## Header) to organize long answers into sections
- Use tables when comparing multiple items

REFUSAL POLICIES:
- If asked for Personal Identifiable Information (PII) like SSNs, addresses, phone numbers,
  DO NOT provide them. Say: "I cannot provide personal identifiable information."
- If asked for legal advice, add this disclaimer: "**Disclaimer**: This is informational
  only, not legal advice. Consult a qualified attorney."
- If asked for medical advice, add this disclaimer: "**Disclaimer**: This is informational
  only, not medical advice. Consult a healthcare professional."

Answer:"""

        try:
            response = self.client.chat.complete(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            logger.debug(f"Generated answer ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def classify_intent(self, query: str) -> str:
        """
        Classify query intent to determine if retrieval is needed

        Args:
            query: User query

        Returns:
            Intent classification: greeting, chitchat, factual, retrieval_needed
        """
        prompt = f"""Classify the intent of this user query into one of these categories:
- greeting: Simple greetings like "hi", "hello", "hey"
- chitchat: Small talk like "how are you", "thank you"
- factual: Questions that need information from documents
- retrieval_needed: Any other query requiring document search

Query: "{query}"

Respond with only the category name, nothing else."""

        try:
            response = self.client.chat.complete(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=20
            )

            intent = response.choices[0].message.content.strip().lower()
            logger.debug(f"Classified intent: {intent}")
            return intent

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            # Default to retrieval if classification fails
            return "retrieval_needed"

    async def transform_query(self, query: str, conversation_history: List[str] = None) -> str:
        """
        Transform/enhance query for better retrieval

        Args:
            query: Original user query
            conversation_history: Previous queries for context

        Returns:
            Enhanced query
        """
        context = ""
        if conversation_history:
            context = "Previous queries:\n" + "\n".join(conversation_history[-3:]) + "\n\n"

        prompt = f"""{context}Current query: "{query}"

Improve this query for better document search by:
1. Expanding abbreviations
2. Adding relevant synonyms
3. Making it more specific if it references previous context

Return only the improved query, nothing else."""

        try:
            response = self.client.chat.complete(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=100
            )

            transformed = response.choices[0].message.content.strip()
            logger.debug(f"Transformed query: '{query}' -> '{transformed}'")
            return transformed

        except Exception as e:
            logger.error(f"Error transforming query: {e}")
            # Return original query if transformation fails
            return query

    async def detect_hallucination(self, answer: str, context: List[str]) -> Dict[str, Any]:
        """
        Verify if the answer is supported by the context

        Args:
            answer: Generated answer
            context: Context chunks used for generation

        Returns:
            Dict with verification result and confidence
        """
        context_text = "\n\n".join(context)

        prompt = f"""Context:
{context_text}

Generated Answer:
{answer}

Is the answer fully supported by the context? Answer with one word:
- SUPPORTED: All claims in the answer are backed by the context
- PARTIAL: Some claims are supported, some are not
- UNSUPPORTED: Answer contains information not in the context

Answer:"""

        try:
            response = self.client.chat.complete(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=20
            )

            result = response.choices[0].message.content.strip().upper()

            return {
                "is_hallucination": result == "UNSUPPORTED",
                "verification_status": result,
                "confidence": 1.0 if result == "SUPPORTED" else 0.5 if result == "PARTIAL" else 0.0,
            }

        except Exception as e:
            logger.error(f"Error detecting hallucination: {e}")
            return {"is_hallucination": False, "verification_status": "UNKNOWN", "confidence": 0.5}

    def process_pdf_ocr(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2

        Note: Mistral OCR API requires specific SDK support that may not be
        available in the current version. Using PyPDF2 as a reliable fallback.
        For production with scanned PDFs, consider upgrading to Mistral SDK
        with OCR support or using dedicated OCR services.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            from PyPDF2 import PdfReader

            logger.info(f"Extracting text from PDF: {os.path.basename(pdf_path)}")

            reader = PdfReader(pdf_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path} ({len(reader.pages)} pages)")

            if not full_text.strip():
                raise ValueError("No text could be extracted from PDF. The PDF might be scanned or image-based.")

            return full_text

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise


# Global client instance (singleton pattern)
_client = None


def get_mistral_client() -> MistralClient:
    """Get or create the global Mistral client instance"""
    global _client
    if _client is None:
        _client = MistralClient()
    return _client
