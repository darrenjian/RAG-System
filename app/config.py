"""App settings and configuration."""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """App settings from environment variables."""

    # Mistral API Configuration
    mistral_api_key: str

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k_results: int = 5
    top_n_semantic: int = 10  # Top N results from semantic search
    top_k_keyword: int = 10   # Top K results from keyword search
    similarity_threshold: float = 0.7
    max_context_length: int = 4000

    # Model Configuration
    embedding_model: str = "mistral-embed"
    llm_model: str = "mistral-small-latest"
    ocr_model: str = "mistral-ocr-latest"

    # Hybrid Search Weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
