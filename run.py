#!/usr/bin/env python3
"""
Simple script to run the RAG System

Usage:
    python run.py
"""

import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()

    print("=" * 60)
    print("Starting RAG System")
    print("=" * 60)
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"Chat UI: http://{settings.host}:{settings.port}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True, log_level="info")
