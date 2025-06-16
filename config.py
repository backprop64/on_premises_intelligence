"""
Configuration file for On-Premises Intelligence RAG System

This file contains all configurable parameters for the RAG pipeline.
Modify these values to adjust system behavior without changing code.
"""

from typing import Literal

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Language Model Settings
MODEL_TYPE: Literal["text", "vision"] = "text"  # "text" for SmolLM2, "vision" for SmolVLM2

# Text-only model sizes (SmolLM2)
TEXT_MODEL_SIZE: Literal["135m", "360m", "1.7b"] = "360m"  # Use larger 1.7B variant by default

# Vision model sizes (SmolVLM2) 
VISION_MODEL_SIZE: Literal["256m", "500m", "2.2b"] = "256m"  # For vision tasks

MODEL_DEVICE: str = "cpu"  # Device to run the model on
MAX_TOKENS: int = 1024  # Maximum tokens to generate 

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Document Retrieval Settings
RETRIEVAL_K: int = 5  # Number of documents to retrieve (reduced for small model)
RETRIEVAL_SCORE_THRESHOLD: float = 0.0  # Minimum similarity score for retrieval

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Text Chunking Settings
CHUNK_SIZE: int = 300  # Size of text chunks in tokens 
CHUNK_OVERLAP: int = 60  # Overlap between consecutive chunks in tokens
CHUNK_METHOD: Literal["token", "sentence", "paragraph"] = "token"

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Timeout Settings
SERVER_TIMEOUT: int = 120  # Server response timeout in seconds
CLIENT_TIMEOUT: int = 120  # Client-side timeout in seconds (frontend)

# Memory and Processing
MAX_CONTEXT_LENGTH: int = 8192  # Maximum context length now matches SmolLM2 default
ENABLE_STREAMING: bool = True  # Enable streaming responses
ENABLE_DEBUG: bool = True  # Temporarily enabled for debugging

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Server Settings
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
ENABLE_CORS: bool = True

# Upload Settings
MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".bmp"]

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Vector Database Settings
VECTOR_DB_PATH: str = "src/database/vector_store"
ENABLE_PERSISTENCE: bool = True
INDEX_TYPE: str = "faiss"  # Vector index type

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Text Embedding Settings
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
NORMALIZE_EMBEDDINGS: bool = True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_config() -> dict:
    """Get model configuration as a dictionary."""
    if MODEL_TYPE == "text":
        return {
            "type": "text",
            "size": TEXT_MODEL_SIZE,
            "device": MODEL_DEVICE,
            "max_tokens": MAX_TOKENS,
        }
    else:  # vision
        return {
            "type": "vision", 
            "size": VISION_MODEL_SIZE,
            "device": MODEL_DEVICE,
            "max_tokens": MAX_TOKENS,
        }

def get_retrieval_config() -> dict:
    """Get retrieval configuration as a dictionary."""
    return {
        "k": RETRIEVAL_K,
        "score_threshold": RETRIEVAL_SCORE_THRESHOLD,
    }

def get_chunking_config() -> dict:
    """Get chunking configuration as a dictionary."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "method": CHUNK_METHOD,
    }

def get_performance_config() -> dict:
    """Get performance configuration as a dictionary."""
    return {
        "server_timeout": SERVER_TIMEOUT,
        "client_timeout": CLIENT_TIMEOUT,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "enable_streaming": ENABLE_STREAMING,
        "enable_debug": ENABLE_DEBUG,
    }

def print_config() -> None:
    """Print current configuration for debugging."""
    print("=" * 60)
    print("🔧 CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"Model Type: {MODEL_TYPE}")
    if MODEL_TYPE == "text":
        print(f"Text Model Size: {TEXT_MODEL_SIZE}")
    else:
        print(f"Vision Model Size: {VISION_MODEL_SIZE}")
    print(f"Model Device: {MODEL_DEVICE}")
    print(f"Max Tokens: {MAX_TOKENS}")
    print(f"Retrieval K: {RETRIEVAL_K}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"Server Timeout: {SERVER_TIMEOUT}s")
    print(f"Client Timeout: {CLIENT_TIMEOUT}s")
    print(f"Streaming Enabled: {ENABLE_STREAMING}")
    print(f"Debug Enabled: {ENABLE_DEBUG}")
    print("=" * 60)

if __name__ == "__main__":
    print_config() 