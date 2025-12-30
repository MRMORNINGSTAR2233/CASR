"""
CASR API Package

FastAPI application for Context-Aware Secure Retrieval.
"""

from .app import create_app
from .dependencies import (
    get_current_user,
    get_embedder,
    get_index_manager,
    get_policy_engine,
    get_retriever,
    get_vector_store,
)

__all__ = [
    "create_app",
    "get_current_user",
    "get_embedder",
    "get_index_manager",
    "get_policy_engine",
    "get_retriever",
    "get_vector_store",
]
