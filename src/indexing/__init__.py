"""
CASR Indexing Package

Document processing, chunking, contextualization, and embedding.
"""

from .chunker import Chunker, ChunkingStrategy
from .contextualizer import Contextualizer
from .embedder import Embedder, EmbeddingProvider
from .index_manager import IndexManager

__all__ = [
    "Chunker",
    "ChunkingStrategy",
    "Contextualizer",
    "Embedder",
    "EmbeddingProvider",
    "IndexManager",
]
