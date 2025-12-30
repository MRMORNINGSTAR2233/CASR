"""
Vector Store Base Class

Abstract base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from src.models.documents import DocumentChunk


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Defines the interface for storing and retrieving document chunks
    with their embeddings and metadata.
    """
    
    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            List of chunk IDs that were stored
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of (chunk, score) tuples ordered by similarity
        """
        pass
    
    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk.
        
        Args:
            chunk_id: ID of the chunk to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a single chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            The document chunk or None if not found
        """
        pass
    
    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get total number of chunks in the store.
        
        Returns:
            Total chunk count
        """
        pass
    
    def search_with_text(
        self,
        query_text: str,
        embedder: Any,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search using text query (embeds the query first).
        
        Args:
            query_text: Text query
            embedder: Embedder instance
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of (chunk, score) tuples
        """
        query_embedding = embedder.embed_query(query_text)
        return self.search(query_embedding, top_k, filters)
    
    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Hybrid search combining vector and keyword search.
        
        Default implementation just does vector search.
        Override in implementations that support hybrid search.
        
        Args:
            query_embedding: Query vector
            query_text: Query text for keyword search
            top_k: Number of results
            filters: Metadata filters
            alpha: Weight for vector vs keyword (0=keyword, 1=vector)
            
        Returns:
            List of (chunk, score) tuples
        """
        return self.search(query_embedding, top_k, filters)
    
    async def aadd_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Async version of add_chunks. Default implementation is sync."""
        return self.add_chunks(chunks)
    
    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Async version of search. Default implementation is sync."""
        return self.search(query_embedding, top_k, filters)
