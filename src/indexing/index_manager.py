"""
Index Manager

Orchestrates the full indexing pipeline: chunking, contextualization, embedding, and storage.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from config import get_settings
from src.models.documents import Document, DocumentChunk, DocumentMetadata

from .chunker import Chunker, ChunkingStrategy
from .contextualizer import Contextualizer, SimpleContextualizer
from .embedder import Embedder, EmbeddingProvider


class IndexManager:
    """
    Manages the document indexing pipeline.
    
    Coordinates:
    1. Document chunking
    2. Context generation
    3. Embedding generation
    4. Storage in vector store
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,  # Will be a VectorStore instance
        chunker: Optional[Chunker] = None,
        contextualizer: Optional[Contextualizer] = None,
        embedder: Optional[Embedder] = None,
        enable_contextualization: bool = True,
        use_simple_contextualizer: bool = False,
    ):
        """
        Initialize the index manager.
        
        Args:
            vector_store: Vector store for chunk storage
            chunker: Document chunker instance
            contextualizer: Contextualizer instance
            embedder: Embedder instance
            enable_contextualization: Whether to generate context
            use_simple_contextualizer: Use rule-based instead of LLM contextualizer
        """
        settings = get_settings()
        
        self.vector_store = vector_store
        self.enable_contextualization = enable_contextualization
        
        # Initialize chunker
        self.chunker = chunker or Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        
        # Initialize contextualizer
        if enable_contextualization:
            if use_simple_contextualizer:
                self.contextualizer = SimpleContextualizer()
            else:
                self.contextualizer = contextualizer or Contextualizer()
        else:
            self.contextualizer = None
        
        # Initialize embedder
        self.embedder = embedder or Embedder(
            provider=EmbeddingProvider(settings.embedding_provider.value),
        )
        
        # Index statistics
        self._stats = {
            "documents_indexed": 0,
            "chunks_created": 0,
            "last_index_time": None,
        }
    
    def index_document(
        self,
        document: Document,
        store: bool = True
    ) -> list[DocumentChunk]:
        """
        Index a single document through the full pipeline.
        
        Args:
            document: Document to index
            store: Whether to store in vector store
            
        Returns:
            List of processed document chunks
        """
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_document(document)
        
        # Step 2: Contextualize chunks
        if self.contextualizer and self.enable_contextualization:
            chunks = self.contextualizer.contextualize_chunks(document, chunks)
        
        # Step 3: Generate embeddings
        chunks = self.embedder.embed_chunks(chunks)
        
        # Step 4: Store in vector store
        if store and self.vector_store:
            self.vector_store.add_chunks(chunks)
        
        # Update document status
        document.is_indexed = True
        document.chunk_count = len(chunks)
        
        # Update statistics
        self._stats["documents_indexed"] += 1
        self._stats["chunks_created"] += len(chunks)
        self._stats["last_index_time"] = datetime.utcnow()
        
        return chunks
    
    def index_documents(
        self,
        documents: list[Document],
        store: bool = True,
        batch_size: int = 10
    ) -> dict[UUID, list[DocumentChunk]]:
        """
        Index multiple documents.
        
        Args:
            documents: List of documents to index
            store: Whether to store in vector store
            batch_size: Documents to process at a time
            
        Returns:
            Dictionary mapping document IDs to their chunks
        """
        results: dict[UUID, list[DocumentChunk]] = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for document in batch:
                try:
                    chunks = self.index_document(document, store=store)
                    results[document.id] = chunks
                except Exception as e:
                    print(f"Error indexing document {document.id}: {e}")
                    results[document.id] = []
        
        return results
    
    async def aindex_document(
        self,
        document: Document,
        store: bool = True
    ) -> list[DocumentChunk]:
        """
        Async version of index_document.
        
        Args:
            document: Document to index
            store: Whether to store in vector store
            
        Returns:
            List of processed document chunks
        """
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_document(document)
        
        # Step 2: Contextualize chunks
        if self.contextualizer and self.enable_contextualization:
            if hasattr(self.contextualizer, 'acontextualize_chunks'):
                chunks = await self.contextualizer.acontextualize_chunks(document, chunks)
            else:
                chunks = self.contextualizer.contextualize_chunks(document, chunks)
        
        # Step 3: Generate embeddings
        chunks = await self.embedder.aembed_chunks(chunks)
        
        # Step 4: Store in vector store
        if store and self.vector_store:
            if hasattr(self.vector_store, 'aadd_chunks'):
                await self.vector_store.aadd_chunks(chunks)
            else:
                self.vector_store.add_chunks(chunks)
        
        # Update document status
        document.is_indexed = True
        document.chunk_count = len(chunks)
        
        # Update statistics
        self._stats["documents_indexed"] += 1
        self._stats["chunks_created"] += len(chunks)
        self._stats["last_index_time"] = datetime.utcnow()
        
        return chunks
    
    async def aindex_documents(
        self,
        documents: list[Document],
        store: bool = True,
        max_concurrent: int = 5
    ) -> dict[UUID, list[DocumentChunk]]:
        """
        Async indexing of multiple documents with concurrency control.
        
        Args:
            documents: List of documents to index
            store: Whether to store in vector store
            max_concurrent: Maximum concurrent indexing operations
            
        Returns:
            Dictionary mapping document IDs to their chunks
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: dict[UUID, list[DocumentChunk]] = {}
        
        async def index_with_semaphore(doc: Document) -> tuple[UUID, list[DocumentChunk]]:
            async with semaphore:
                try:
                    chunks = await self.aindex_document(doc, store=store)
                    return doc.id, chunks
                except Exception as e:
                    print(f"Error indexing document {doc.id}: {e}")
                    return doc.id, []
        
        tasks = [index_with_semaphore(doc) for doc in documents]
        completed = await asyncio.gather(*tasks)
        
        for doc_id, chunks in completed:
            results[doc_id] = chunks
        
        return results
    
    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and its chunks from the index.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        if self.vector_store:
            return self.vector_store.delete_by_document_id(str(document_id))
        return False
    
    def update_document(
        self,
        document: Document
    ) -> list[DocumentChunk]:
        """
        Update an existing document (delete and re-index).
        
        Args:
            document: Updated document
            
        Returns:
            New document chunks
        """
        # Delete existing chunks
        self.delete_document(document.id)
        
        # Re-index
        return self.index_document(document)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get indexing statistics."""
        return {
            **self._stats,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "chunking_strategy": self.chunker.strategy.value,
            "embedding_provider": self.embedder.provider.value,
            "embedding_model": self.embedder.model,
            "embedding_dimension": self.embedder.dimension,
            "contextualization_enabled": self.enable_contextualization,
        }
    
    def create_document(
        self,
        content: str,
        title: str,
        source: str = "manual",
        domain: str = "general",
        classification: str = "internal",
        tags: Optional[list[str]] = None,
        **metadata_kwargs
    ) -> Document:
        """
        Helper to create a document with metadata.
        
        Args:
            content: Document content
            title: Document title
            source: Source identifier
            domain: Business domain
            classification: Security classification
            tags: Document tags
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            Created Document instance
        """
        from src.models.documents import SecurityClassification
        
        metadata = DocumentMetadata(
            title=title,
            source=source,
            domain=domain,
            classification=SecurityClassification(classification),
            tags=tags or [],
            **metadata_kwargs,
        )
        
        return Document(
            content=content,
            metadata=metadata,
        )
    
    def create_and_index(
        self,
        content: str,
        title: str,
        store: bool = True,
        **metadata_kwargs
    ) -> tuple[Document, list[DocumentChunk]]:
        """
        Create and immediately index a document.
        
        Args:
            content: Document content
            title: Document title
            store: Whether to store in vector store
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            Tuple of (document, chunks)
        """
        document = self.create_document(content=content, title=title, **metadata_kwargs)
        chunks = self.index_document(document, store=store)
        return document, chunks
