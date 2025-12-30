"""
Pinecone Vector Store Implementation

Cloud-hosted vector store using Pinecone.
"""

import json
from typing import Any, Optional
from uuid import UUID

from config import get_settings
from src.models.documents import DocumentChunk, DocumentMetadata, SecurityClassification

from .base import VectorStore


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector store implementation.
    
    Cloud-hosted, scalable vector database with metadata filtering.
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: str = "default",
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            dimension: Vector dimension
            metric: Distance metric
        """
        from pinecone import Pinecone, ServerlessSpec
        
        settings = get_settings()
        self.index_name = index_name or settings.pinecone_index_name
        self.namespace = namespace
        self.dimension = dimension
        
        # Initialize Pinecone client
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            self._pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Get index reference
        self._index = self._pc.Index(self.index_name)
    
    def _chunk_to_vector(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Convert chunk to Pinecone vector format."""
        meta = chunk.metadata
        
        return {
            "id": str(chunk.id),
            "values": chunk.embedding,
            "metadata": {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content[:1000],  # Pinecone metadata size limit
                "contextualized_content": (chunk.contextualized_content or "")[:1000],
                "title": meta.title[:200],
                "source": meta.source,
                "classification": meta.classification.value,
                "classification_level": meta.classification.level,
                "domain": meta.domain,
                "department": meta.department or "",
                "data_type": meta.data_type,
                "tags": meta.tags[:10],  # Limit tags
                "allowed_roles": meta.allowed_roles,
                "denied_roles": meta.denied_roles,
            }
        }
    
    def _vector_to_chunk(
        self,
        match: dict[str, Any]
    ) -> DocumentChunk:
        """Convert Pinecone match to DocumentChunk."""
        metadata = match.get("metadata", {})
        
        doc_metadata = DocumentMetadata(
            title=metadata.get("title", ""),
            source=metadata.get("source", ""),
            classification=SecurityClassification(metadata.get("classification", "internal")),
            domain=metadata.get("domain", "general"),
            department=metadata.get("department") or None,
            data_type=metadata.get("data_type", "unstructured"),
            tags=metadata.get("tags", []),
            allowed_roles=metadata.get("allowed_roles", []),
            denied_roles=metadata.get("denied_roles", []),
        )
        
        content = metadata.get("content", "")
        contextualized = metadata.get("contextualized_content")
        
        return DocumentChunk(
            id=UUID(match["id"]),
            document_id=UUID(metadata.get("document_id")),
            content=content,
            contextualized_content=contextualized if contextualized else None,
            chunk_index=metadata.get("chunk_index", 0),
            start_char=0,
            end_char=len(content),
            metadata=doc_metadata,
            embedding=match.get("values"),
        )
    
    def _convert_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Convert generic filters to Pinecone filter format."""
        if not filters:
            return {}
        
        pinecone_filter = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in":
                        pinecone_filter[key] = {"$in": op_value}
                    elif op == "$nin":
                        pinecone_filter[key] = {"$nin": op_value}
                    elif op == "$eq":
                        pinecone_filter[key] = {"$eq": op_value}
                    elif op == "$ne":
                        pinecone_filter[key] = {"$ne": op_value}
                    elif op == "$gt":
                        pinecone_filter[key] = {"$gt": op_value}
                    elif op == "$gte":
                        pinecone_filter[key] = {"$gte": op_value}
                    elif op == "$lt":
                        pinecone_filter[key] = {"$lt": op_value}
                    elif op == "$lte":
                        pinecone_filter[key] = {"$lte": op_value}
            else:
                pinecone_filter[key] = {"$eq": value}
        
        return pinecone_filter
    
    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to Pinecone."""
        if not chunks:
            return []
        
        vectors = [self._chunk_to_vector(chunk) for chunk in chunks]
        
        # Upsert in batches (Pinecone limit is 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)
        
        return [str(chunk.id) for chunk in chunks]
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks in Pinecone."""
        pinecone_filter = self._convert_filters(filters) if filters else None
        
        results = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=pinecone_filter,
            include_metadata=True,
            include_values=True,
        )
        
        chunks_with_scores: list[tuple[DocumentChunk, float]] = []
        
        for match in results.matches:
            chunk = self._vector_to_chunk({
                "id": match.id,
                "values": match.values,
                "metadata": match.metadata,
            })
            chunks_with_scores.append((chunk, match.score))
        
        return chunks_with_scores
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # First, find all chunks for this document
            # Pinecone requires knowing the IDs to delete
            results = self._index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,
                namespace=self.namespace,
                filter={"document_id": {"$eq": document_id}},
                include_metadata=False,
            )
            
            if results.matches:
                ids_to_delete = [match.id for match in results.matches]
                self._index.delete(ids=ids_to_delete, namespace=self.namespace)
            
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk."""
        try:
            self._index.delete(ids=[chunk_id], namespace=self.namespace)
            return True
        except Exception as e:
            print(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        try:
            results = self._index.fetch(
                ids=[chunk_id],
                namespace=self.namespace,
            )
            
            if chunk_id in results.vectors:
                vector = results.vectors[chunk_id]
                return self._vector_to_chunk({
                    "id": chunk_id,
                    "values": vector.values,
                    "metadata": vector.metadata,
                })
            
            return None
        except Exception:
            return None
    
    def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document."""
        results = self._index.query(
            vector=[0.0] * self.dimension,  # Dummy vector
            top_k=10000,
            namespace=self.namespace,
            filter={"document_id": {"$eq": document_id}},
            include_metadata=True,
            include_values=True,
        )
        
        chunks = []
        for match in results.matches:
            chunk = self._vector_to_chunk({
                "id": match.id,
                "values": match.values,
                "metadata": match.metadata,
            })
            chunks.append(chunk)
        
        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks
    
    def count(self) -> int:
        """Get total number of chunks."""
        stats = self._index.describe_index_stats()
        namespace_stats = stats.namespaces.get(self.namespace, {})
        return namespace_stats.get("vector_count", 0)
    
    def clear_namespace(self) -> None:
        """Clear all data from the namespace."""
        self._index.delete(delete_all=True, namespace=self.namespace)
