"""
ChromaDB Vector Store Implementation

Local/embedded vector store using ChromaDB.
"""

import json
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings
from src.models.documents import DocumentChunk, DocumentMetadata, SecurityClassification

from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store implementation.
    
    Suitable for local development and small-medium datasets.
    Supports persistence and metadata filtering.
    """
    
    def __init__(
        self,
        collection_name: str = "casr_chunks",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (None for in-memory)
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        
        # Initialize ChromaDB client
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
    
    def _chunk_to_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Convert chunk metadata to ChromaDB-compatible format."""
        meta = chunk.metadata
        
        # ChromaDB only supports str, int, float, bool in metadata
        return {
            "document_id": str(chunk.document_id),
            "chunk_index": chunk.chunk_index,
            "title": meta.title,
            "source": meta.source,
            "classification": meta.classification.value,
            "classification_level": meta.classification.level,
            "domain": meta.domain,
            "department": meta.department or "",
            "data_type": meta.data_type,
            "tags": json.dumps(meta.tags),  # Serialize list as JSON string
            "allowed_roles": json.dumps(meta.allowed_roles),
            "denied_roles": json.dumps(meta.denied_roles),
        }
    
    def _metadata_to_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None
    ) -> DocumentChunk:
        """Reconstruct DocumentChunk from stored data."""
        # Parse JSON fields
        tags = json.loads(metadata.get("tags", "[]"))
        allowed_roles = json.loads(metadata.get("allowed_roles", "[]"))
        denied_roles = json.loads(metadata.get("denied_roles", "[]"))
        
        doc_metadata = DocumentMetadata(
            title=metadata.get("title", ""),
            source=metadata.get("source", ""),
            classification=SecurityClassification(metadata.get("classification", "internal")),
            domain=metadata.get("domain", "general"),
            department=metadata.get("department") or None,
            data_type=metadata.get("data_type", "unstructured"),
            tags=tags,
            allowed_roles=allowed_roles,
            denied_roles=denied_roles,
        )
        
        return DocumentChunk(
            id=UUID(chunk_id),
            document_id=UUID(metadata.get("document_id")),
            content=content,
            chunk_index=metadata.get("chunk_index", 0),
            start_char=0,  # Not stored in ChromaDB
            end_char=len(content),
            metadata=doc_metadata,
            embedding=embedding,
        )
    
    def _convert_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Convert generic filters to ChromaDB where clause."""
        if not filters:
            return {}
        
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operators like {"$in": [...], "$lte": 5}
                for op, op_value in value.items():
                    if op == "$in":
                        where_clause[key] = {"$in": op_value}
                    elif op == "$nin":
                        where_clause[key] = {"$nin": op_value}
                    elif op == "$eq":
                        where_clause[key] = {"$eq": op_value}
                    elif op == "$ne":
                        where_clause[key] = {"$ne": op_value}
                    elif op == "$gt":
                        where_clause[key] = {"$gt": op_value}
                    elif op == "$gte":
                        where_clause[key] = {"$gte": op_value}
                    elif op == "$lt":
                        where_clause[key] = {"$lt": op_value}
                    elif op == "$lte":
                        where_clause[key] = {"$lte": op_value}
            else:
                where_clause[key] = {"$eq": value}
        
        return where_clause
    
    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to ChromaDB."""
        if not chunks:
            return []
        
        ids = [str(chunk.id) for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content_for_embedding for chunk in chunks]
        metadatas = [self._chunk_to_metadata(chunk) for chunk in chunks]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        return ids
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks in ChromaDB."""
        where_clause = self._convert_filters(filters) if filters else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        
        chunks_with_scores: list[tuple[DocumentChunk, float]] = []
        
        if not results["ids"] or not results["ids"][0]:
            return chunks_with_scores
        
        for i, chunk_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i] if results["documents"] else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            embedding = results["embeddings"][0][i] if results.get("embeddings") else None
            
            # Convert distance to similarity score (ChromaDB returns distance)
            # For cosine distance: similarity = 1 - distance
            score = 1.0 - distance
            
            chunk = self._metadata_to_chunk(chunk_id, content, metadata, embedding)
            chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            self._collection.delete(
                where={"document_id": {"$eq": document_id}}
            )
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk."""
        try:
            self._collection.delete(ids=[chunk_id])
            return True
        except Exception as e:
            print(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        try:
            results = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"],
            )
            
            if not results["ids"]:
                return None
            
            content = results["documents"][0] if results["documents"] else ""
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            embedding = results["embeddings"][0] if results.get("embeddings") else None
            
            return self._metadata_to_chunk(chunk_id, content, metadata, embedding)
        except Exception:
            return None
    
    def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document."""
        results = self._collection.get(
            where={"document_id": {"$eq": document_id}},
            include=["documents", "metadatas", "embeddings"],
        )
        
        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            content = results["documents"][i] if results["documents"] else ""
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            embedding = results["embeddings"][i] if results.get("embeddings") else None
            
            chunk = self._metadata_to_chunk(chunk_id, content, metadata, embedding)
            chunks.append(chunk)
        
        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks
    
    def count(self) -> int:
        """Get total number of chunks."""
        return self._collection.count()
    
    def list_documents(self) -> list[str]:
        """List all unique document IDs in the store."""
        results = self._collection.get(
            include=["metadatas"],
        )
        
        doc_ids = set()
        for metadata in results["metadatas"]:
            if "document_id" in metadata:
                doc_ids.add(metadata["document_id"])
        
        return list(doc_ids)
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
        )
