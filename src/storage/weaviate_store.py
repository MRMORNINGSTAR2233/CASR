"""
Weaviate Vector Store Implementation

Cloud/self-hosted vector store using Weaviate.
"""

from typing import Any, Optional
from uuid import UUID

from config import get_settings
from src.models.documents import DocumentChunk, DocumentMetadata, SecurityClassification

from .base import VectorStore


class WeaviateVectorStore(VectorStore):
    """
    Weaviate vector store implementation.
    
    Supports both cloud and self-hosted Weaviate instances.
    Features native hybrid search and GraphQL queries.
    """
    
    # Schema for the document chunk class
    CHUNK_SCHEMA = {
        "class": "DocumentChunk",
        "description": "A chunk of a document with metadata",
        "vectorizer": "none",  # We provide our own vectors
        "properties": [
            {"name": "chunkId", "dataType": ["text"]},
            {"name": "documentId", "dataType": ["text"]},
            {"name": "content", "dataType": ["text"]},
            {"name": "contextualizedContent", "dataType": ["text"]},
            {"name": "chunkIndex", "dataType": ["int"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "classification", "dataType": ["text"]},
            {"name": "classificationLevel", "dataType": ["int"]},
            {"name": "domain", "dataType": ["text"]},
            {"name": "department", "dataType": ["text"]},
            {"name": "dataType", "dataType": ["text"]},
            {"name": "tags", "dataType": ["text[]"]},
            {"name": "allowedRoles", "dataType": ["text[]"]},
            {"name": "deniedRoles", "dataType": ["text[]"]},
        ],
    }
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "DocumentChunk"
    ):
        """
        Initialize Weaviate vector store.
        
        Args:
            url: Weaviate server URL
            api_key: Weaviate API key (for cloud)
            class_name: Name of the Weaviate class
        """
        import weaviate
        from weaviate.classes.init import Auth
        
        settings = get_settings()
        self.url = url or settings.weaviate_url
        self.api_key = api_key or settings.weaviate_api_key
        self.class_name = class_name
        
        # Initialize Weaviate client
        if self.api_key:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
            )
        else:
            self._client = weaviate.connect_to_local(
                host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(self.url.split(":")[-1]) if ":" in self.url.split("/")[-1] else 8080,
            )
        
        # Get or create collection
        self._ensure_schema()
        self._collection = self._client.collections.get(self.class_name)
    
    def _ensure_schema(self) -> None:
        """Ensure the schema exists in Weaviate."""
        try:
            # Check if collection exists
            if not self._client.collections.exists(self.class_name):
                from weaviate.classes.config import Configure, Property, DataType
                
                self._client.collections.create(
                    name=self.class_name,
                    description="A chunk of a document with metadata",
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="chunkId", data_type=DataType.TEXT),
                        Property(name="documentId", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="contextualizedContent", data_type=DataType.TEXT),
                        Property(name="chunkIndex", data_type=DataType.INT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="classification", data_type=DataType.TEXT),
                        Property(name="classificationLevel", data_type=DataType.INT),
                        Property(name="domain", data_type=DataType.TEXT),
                        Property(name="department", data_type=DataType.TEXT),
                        Property(name="dataType", data_type=DataType.TEXT),
                        Property(name="tags", data_type=DataType.TEXT_ARRAY),
                        Property(name="allowedRoles", data_type=DataType.TEXT_ARRAY),
                        Property(name="deniedRoles", data_type=DataType.TEXT_ARRAY),
                    ],
                )
        except Exception as e:
            print(f"Warning: Could not ensure schema: {e}")
    
    def _chunk_to_properties(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Convert chunk to Weaviate properties."""
        meta = chunk.metadata
        
        return {
            "chunkId": str(chunk.id),
            "documentId": str(chunk.document_id),
            "content": chunk.content,
            "contextualizedContent": chunk.contextualized_content or "",
            "chunkIndex": chunk.chunk_index,
            "title": meta.title,
            "source": meta.source,
            "classification": meta.classification.value,
            "classificationLevel": meta.classification.level,
            "domain": meta.domain,
            "department": meta.department or "",
            "dataType": meta.data_type,
            "tags": meta.tags,
            "allowedRoles": meta.allowed_roles,
            "deniedRoles": meta.denied_roles,
        }
    
    def _properties_to_chunk(
        self,
        properties: dict[str, Any],
        embedding: Optional[list[float]] = None
    ) -> DocumentChunk:
        """Convert Weaviate properties to DocumentChunk."""
        doc_metadata = DocumentMetadata(
            title=properties.get("title", ""),
            source=properties.get("source", ""),
            classification=SecurityClassification(properties.get("classification", "internal")),
            domain=properties.get("domain", "general"),
            department=properties.get("department") or None,
            data_type=properties.get("dataType", "unstructured"),
            tags=properties.get("tags", []),
            allowed_roles=properties.get("allowedRoles", []),
            denied_roles=properties.get("deniedRoles", []),
        )
        
        content = properties.get("content", "")
        contextualized = properties.get("contextualizedContent")
        
        return DocumentChunk(
            id=UUID(properties.get("chunkId")),
            document_id=UUID(properties.get("documentId")),
            content=content,
            contextualized_content=contextualized if contextualized else None,
            chunk_index=properties.get("chunkIndex", 0),
            start_char=0,
            end_char=len(content),
            metadata=doc_metadata,
            embedding=embedding,
        )
    
    def _build_filter(self, filters: dict[str, Any]) -> Any:
        """Build Weaviate filter from generic filter dict."""
        from weaviate.classes.query import Filter
        
        if not filters:
            return None
        
        conditions = []
        
        for key, value in filters.items():
            # Map our keys to Weaviate property names
            prop_map = {
                "classification": "classification",
                "classification_level": "classificationLevel",
                "domain": "domain",
                "document_id": "documentId",
            }
            weaviate_key = prop_map.get(key, key)
            
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in":
                        conditions.append(
                            Filter.by_property(weaviate_key).contains_any(op_value)
                        )
                    elif op == "$eq":
                        conditions.append(
                            Filter.by_property(weaviate_key).equal(op_value)
                        )
                    elif op == "$lte":
                        conditions.append(
                            Filter.by_property(weaviate_key).less_or_equal(op_value)
                        )
                    elif op == "$gte":
                        conditions.append(
                            Filter.by_property(weaviate_key).greater_or_equal(op_value)
                        )
            else:
                conditions.append(
                    Filter.by_property(weaviate_key).equal(value)
                )
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            result = conditions[0]
            for cond in conditions[1:]:
                result = result & cond
            return result
        
        return None
    
    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Add document chunks to Weaviate."""
        if not chunks:
            return []
        
        ids = []
        
        with self._collection.batch.dynamic() as batch:
            for chunk in chunks:
                properties = self._chunk_to_properties(chunk)
                batch.add_object(
                    properties=properties,
                    vector=chunk.embedding,
                    uuid=chunk.id,
                )
                ids.append(str(chunk.id))
        
        return ids
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks in Weaviate."""
        weaviate_filter = self._build_filter(filters) if filters else None
        
        results = self._collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=weaviate_filter,
            include_vector=True,
            return_metadata=["distance"],
        )
        
        chunks_with_scores: list[tuple[DocumentChunk, float]] = []
        
        for obj in results.objects:
            properties = {k: v for k, v in obj.properties.items()}
            embedding = obj.vector.get("default") if obj.vector else None
            chunk = self._properties_to_chunk(properties, embedding)
            
            # Convert distance to similarity
            distance = obj.metadata.distance if obj.metadata else 0
            score = 1.0 - distance
            
            chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """Hybrid search combining vector and BM25 keyword search."""
        weaviate_filter = self._build_filter(filters) if filters else None
        
        results = self._collection.query.hybrid(
            query=query_text,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            filters=weaviate_filter,
            include_vector=True,
            return_metadata=["score"],
        )
        
        chunks_with_scores: list[tuple[DocumentChunk, float]] = []
        
        for obj in results.objects:
            properties = {k: v for k, v in obj.properties.items()}
            embedding = obj.vector.get("default") if obj.vector else None
            chunk = self._properties_to_chunk(properties, embedding)
            
            score = obj.metadata.score if obj.metadata else 0
            chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            from weaviate.classes.query import Filter
            
            self._collection.data.delete_many(
                where=Filter.by_property("documentId").equal(document_id)
            )
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk."""
        try:
            self._collection.data.delete_by_id(UUID(chunk_id))
            return True
        except Exception as e:
            print(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        try:
            obj = self._collection.query.fetch_object_by_id(
                UUID(chunk_id),
                include_vector=True,
            )
            
            if obj:
                properties = {k: v for k, v in obj.properties.items()}
                embedding = obj.vector.get("default") if obj.vector else None
                return self._properties_to_chunk(properties, embedding)
            
            return None
        except Exception:
            return None
    
    def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document."""
        from weaviate.classes.query import Filter
        
        results = self._collection.query.fetch_objects(
            filters=Filter.by_property("documentId").equal(document_id),
            include_vector=True,
        )
        
        chunks = []
        for obj in results.objects:
            properties = {k: v for k, v in obj.properties.items()}
            embedding = obj.vector.get("default") if obj.vector else None
            chunk = self._properties_to_chunk(properties, embedding)
            chunks.append(chunk)
        
        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks
    
    def count(self) -> int:
        """Get total number of chunks."""
        aggregate = self._collection.aggregate.over_all(total_count=True)
        return aggregate.total_count if aggregate else 0
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.collections.delete(self.class_name)
        self._ensure_schema()
        self._collection = self._client.collections.get(self.class_name)
    
    def close(self) -> None:
        """Close the Weaviate client connection."""
        self._client.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
