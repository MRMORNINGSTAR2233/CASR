"""
Document Models

Defines the document structure, metadata, and security classifications
for the CASR system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SecurityClassification(str, Enum):
    """
    Security classification levels for documents.
    
    Follows a hierarchical model where higher levels can access lower levels.
    """
    PUBLIC = "public"              # Accessible by all authenticated users
    INTERNAL = "internal"          # Accessible by internal employees
    CONFIDENTIAL = "confidential"  # Accessible by authorized personnel
    SECRET = "secret"              # Accessible by specific roles only
    TOP_SECRET = "top_secret"      # Highest classification, need-to-know basis
    
    @property
    def level(self) -> int:
        """Return numeric level for comparison."""
        levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2,
            "secret": 3,
            "top_secret": 4,
        }
        return levels[self.value]
    
    def can_access(self, user_clearance: "SecurityClassification") -> bool:
        """Check if a user with given clearance can access this classification."""
        return user_clearance.level >= self.level


class DocumentMetadata(BaseModel):
    """
    Metadata associated with a document.
    
    Contains information for filtering, access control, and contextual retrieval.
    """
    
    # Core metadata
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Source system or origin")
    author: Optional[str] = Field(default=None, description="Document author")
    
    # Temporal metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Classification and access
    classification: SecurityClassification = Field(
        default=SecurityClassification.INTERNAL,
        description="Security classification level"
    )
    
    # Domain and categorization
    domain: str = Field(
        default="general",
        description="Business domain (e.g., finance, legal, engineering)"
    )
    department: Optional[str] = Field(
        default=None,
        description="Owning department"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorical tags for filtering"
    )
    
    # Data type information
    data_type: str = Field(
        default="unstructured",
        description="Type of data: structured, unstructured, semi-structured"
    )
    file_type: Optional[str] = Field(
        default=None,
        description="Original file type (pdf, docx, etc.)"
    )
    
    # Access control
    allowed_roles: list[str] = Field(
        default_factory=list,
        description="Roles explicitly allowed access"
    )
    denied_roles: list[str] = Field(
        default_factory=list,
        description="Roles explicitly denied access"
    )
    allowed_users: list[str] = Field(
        default_factory=list,
        description="User IDs explicitly allowed access"
    )
    
    # Additional custom metadata
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )
    
    def to_filter_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for vector store filtering."""
        return {
            "classification": self.classification.value,
            "domain": self.domain,
            "department": self.department,
            "tags": self.tags,
            "data_type": self.data_type,
            "allowed_roles": self.allowed_roles,
            "denied_roles": self.denied_roles,
        }


class Document(BaseModel):
    """
    A document in the CASR system.
    
    Represents a complete document before chunking.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique document ID")
    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Processing state
    is_indexed: bool = Field(default=False, description="Whether document is indexed")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class DocumentChunk(BaseModel):
    """
    A chunk of a document with contextual information.
    
    Created during the indexing process with added context for better retrieval.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique chunk ID")
    document_id: UUID = Field(..., description="Parent document ID")
    
    # Content
    content: str = Field(..., description="Original chunk content")
    contextualized_content: Optional[str] = Field(
        default=None,
        description="Content with prepended context"
    )
    
    # Position information
    chunk_index: int = Field(..., description="Index of chunk within document")
    start_char: int = Field(..., description="Start character position in original")
    end_char: int = Field(..., description="End character position in original")
    
    # Inherited metadata (denormalized for efficient filtering)
    metadata: DocumentMetadata = Field(..., description="Inherited document metadata")
    
    # Embedding
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding of the chunk"
    )
    
    # Context generation metadata
    context_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated context summary"
    )
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    @property
    def content_for_embedding(self) -> str:
        """Return the content to be used for embedding."""
        return self.contextualized_content or self.content
    
    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for vector store storage."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "content": self.content,
            "contextualized_content": self.contextualized_content,
            "chunk_index": self.chunk_index,
            "context_summary": self.context_summary,
            **self.metadata.to_filter_dict(),
        }
