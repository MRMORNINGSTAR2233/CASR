"""
Query Models

Defines query structure, context, and retrieval results for the CASR system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .documents import DocumentChunk, SecurityClassification


class QueryIntent(str, Enum):
    """
    Detected intent categories for queries.
    
    Used to route queries to appropriate retrieval strategies.
    """
    FACTUAL = "factual"              # Looking for specific facts
    ANALYTICAL = "analytical"         # Requires analysis or comparison
    PROCEDURAL = "procedural"         # How-to or process questions
    DEFINITIONAL = "definitional"     # What is X?
    EXPLORATORY = "exploratory"       # Open-ended exploration
    TROUBLESHOOTING = "troubleshooting"  # Problem-solving queries
    SUMMARIZATION = "summarization"   # Summarize content
    UNKNOWN = "unknown"               # Could not determine intent


class DataTypeRequirement(str, Enum):
    """Types of data that may be required to answer a query."""
    STRUCTURED = "structured"         # Database/tabular data
    UNSTRUCTURED = "unstructured"     # Documents, text
    SEMI_STRUCTURED = "semi-structured"  # JSON, XML, logs
    MIXED = "mixed"                   # Multiple types needed
    ANY = "any"                       # No specific requirement


class QueryContext(BaseModel):
    """
    Context information for a query.
    
    Contains user context, session history, and domain information
    to enable contextual retrieval.
    """
    
    # User context
    user_id: UUID = Field(..., description="Querying user ID")
    user_role: str = Field(..., description="User's role")
    user_clearance: SecurityClassification = Field(
        ...,
        description="User's security clearance"
    )
    user_department: Optional[str] = Field(default=None)
    user_allowed_domains: list[str] = Field(default_factory=list)
    
    # Session context
    session_id: Optional[UUID] = Field(default=None)
    session_domain: Optional[str] = Field(
        default=None,
        description="Current domain context from session"
    )
    session_history: list[str] = Field(
        default_factory=list,
        description="Recent queries from session"
    )
    recent_topics: list[str] = Field(
        default_factory=list,
        description="Recently discussed topics"
    )
    
    # Temporal context
    query_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Custom attributes for policy evaluation
    custom_attributes: dict[str, Any] = Field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompts."""
        lines = [
            f"User Role: {self.user_role}",
            f"Clearance Level: {self.user_clearance.value}",
        ]
        
        if self.user_department:
            lines.append(f"Department: {self.user_department}")
        
        if self.session_domain:
            lines.append(f"Current Domain: {self.session_domain}")
        
        if self.recent_topics:
            lines.append(f"Recent Topics: {', '.join(self.recent_topics[-3:])}")
        
        return "\n".join(lines)


class SearchQuery(BaseModel):
    """
    A search query submitted to the CASR system.
    
    Contains the raw query text and associated context.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Query ID")
    text: str = Field(..., min_length=1, description="Query text")
    context: QueryContext = Field(..., description="Query context")
    
    # Optional hints from user
    preferred_domain: Optional[str] = Field(
        default=None,
        description="User-specified domain preference"
    )
    preferred_data_type: Optional[DataTypeRequirement] = Field(
        default=None,
        description="User-specified data type preference"
    )
    
    # Query parameters
    max_results: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = Field(default=True)
    
    # Timestamps
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class AnalyzedQuery(BaseModel):
    """
    Result of query analysis by the Query Analyzer.
    
    Contains extracted intent, entities, and retrieval parameters.
    """
    
    original_query: str = Field(..., description="Original query text")
    
    # Intent analysis
    intent: QueryIntent = Field(..., description="Detected query intent")
    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in intent detection"
    )
    
    # Query reformulation
    reformulated_query: str = Field(
        ...,
        description="Reformulated query for better retrieval"
    )
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Query expansions for broader search"
    )
    
    # Extracted information
    entities: list[dict[str, str]] = Field(
        default_factory=list,
        description="Extracted entities with type"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Important keywords"
    )
    
    # Domain detection
    detected_domains: list[str] = Field(
        default_factory=list,
        description="Detected relevant domains"
    )
    primary_domain: Optional[str] = Field(
        default=None,
        description="Most likely relevant domain"
    )
    
    # Data requirements
    data_type_required: DataTypeRequirement = Field(
        default=DataTypeRequirement.ANY,
        description="Type of data needed"
    )
    
    # Ambiguity detection
    is_ambiguous: bool = Field(
        default=False,
        description="Whether query is ambiguous"
    )
    ambiguity_type: Optional[str] = Field(
        default=None,
        description="Type of ambiguity if detected"
    )
    disambiguation_options: list[str] = Field(
        default_factory=list,
        description="Possible interpretations"
    )
    
    # Security requirements
    minimum_clearance_needed: Optional[SecurityClassification] = Field(
        default=None,
        description="Estimated minimum clearance for relevant docs"
    )
    
    # Retrieval strategy
    recommended_top_k: int = Field(
        default=20,
        description="Recommended number of chunks to retrieve"
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Whether to use hybrid (semantic + keyword) search"
    )
    
    # Metadata filters to apply
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filters to apply during retrieval"
    )
    
    # Analysis metadata
    analysis_time_ms: float = Field(
        default=0.0,
        description="Time taken for analysis in milliseconds"
    )


class RetrievalResult(BaseModel):
    """
    Result of a retrieval operation.
    
    Contains retrieved chunks with scores and metadata.
    """
    
    query_id: UUID = Field(..., description="Associated query ID")
    
    # Retrieved chunks
    chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="Retrieved document chunks"
    )
    scores: list[float] = Field(
        default_factory=list,
        description="Relevance scores for each chunk"
    )
    
    # Retrieval metadata
    total_candidates: int = Field(
        default=0,
        description="Total candidates before filtering"
    )
    filtered_by_security: int = Field(
        default=0,
        description="Number filtered by security policies"
    )
    filtered_by_domain: int = Field(
        default=0,
        description="Number filtered by domain restrictions"
    )
    
    # Reranking information
    was_reranked: bool = Field(default=False)
    rerank_model: Optional[str] = Field(default=None)
    
    # Timing
    retrieval_time_ms: float = Field(default=0.0)
    rerank_time_ms: float = Field(default=0.0)
    total_time_ms: float = Field(default=0.0)
    
    # Sources summary
    unique_documents: int = Field(
        default=0,
        description="Number of unique source documents"
    )
    domains_represented: list[str] = Field(
        default_factory=list,
        description="Domains represented in results"
    )
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    def get_context_text(self, max_chunks: int = 5) -> str:
        """Generate context text from top chunks for LLM."""
        context_parts = []
        for i, chunk in enumerate(self.chunks[:max_chunks]):
            context_parts.append(
                f"[Source {i+1}: {chunk.metadata.title}]\n{chunk.content}"
            )
        return "\n\n".join(context_parts)
    
    @property
    def top_chunk(self) -> Optional[DocumentChunk]:
        """Get the highest-scoring chunk."""
        return self.chunks[0] if self.chunks else None
    
    @property
    def average_score(self) -> float:
        """Calculate average relevance score."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
