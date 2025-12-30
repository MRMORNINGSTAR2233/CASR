"""
Query Routes

Search and retrieval endpoints.
"""

import time
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_current_user,
    get_retriever,
)
from src.models.documents import SecurityClassification
from src.models.queries import (
    AnalyzedQuery,
    QueryContext,
    RetrievalResult,
    SearchQuery,
)
from src.models.users import User
from src.retrieval import SecureRetriever

router = APIRouter()


# Request/Response Models
class SearchRequest(BaseModel):
    """Search query request."""
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain: Optional[str] = None
    classification_max: Optional[str] = None
    use_reranking: bool = True
    use_hybrid: bool = True


class ChunkResult(BaseModel):
    """Single chunk result."""
    chunk_id: str
    document_id: str
    content: str
    context: Optional[str]
    score: float
    title: str
    source: str
    domain: str
    classification: str


class SearchResponse(BaseModel):
    """Search response."""
    query_id: UUID
    query: str
    results: list[ChunkResult]
    total_candidates: int
    filtered_by_security: int
    was_reranked: bool
    retrieval_time_ms: float
    total_time_ms: float


class QueryAnalysisResponse(BaseModel):
    """Query analysis response."""
    original_query: str
    intent: str
    intent_confidence: float
    reformulated_query: str
    expanded_queries: list[str]
    entities: list[dict]
    keywords: list[str]
    detected_domains: list[str]
    is_ambiguous: bool
    disambiguation_options: list[str]
    recommended_top_k: int
    use_hybrid_search: bool
    analysis_time_ms: float


class QuickSearchRequest(BaseModel):
    """Simple search request."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=50)


class QuickSearchResponse(BaseModel):
    """Simple search response."""
    results: list[ChunkResult]
    count: int
    time_ms: float


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    retriever: SecureRetriever = Depends(get_retriever),
) -> SearchResponse:
    """
    Perform a secure search with full analysis pipeline.
    
    The search will:
    1. Analyze the query for intent and entities
    2. Apply security filters based on user permissions
    3. Search across document chunks
    4. Rerank results for relevance
    5. Return filtered, ranked results
    """
    start_time = time.time()
    
    # Build query context from user
    context = QueryContext(
        user_id=user.id,
        user_role=user.role.value,
        user_clearance=user.effective_clearance,
        user_department=user.department,
        user_allowed_domains=user.allowed_domains,
    )
    
    # Handle classification filter
    classification = None
    if request.classification_max:
        try:
            classification = SecurityClassification(request.classification_max)
        except ValueError:
            pass
    
    # Create search query
    query = SearchQuery(
        text=request.query,
        context=context,
        max_results=request.max_results,
        min_score=request.min_score,
        domain_filter=request.domain,
        classification_filter=classification,
    )
    
    # Perform retrieval
    result = retriever.retrieve(query, user)
    
    # Format results
    chunk_results = []
    for chunk, score in zip(result.chunks, result.scores):
        if score >= request.min_score:
            chunk_results.append(
                ChunkResult(
                    chunk_id=chunk.id,
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    context=chunk.context,
                    score=score,
                    title=chunk.metadata.title,
                    source=chunk.metadata.source,
                    domain=chunk.metadata.domain,
                    classification=chunk.metadata.classification.value,
                )
            )
    
    total_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        query_id=query.id,
        query=request.query,
        results=chunk_results,
        total_candidates=result.total_candidates,
        filtered_by_security=result.filtered_by_security,
        was_reranked=result.was_reranked,
        retrieval_time_ms=result.retrieval_time_ms,
        total_time_ms=total_time,
    )


@router.post("/quick", response_model=QuickSearchResponse)
async def quick_search(
    request: QuickSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    retriever: SecureRetriever = Depends(get_retriever),
) -> QuickSearchResponse:
    """
    Quick search without full analysis pipeline.
    
    Faster but less sophisticated than the full search endpoint.
    """
    start_time = time.time()
    
    # Perform simple retrieval
    results = retriever.simple_retrieve(
        query_text=request.query,
        user=user,
        max_results=request.limit,
    )
    
    # Format results
    chunk_results = []
    for chunk, score in results:
        chunk_results.append(
            ChunkResult(
                chunk_id=chunk.id,
                document_id=str(chunk.document_id),
                content=chunk.content,
                context=chunk.context,
                score=score,
                title=chunk.metadata.title,
                source=chunk.metadata.source,
                domain=chunk.metadata.domain,
                classification=chunk.metadata.classification.value,
            )
        )
    
    total_time = (time.time() - start_time) * 1000
    
    return QuickSearchResponse(
        results=chunk_results,
        count=len(chunk_results),
        time_ms=total_time,
    )


@router.post("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QuickSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    retriever: SecureRetriever = Depends(get_retriever),
) -> QueryAnalysisResponse:
    """
    Analyze a query without performing retrieval.
    
    Returns intent, entities, and optimization suggestions.
    """
    if not retriever.query_analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query analysis not available",
        )
    
    # Build context
    context = QueryContext(
        user_id=user.id,
        user_role=user.role.value,
        user_clearance=user.effective_clearance,
        user_department=user.department,
        user_allowed_domains=user.allowed_domains,
    )
    
    # Create query
    query = SearchQuery(text=request.query, context=context)
    
    # Analyze
    analyzed = retriever.query_analyzer.analyze(query)
    
    return QueryAnalysisResponse(
        original_query=analyzed.original_query,
        intent=analyzed.intent.value,
        intent_confidence=analyzed.intent_confidence,
        reformulated_query=analyzed.reformulated_query,
        expanded_queries=analyzed.expanded_queries,
        entities=analyzed.entities,
        keywords=analyzed.keywords,
        detected_domains=analyzed.detected_domains,
        is_ambiguous=analyzed.is_ambiguous,
        disambiguation_options=analyzed.disambiguation_options,
        recommended_top_k=analyzed.recommended_top_k,
        use_hybrid_search=analyzed.use_hybrid_search,
        analysis_time_ms=analyzed.analysis_time_ms,
    )


@router.get("/history", response_model=list[dict])
async def get_query_history(
    user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """
    Get query history for the current user.
    
    Note: In production, this would fetch from a persistent store.
    """
    # Placeholder - would integrate with session/database
    return []


@router.post("/feedback")
async def submit_feedback(
    query_id: UUID,
    chunk_id: str,
    relevant: bool,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """
    Submit relevance feedback for a search result.
    
    This feedback can be used to improve retrieval quality over time.
    """
    # Placeholder for feedback collection
    return {
        "status": "received",
        "query_id": str(query_id),
        "chunk_id": chunk_id,
        "relevant": relevant,
    }
