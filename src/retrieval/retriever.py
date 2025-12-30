"""
Retriever

Main retrieval orchestration with security filtering.
"""

import time
from typing import Any, Optional
from uuid import UUID

from config import get_settings
from src.indexing.embedder import Embedder
from src.models.documents import DocumentChunk
from src.models.policies import PolicyAction
from src.models.queries import (
    AnalyzedQuery,
    QueryContext,
    RetrievalResult,
    SearchQuery,
)
from src.models.users import User
from src.security.audit import AuditLogger
from src.security.policies import PolicyEngine
from src.storage.base import VectorStore

from .analyzer import QueryAnalyzer
from .reranker import Reranker, RerankerProvider, SimpleReranker


class Retriever:
    """
    Base retriever for document search.
    
    Orchestrates the retrieval pipeline:
    1. Query embedding
    2. Vector search
    3. Optional reranking
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        enable_reranking: bool = True,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store for document storage
            embedder: Embedder for query embedding
            reranker: Reranker for result reranking
            enable_reranking: Whether to use reranking
            top_k: Number of initial results to retrieve
            rerank_top_k: Number of results after reranking
        """
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()
        self.enable_reranking = enable_reranking
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        settings = get_settings()
        
        if enable_reranking:
            self.reranker = reranker or (
                Reranker(top_k=rerank_top_k)
                if settings.cohere_api_key
                else SimpleReranker(top_k=rerank_top_k)
            )
        else:
            self.reranker = None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        use_reranking: Optional[bool] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve relevant document chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters
            use_reranking: Override reranking setting
            
        Returns:
            List of (chunk, score) tuples
        """
        effective_top_k = top_k or self.top_k
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve from vector store
        # Get more candidates if we're going to rerank
        search_top_k = effective_top_k * 3 if should_rerank and self.reranker else effective_top_k
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_top_k,
            filters=filters,
        )
        
        # Rerank if enabled
        if should_rerank and self.reranker and results:
            results = self.reranker.rerank_with_scores(
                query=query,
                chunks_with_scores=results,
                top_k=effective_top_k,
            )
        
        return results[:effective_top_k]
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval combining vector and keyword search.
        
        Args:
            query: Search query text
            top_k: Number of results
            filters: Metadata filters
            alpha: Balance between vector and keyword (0=keyword, 1=vector)
            
        Returns:
            List of (chunk, score) tuples
        """
        effective_top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Use hybrid search if supported
        if hasattr(self.vector_store, 'hybrid_search'):
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=effective_top_k,
                filters=filters,
                alpha=alpha,
            )
        else:
            # Fallback to regular search
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                filters=filters,
            )
        
        # Optionally rerank
        if self.enable_reranking and self.reranker and results:
            results = self.reranker.rerank_with_scores(
                query=query,
                chunks_with_scores=results,
                top_k=effective_top_k,
            )
        
        return results


class SecureRetriever:
    """
    Security-aware retriever with policy enforcement.
    
    Extends base retrieval with:
    - User context awareness
    - Access control filtering
    - Query analysis
    - Audit logging
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        policy_engine: Optional[PolicyEngine] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        audit_logger: Optional[AuditLogger] = None,
        enable_analysis: bool = True,
        enable_reranking: bool = True,
    ):
        """
        Initialize the secure retriever.
        
        Args:
            vector_store: Vector store for document storage
            policy_engine: Policy engine for access control
            query_analyzer: Query analyzer for intent detection
            embedder: Embedder for query embedding
            reranker: Reranker for result reranking
            audit_logger: Audit logger for security logging
            enable_analysis: Whether to analyze queries
            enable_reranking: Whether to rerank results
        """
        settings = get_settings()
        
        self.vector_store = vector_store
        self.policy_engine = policy_engine or PolicyEngine()
        self.embedder = embedder or Embedder()
        self.audit_logger = audit_logger or AuditLogger()
        
        self.enable_analysis = enable_analysis
        self.enable_reranking = enable_reranking
        
        if enable_analysis:
            self.query_analyzer = query_analyzer or QueryAnalyzer()
        else:
            self.query_analyzer = None
        
        if enable_reranking:
            self.reranker = reranker or (
                Reranker()
                if settings.cohere_api_key
                else SimpleReranker()
            )
        else:
            self.reranker = None
    
    def _apply_security_filters(
        self,
        user: User,
        base_filters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Generate security filters based on user permissions.
        
        Args:
            user: The requesting user
            base_filters: Additional filters to apply
            
        Returns:
            Combined filter dictionary
        """
        # Get filters from policy engine
        security_filters = self.policy_engine.get_query_filters(
            user=user,
            action=PolicyAction.READ,
        )
        
        # Merge with base filters
        if base_filters:
            for key, value in base_filters.items():
                if key in security_filters:
                    # Merge filter conditions
                    if isinstance(security_filters[key], dict) and isinstance(value, dict):
                        security_filters[key].update(value)
                else:
                    security_filters[key] = value
        
        return security_filters
    
    def _filter_results_by_policy(
        self,
        user: User,
        chunks_with_scores: list[tuple[DocumentChunk, float]]
    ) -> tuple[list[tuple[DocumentChunk, float]], int]:
        """
        Filter results based on access policies.
        
        Returns:
            Tuple of (filtered_results, count_filtered)
        """
        filtered = []
        filtered_count = 0
        
        for chunk, score in chunks_with_scores:
            allowed, _ = self.policy_engine.evaluate_access(
                user=user,
                resource=chunk.metadata,
                action=PolicyAction.READ,
            )
            
            if allowed:
                filtered.append((chunk, score))
            else:
                filtered_count += 1
        
        return filtered, filtered_count
    
    def retrieve(
        self,
        query: SearchQuery,
        user: User,
    ) -> RetrievalResult:
        """
        Perform secure retrieval with full pipeline.
        
        Args:
            query: The search query
            user: The requesting user
            
        Returns:
            RetrievalResult with chunks and metadata
        """
        start_time = time.time()
        
        # Step 1: Analyze query (if enabled)
        analyzed: Optional[AnalyzedQuery] = None
        if self.enable_analysis and self.query_analyzer:
            analyzed = self.query_analyzer.analyze(query)
        
        # Step 2: Apply security filters
        base_filters = {}
        if analyzed:
            base_filters.update(analyzed.metadata_filters)
            if analyzed.primary_domain:
                base_filters["domain"] = analyzed.primary_domain
        
        security_filters = self._apply_security_filters(user, base_filters)
        
        # Step 3: Generate query embedding
        query_text = analyzed.reformulated_query if analyzed else query.text
        query_embedding = self.embedder.embed_query(query_text)
        
        # Step 4: Retrieve from vector store
        top_k = analyzed.recommended_top_k if analyzed else query.max_results * 3
        
        if analyzed and analyzed.use_hybrid_search and hasattr(self.vector_store, 'hybrid_search'):
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                filters=security_filters,
            )
        else:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=security_filters,
            )
        
        retrieval_time = (time.time() - start_time) * 1000
        total_candidates = len(results)
        
        # Step 5: Post-filter by policy (belt and suspenders)
        results, filtered_by_security = self._filter_results_by_policy(user, results)
        
        # Step 6: Rerank (if enabled)
        rerank_time = 0.0
        if self.enable_reranking and self.reranker and results:
            rerank_start = time.time()
            results = self.reranker.rerank_with_scores(
                query=query.text,
                chunks_with_scores=results,
                top_k=query.max_results,
            )
            rerank_time = (time.time() - rerank_start) * 1000
        else:
            results = results[:query.max_results]
        
        total_time = (time.time() - start_time) * 1000
        
        # Extract chunks and scores
        chunks = [chunk for chunk, _ in results]
        scores = [score for _, score in results]
        
        # Get unique documents and domains
        unique_docs = set(str(chunk.document_id) for chunk in chunks)
        domains = list(set(chunk.metadata.domain for chunk in chunks))
        
        # Log the retrieval
        self.audit_logger.log_retrieval(
            user_id=str(user.id),
            query_id=str(query.id),
            documents_retrieved=len(chunks),
            documents_filtered=filtered_by_security,
            retrieval_time_ms=total_time,
        )
        
        return RetrievalResult(
            query_id=query.id,
            chunks=chunks,
            scores=scores,
            total_candidates=total_candidates,
            filtered_by_security=filtered_by_security,
            filtered_by_domain=0,  # Already handled in security filters
            was_reranked=self.enable_reranking and self.reranker is not None,
            rerank_model=self.reranker.model if self.reranker else None,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            unique_documents=len(unique_docs),
            domains_represented=domains,
        )
    
    def simple_retrieve(
        self,
        query_text: str,
        user: User,
        max_results: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Simple retrieval interface without full query object.
        
        Args:
            query_text: The query text
            user: The requesting user
            max_results: Maximum results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        # Build query context from user
        context = QueryContext(
            user_id=user.id,
            user_role=user.role.value,
            user_clearance=user.effective_clearance,
            user_department=user.department,
            user_allowed_domains=user.allowed_domains,
        )
        
        # Create search query
        query = SearchQuery(
            text=query_text,
            context=context,
            max_results=max_results,
        )
        
        # Perform retrieval
        result = self.retrieve(query, user)
        
        return list(zip(result.chunks, result.scores))
    
    async def aretrieve(
        self,
        query: SearchQuery,
        user: User,
    ) -> RetrievalResult:
        """Async version of retrieve."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, user)
