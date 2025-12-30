"""
Reranker

Reranks retrieved documents for improved relevance.
"""

from enum import Enum
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models.documents import DocumentChunk


class RerankerProvider(str, Enum):
    """Supported reranking providers."""
    COHERE = "cohere"
    CROSS_ENCODER = "cross-encoder"


class Reranker:
    """
    Document reranker for improving retrieval quality.
    
    Uses a reranking model to re-score retrieved documents
    based on their relevance to the query.
    """
    
    def __init__(
        self,
        provider: RerankerProvider = RerankerProvider.COHERE,
        model: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize the reranker.
        
        Args:
            provider: Reranking provider
            model: Model name
            top_k: Number of documents to return after reranking
        """
        self.provider = provider
        self.top_k = top_k
        
        settings = get_settings()
        
        match provider:
            case RerankerProvider.COHERE:
                import cohere
                self.model = model or "rerank-english-v3.0"
                self._client = cohere.Client(settings.cohere_api_key)
            
            case RerankerProvider.CROSS_ENCODER:
                from sentence_transformers import CrossEncoder
                self.model = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self._cross_encoder = CrossEncoder(self.model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _rerank_cohere(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank using Cohere."""
        if not chunks:
            return []
        
        # Prepare documents for Cohere
        documents = [chunk.content_for_embedding for chunk in chunks]
        
        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(chunks)),
            return_documents=False,
        )
        
        reranked: list[tuple[DocumentChunk, float]] = []
        for result in response.results:
            chunk = chunks[result.index]
            reranked.append((chunk, result.relevance_score))
        
        return reranked
    
    def _rerank_cross_encoder(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank using a cross-encoder model."""
        if not chunks:
            return []
        
        # Create query-document pairs
        pairs = [(query, chunk.content_for_embedding) for chunk in chunks]
        
        # Score all pairs
        scores = self._cross_encoder.predict(pairs)
        
        # Create scored list and sort
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_k]]
    
    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: Optional[int] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank document chunks by relevance to query.
        
        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        effective_top_k = top_k or self.top_k
        
        if not chunks:
            return []
        
        match self.provider:
            case RerankerProvider.COHERE:
                return self._rerank_cohere(query, chunks, effective_top_k)
            case RerankerProvider.CROSS_ENCODER:
                return self._rerank_cross_encoder(query, chunks, effective_top_k)
            case _:
                # Fallback: return as-is with normalized scores
                return [(chunk, 1.0 - i/len(chunks)) for i, chunk in enumerate(chunks[:effective_top_k])]
    
    def rerank_with_scores(
        self,
        query: str,
        chunks_with_scores: list[tuple[DocumentChunk, float]],
        top_k: Optional[int] = None,
        combine_scores: bool = True,
        alpha: float = 0.7
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank chunks that already have retrieval scores.
        
        Args:
            query: The search query
            chunks_with_scores: List of (chunk, retrieval_score) tuples
            top_k: Number of top results
            combine_scores: Whether to combine retrieval and rerank scores
            alpha: Weight for rerank score (1-alpha for retrieval score)
            
        Returns:
            Reranked list of (chunk, final_score) tuples
        """
        effective_top_k = top_k or self.top_k
        
        if not chunks_with_scores:
            return []
        
        chunks = [chunk for chunk, _ in chunks_with_scores]
        retrieval_scores = {id(chunk): score for chunk, score in chunks_with_scores}
        
        # Get rerank scores
        reranked = self.rerank(query, chunks, len(chunks))  # Get all, we'll filter later
        
        if combine_scores:
            # Combine retrieval and rerank scores
            combined: list[tuple[DocumentChunk, float]] = []
            for chunk, rerank_score in reranked:
                retrieval_score = retrieval_scores.get(id(chunk), 0.0)
                final_score = alpha * rerank_score + (1 - alpha) * retrieval_score
                combined.append((chunk, final_score))
            
            # Sort by combined score
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined[:effective_top_k]
        else:
            return reranked[:effective_top_k]


class SimpleReranker:
    """
    Simple rule-based reranker for cases where ML reranking is not available.
    
    Uses keyword matching and metadata to rerank results.
    """
    
    def __init__(self, top_k: int = 5):
        """Initialize simple reranker."""
        self.top_k = top_k
    
    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: Optional[int] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank using simple keyword matching.
        
        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of results
            
        Returns:
            Reranked list of (chunk, score) tuples
        """
        effective_top_k = top_k or self.top_k
        
        if not chunks:
            return []
        
        query_words = set(query.lower().split())
        
        scored_chunks: list[tuple[DocumentChunk, float]] = []
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            # Calculate keyword overlap score
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            keyword_score = overlap / len(query_words) if query_words else 0
            
            # Bonus for exact phrase match
            if query.lower() in content_lower:
                keyword_score += 0.3
            
            # Bonus for title match
            if query.lower() in chunk.metadata.title.lower():
                keyword_score += 0.2
            
            # Normalize score to 0-1 range
            score = min(keyword_score, 1.0)
            scored_chunks.append((chunk, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:effective_top_k]
