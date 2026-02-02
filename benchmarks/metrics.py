"""
Comprehensive metrics calculation for CASR evaluation.

Includes:
- Retrieval metrics (Recall@K, Precision@K, MRR, NDCG)
- Security metrics (Policy violations, unauthorized access)
- Performance metrics (Latency, throughput, resource usage)
- Quality metrics (Contextual relevance, answer quality)
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import time


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality evaluation."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)  # {k: recall}
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)  # Normalized DCG
    map_score: float = 0.0  # Mean Average Precision
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg@k": self.ndcg_at_k,
            "map": self.map_score,
            "hit_rate@k": self.hit_rate_at_k
        }


@dataclass
class SecurityMetrics:
    """Metrics for security enforcement evaluation."""
    total_queries: int = 0
    authorized_queries: int = 0
    unauthorized_blocked: int = 0
    unauthorized_leaked: int = 0  # Critical: should be 0
    policy_violations: int = 0
    false_positives: int = 0  # Authorized but blocked
    false_negatives: int = 0  # Unauthorized but allowed
    rbac_accuracy: float = 0.0
    abac_accuracy: float = 0.0
    security_classification_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "authorized_queries": self.authorized_queries,
            "unauthorized_blocked": self.unauthorized_blocked,
            "unauthorized_leaked": self.unauthorized_leaked,
            "policy_violations": self.policy_violations,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "rbac_accuracy": self.rbac_accuracy,
            "abac_accuracy": self.abac_accuracy,
            "security_classification_accuracy": self.security_classification_accuracy
        }


@dataclass
class PerformanceMetrics:
    """Metrics for performance evaluation."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    throughput_qps: float = 0.0  # Queries per second
    total_queries: int = 0
    failed_queries: int = 0
    timeout_queries: int = 0
    avg_tokens_per_query: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "throughput_qps": self.throughput_qps,
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "timeout_queries": self.timeout_queries,
            "avg_tokens_per_query": self.avg_tokens_per_query,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


@dataclass
class ContextualMetrics:
    """Metrics for contextual enhancement evaluation."""
    with_context_recall: float = 0.0
    without_context_recall: float = 0.0
    context_improvement: float = 0.0  # Percentage improvement
    avg_context_length: float = 0.0
    context_generation_time_ms: float = 0.0
    context_relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "with_context_recall": self.with_context_recall,
            "without_context_recall": self.without_context_recall,
            "context_improvement_percent": self.context_improvement,
            "avg_context_length": self.avg_context_length,
            "context_generation_time_ms": self.context_generation_time_ms,
            "context_relevance_score": self.context_relevance_score
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a single configuration."""
    config_name: str
    timestamp: datetime
    retrieval_metrics: RetrievalMetrics
    security_metrics: SecurityMetrics
    performance_metrics: PerformanceMetrics
    contextual_metrics: Optional[ContextualMetrics] = None
    provider: str = "unknown"
    model: str = "unknown"
    dataset: str = "unknown"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "config_name": self.config_name,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "dataset": self.dataset,
            "notes": self.notes,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "security_metrics": self.security_metrics.to_dict(),
            "performance_metrics": self.performance_metrics.to_dict()
        }
        if self.contextual_metrics:
            result["contextual_metrics"] = self.contextual_metrics.to_dict()
        return result


class MetricsCalculator:
    """Calculate comprehensive metrics for CASR evaluation."""
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics calculator.
        
        Args:
            k_values: K values for Recall@K, Precision@K, etc. Default [1, 3, 5, 10, 20]
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K: fraction of relevant documents in top-K results.
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by relevance)
            relevant_ids: List of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)
    
    def calculate_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K: fraction of relevant documents in top-K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        hits = len(top_k & relevant_set)
        return hits / k
    
    def calculate_mrr(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank across multiple queries.
        
        Args:
            all_retrieved: List of retrieved document lists for each query
            all_relevant: List of relevant document lists for each query
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved_ids, relevant_ids in zip(all_retrieved, all_relevant):
            relevant_set = set(relevant_ids)
            
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_dcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevance_scores: Dict mapping document IDs to relevance scores
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            # DCG formula: rel_i / log2(i+2)
            dcg += relevance / np.log2(i + 2)
        
        return dcg
    
    def calculate_ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate Normalized DCG at K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevance_scores: Dict mapping document IDs to relevance scores (0.0 to 1.0)
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        dcg = self.calculate_dcg_at_k(retrieved_ids, relevance_scores, k)
        
        # Ideal DCG: sort by relevance scores
        ideal_order = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ideal_ids = [doc_id for doc_id, _ in ideal_order]
        idcg = self.calculate_dcg_at_k(ideal_ids, relevance_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K: binary indicator if any relevant doc in top-K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        return 1.0 if (top_k & relevant_set) else 0.0
    
    def calculate_map(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[List[str]]
    ) -> float:
        """
        Calculate Mean Average Precision across multiple queries.
        
        Args:
            all_retrieved: List of retrieved document lists for each query
            all_relevant: List of relevant document lists for each query
            
        Returns:
            MAP score
        """
        average_precisions = []
        
        for retrieved_ids, relevant_ids in zip(all_retrieved, all_relevant):
            if not relevant_ids:
                continue
            
            relevant_set = set(relevant_ids)
            precisions = []
            num_hits = 0
            
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_set:
                    num_hits += 1
                    precision_at_i = num_hits / i
                    precisions.append(precision_at_i)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def calculate_retrieval_metrics(
        self,
        all_retrieved: List[List[str]],
        all_relevant: List[List[str]],
        all_relevance_scores: Optional[List[Dict[str, float]]] = None
    ) -> RetrievalMetrics:
        """
        Calculate comprehensive retrieval metrics.
        
        Args:
            all_retrieved: List of retrieved document lists for each query
            all_relevant: List of relevant document lists for each query
            all_relevance_scores: Optional list of relevance score dicts for NDCG
            
        Returns:
            RetrievalMetrics object with all calculated metrics
        """
        metrics = RetrievalMetrics()
        
        # Calculate metrics for each K value
        for k in self.k_values:
            recall_scores = []
            precision_scores = []
            hit_scores = []
            ndcg_scores = []
            
            for i, (retrieved, relevant) in enumerate(zip(all_retrieved, all_relevant)):
                recall_scores.append(self.calculate_recall_at_k(retrieved, relevant, k))
                precision_scores.append(self.calculate_precision_at_k(retrieved, relevant, k))
                hit_scores.append(self.calculate_hit_rate_at_k(retrieved, relevant, k))
                
                if all_relevance_scores and i < len(all_relevance_scores):
                    ndcg_scores.append(
                        self.calculate_ndcg_at_k(retrieved, all_relevance_scores[i], k)
                    )
            
            metrics.recall_at_k[k] = np.mean(recall_scores)
            metrics.precision_at_k[k] = np.mean(precision_scores)
            metrics.hit_rate_at_k[k] = np.mean(hit_scores)
            
            if ndcg_scores:
                metrics.ndcg_at_k[k] = np.mean(ndcg_scores)
        
        # Calculate MRR and MAP
        metrics.mrr = self.calculate_mrr(all_retrieved, all_relevant)
        metrics.map_score = self.calculate_map(all_retrieved, all_relevant)
        
        return metrics
    
    def calculate_security_metrics(
        self,
        expected_access: List[bool],
        actual_access: List[bool],
        expected_documents: List[List[str]],
        retrieved_documents: List[List[str]]
    ) -> SecurityMetrics:
        """
        Calculate security enforcement metrics.
        
        Args:
            expected_access: Expected access decisions (True = should be allowed)
            actual_access: Actual access decisions
            expected_documents: Expected document IDs that should be returned
            retrieved_documents: Actual document IDs retrieved
            
        Returns:
            SecurityMetrics object
        """
        metrics = SecurityMetrics()
        metrics.total_queries = len(expected_access)
        
        for exp_access, act_access, exp_docs, ret_docs in zip(
            expected_access, actual_access, expected_documents, retrieved_documents
        ):
            if exp_access:
                metrics.authorized_queries += 1
                if not act_access:
                    metrics.false_positives += 1  # Should allow but blocked
                else:
                    # Check if only authorized documents returned
                    exp_set = set(exp_docs)
                    ret_set = set(ret_docs)
                    unauthorized = ret_set - exp_set
                    if unauthorized:
                        metrics.unauthorized_leaked += len(unauthorized)
            else:
                # Unauthorized query
                if act_access or ret_docs:
                    metrics.false_negatives += 1  # Should block but allowed
                    metrics.unauthorized_leaked += len(ret_docs)
                else:
                    metrics.unauthorized_blocked += 1
        
        # Calculate accuracy metrics
        correct_decisions = sum(
            1 for exp, act in zip(expected_access, actual_access) if exp == act
        )
        metrics.rbac_accuracy = correct_decisions / metrics.total_queries if metrics.total_queries > 0 else 0.0
        metrics.abac_accuracy = metrics.rbac_accuracy  # Same for now
        
        # Security classification accuracy
        metrics.security_classification_accuracy = 1.0 - (
            metrics.unauthorized_leaked / metrics.total_queries if metrics.total_queries > 0 else 0.0
        )
        
        return metrics
    
    def calculate_performance_metrics(
        self,
        latencies: List[float],
        total_time: float,
        failed_queries: int = 0,
        timeout_queries: int = 0,
        token_counts: Optional[List[int]] = None,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Args:
            latencies: List of query latencies in milliseconds
            total_time: Total time for all queries in seconds
            failed_queries: Number of failed queries
            timeout_queries: Number of timed out queries
            token_counts: Optional list of token counts per query
            memory_usage: Optional memory usage in MB
            cpu_usage: Optional CPU usage percentage
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if latencies:
            metrics.avg_latency_ms = np.mean(latencies)
            metrics.p50_latency_ms = np.percentile(latencies, 50)
            metrics.p95_latency_ms = np.percentile(latencies, 95)
            metrics.p99_latency_ms = np.percentile(latencies, 99)
            metrics.max_latency_ms = np.max(latencies)
            metrics.min_latency_ms = np.min(latencies)
        
        metrics.total_queries = len(latencies) + failed_queries + timeout_queries
        metrics.failed_queries = failed_queries
        metrics.timeout_queries = timeout_queries
        
        successful_queries = len(latencies)
        if total_time > 0:
            metrics.throughput_qps = successful_queries / total_time
        
        if token_counts:
            metrics.avg_tokens_per_query = np.mean(token_counts)
        
        if memory_usage is not None:
            metrics.memory_usage_mb = memory_usage
        
        if cpu_usage is not None:
            metrics.cpu_usage_percent = cpu_usage
        
        return metrics
    
    def calculate_contextual_metrics(
        self,
        with_context_results: List[List[str]],
        without_context_results: List[List[str]],
        relevant_docs: List[List[str]],
        context_lengths: List[int],
        context_gen_times: List[float],
        context_relevance_scores: Optional[List[float]] = None
    ) -> ContextualMetrics:
        """
        Calculate contextual enhancement metrics.
        
        Args:
            with_context_results: Retrieval results with contextualization
            without_context_results: Retrieval results without contextualization
            relevant_docs: Ground truth relevant documents
            context_lengths: Character lengths of generated contexts
            context_gen_times: Time to generate contexts in milliseconds
            context_relevance_scores: Optional relevance scores for contexts
            
        Returns:
            ContextualMetrics object
        """
        metrics = ContextualMetrics()
        
        # Calculate recall for both approaches
        with_recalls = []
        without_recalls = []
        
        for with_ret, without_ret, relevant in zip(
            with_context_results, without_context_results, relevant_docs
        ):
            with_recalls.append(self.calculate_recall_at_k(with_ret, relevant, 10))
            without_recalls.append(self.calculate_recall_at_k(without_ret, relevant, 10))
        
        metrics.with_context_recall = np.mean(with_recalls)
        metrics.without_context_recall = np.mean(without_recalls)
        
        # Calculate improvement
        if metrics.without_context_recall > 0:
            metrics.context_improvement = (
                (metrics.with_context_recall - metrics.without_context_recall) 
                / metrics.without_context_recall * 100
            )
        
        # Context statistics
        if context_lengths:
            metrics.avg_context_length = np.mean(context_lengths)
        
        if context_gen_times:
            metrics.context_generation_time_ms = np.mean(context_gen_times)
        
        if context_relevance_scores:
            metrics.context_relevance_score = np.mean(context_relevance_scores)
        
        return metrics
