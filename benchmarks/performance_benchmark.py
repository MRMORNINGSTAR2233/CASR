"""
Performance and Scalability Benchmarks for CASR.

Evaluates:
1. Query latency under various loads
2. Throughput (queries per second)
3. Scalability with document count
4. Concurrent user handling
5. Resource usage (memory, CPU)
"""

import time
import asyncio
import psutil
import os
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.retrieval.secure_retriever import SecureRetriever
from src.storage.chroma_store import ChromaVectorStore
from src.storage.vector_store import VectorStoreConfig
from src.indexing.embedder import Embedder, EmbeddingProvider
from src.indexing.chunker import Chunker
from src.models.queries import Query
from src.models.users import User, UserRole
from benchmarks.metrics import MetricsCalculator, PerformanceMetrics
from benchmarks.dataset_generator import DatasetGenerator


class PerformanceBenchmark:
    """Comprehensive performance and scalability testing."""
    
    def __init__(
        self,
        dataset_dir: Path,
        results_dir: Path
    ):
        """
        Initialize performance benchmark.
        
        Args:
            dataset_dir: Directory containing test dataset
            results_dir: Directory to save results
        """
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.process = psutil.Process(os.getpid())
        
        # Load dataset
        print("Loading dataset...")
        self.documents, self.queries = DatasetGenerator.load_dataset(dataset_dir)
        print(f"Loaded {len(self.documents)} documents and {len(self.queries)} queries")
        
        # Test user
        self.test_user = User(
            id="perf_test_user",
            username="perf_test",
            email="perf@test.com",
            roles=[UserRole.EMPLOYEE],
            attributes={"department": "test", "region": "US"}
        )
    
    async def _setup_vector_store(self, doc_count: int = None) -> ChromaVectorStore:
        """Set up vector store with specified number of documents."""
        config = VectorStoreConfig(
            store_type="chroma",
            collection_name=f"perf_benchmark_{doc_count or 'all'}",
            persist_directory=str(self.results_dir / "temp_stores")
        )
        vector_store = ChromaVectorStore(config)
        
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        docs_to_index = self.documents[:doc_count] if doc_count else self.documents
        
        print(f"Indexing {len(docs_to_index)} documents...")
        for i, doc in enumerate(docs_to_index):
            if i % 50 == 0:
                print(f"  Indexed {i}/{len(docs_to_index)}...")
            
            chunks = chunker.chunk_document(doc)
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.embed_texts(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding
            
            await vector_store.add_documents(chunks)
        
        return vector_store
    
    def _measure_resources(self) -> Dict[str, float]:
        """Measure current resource usage."""
        mem_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        return {
            "memory_mb": mem_info.rss / 1024 / 1024,
            "cpu_percent": cpu_percent
        }
    
    async def benchmark_latency(self) -> Dict[str, Any]:
        """
        Benchmark query latency distribution.
        
        Returns:
            Latency benchmark results
        """
        print("\n=== Latency Benchmark ===")
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        from src.security.rbac import RBACManager
        from src.security.abac import ABACManager
        from src.security.policy_engine import PolicyEngine
        from src.security.audit import AuditLogger
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=RBACManager(),
            abac_manager=ABACManager(),
            policy_engine=PolicyEngine(),
            audit_logger=AuditLogger()
        )
        
        latencies = []
        embedding_times = []
        retrieval_times = []
        
        num_queries = min(100, len(self.queries))
        print(f"Running {num_queries} queries...")
        
        for i, query in enumerate(self.queries[:num_queries]):
            if i % 20 == 0:
                print(f"  Completed {i}/{num_queries}...")
            
            # Measure embedding time
            embed_start = time.time()
            query_embedding = await embedder.embed_query(query.query)
            embed_time = (time.time() - embed_start) * 1000
            embedding_times.append(embed_time)
            
            # Measure retrieval time
            retrieval_start = time.time()
            try:
                query_obj = Query(text=query.query, user_id=self.test_user.id)
                await secure_retriever.retrieve(
                    query=query_obj,
                    user=self.test_user,
                    query_vector=query_embedding,
                    limit=10
                )
                retrieval_time = (time.time() - retrieval_start) * 1000
                retrieval_times.append(retrieval_time)
                
                total_latency = embed_time + retrieval_time
                latencies.append(total_latency)
            except Exception as e:
                print(f"    Query failed: {e}")
        
        # Calculate metrics
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            latencies=latencies,
            total_time=sum(latencies) / 1000
        )
        
        results = {
            "total_latency": {
                "avg_ms": performance_metrics.avg_latency_ms,
                "p50_ms": performance_metrics.p50_latency_ms,
                "p95_ms": performance_metrics.p95_latency_ms,
                "p99_ms": performance_metrics.p99_latency_ms,
                "max_ms": performance_metrics.max_latency_ms,
                "min_ms": performance_metrics.min_latency_ms
            },
            "embedding_time": {
                "avg_ms": np.mean(embedding_times),
                "p95_ms": np.percentile(embedding_times, 95)
            },
            "retrieval_time": {
                "avg_ms": np.mean(retrieval_times),
                "p95_ms": np.percentile(retrieval_times, 95)
            },
            "num_queries": num_queries
        }
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "latency_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nLatency Results:")
        print(f"  Average: {results['total_latency']['avg_ms']:.2f}ms")
        print(f"  P95: {results['total_latency']['p95_ms']:.2f}ms")
        print(f"  P99: {results['total_latency']['p99_ms']:.2f}ms")
        print(f"Results saved to {results_file}")
        
        return results
    
    async def benchmark_throughput(self, concurrent_users: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark throughput with varying concurrent users.
        
        Args:
            concurrent_users: List of concurrent user counts to test
            
        Returns:
            Throughput benchmark results
        """
        print("\n=== Throughput Benchmark ===")
        
        if concurrent_users is None:
            concurrent_users = [1, 5, 10, 20, 50]
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        from src.security.rbac import RBACManager
        from src.security.abac import ABACManager
        from src.security.policy_engine import PolicyEngine
        from src.security.audit import AuditLogger
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=RBACManager(),
            abac_manager=ABACManager(),
            policy_engine=PolicyEngine(),
            audit_logger=AuditLogger()
        )
        
        results = {
            "by_concurrency": {},
            "summary": []
        }
        
        for num_users in concurrent_users:
            print(f"\nTesting with {num_users} concurrent users...")
            
            # Prepare queries for concurrent execution
            queries_per_user = 20
            total_queries = num_users * queries_per_user
            test_queries = (self.queries * ((total_queries // len(self.queries)) + 1))[:total_queries]
            
            # Execute queries concurrently
            async def execute_query(query_text: str):
                try:
                    query_embedding = await embedder.embed_query(query_text)
                    query_obj = Query(text=query_text, user_id=self.test_user.id)
                    await secure_retriever.retrieve(
                        query=query_obj,
                        user=self.test_user,
                        query_vector=query_embedding,
                        limit=10
                    )
                    return True
                except Exception:
                    return False
            
            start_time = time.time()
            
            # Run queries concurrently
            tasks = [execute_query(q.query) for q in test_queries]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            successful = sum(1 for r in results_list if r is True)
            failed = len(results_list) - successful
            
            qps = successful / duration if duration > 0 else 0.0
            
            # Measure resources
            resources = self._measure_resources()
            
            results["by_concurrency"][num_users] = {
                "total_queries": total_queries,
                "successful_queries": successful,
                "failed_queries": failed,
                "duration_seconds": duration,
                "queries_per_second": qps,
                "avg_latency_ms": (duration * 1000) / successful if successful > 0 else 0,
                "memory_mb": resources["memory_mb"],
                "cpu_percent": resources["cpu_percent"]
            }
            
            results["summary"].append({
                "concurrent_users": num_users,
                "qps": qps,
                "avg_latency_ms": (duration * 1000) / successful if successful > 0 else 0
            })
            
            print(f"  QPS: {qps:.2f}")
            print(f"  Avg Latency: {(duration * 1000) / successful if successful > 0 else 0:.2f}ms")
            print(f"  Success Rate: {successful / total_queries * 100:.1f}%")
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "throughput_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    async def benchmark_scalability(self, doc_counts: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark scalability with varying document counts.
        
        Args:
            doc_counts: List of document counts to test
            
        Returns:
            Scalability benchmark results
        """
        print("\n=== Scalability Benchmark ===")
        
        if doc_counts is None:
            doc_counts = [100, 500, 1000, 2000]
        
        # Filter to available document counts
        doc_counts = [count for count in doc_counts if count <= len(self.documents)]
        
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        from src.security.rbac import RBACManager
        from src.security.abac import ABACManager
        from src.security.policy_engine import PolicyEngine
        from src.security.audit import AuditLogger
        
        results = {
            "by_doc_count": {},
            "scaling_factor": []
        }
        
        baseline_qps = None
        
        for doc_count in doc_counts:
            print(f"\nTesting with {doc_count} documents...")
            
            vector_store = await self._setup_vector_store(doc_count=doc_count)
            
            secure_retriever = SecureRetriever(
                vector_store=vector_store,
                rbac_manager=RBACManager(),
                abac_manager=ABACManager(),
                policy_engine=PolicyEngine(),
                audit_logger=AuditLogger()
            )
            
            # Run test queries
            num_queries = 50
            latencies = []
            
            start_time = time.time()
            
            for query in self.queries[:num_queries]:
                try:
                    query_embedding = await embedder.embed_query(query.query)
                    query_obj = Query(text=query.query, user_id=self.test_user.id)
                    
                    query_start = time.time()
                    await secure_retriever.retrieve(
                        query=query_obj,
                        user=self.test_user,
                        query_vector=query_embedding,
                        limit=10
                    )
                    latency = (time.time() - query_start) * 1000
                    latencies.append(latency)
                except Exception:
                    pass
            
            duration = time.time() - start_time
            qps = len(latencies) / duration if duration > 0 else 0.0
            
            if baseline_qps is None:
                baseline_qps = qps
            
            # Measure resources
            resources = self._measure_resources()
            
            results["by_doc_count"][doc_count] = {
                "num_queries": len(latencies),
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "queries_per_second": qps,
                "memory_mb": resources["memory_mb"],
                "cpu_percent": resources["cpu_percent"]
            }
            
            scaling_factor = baseline_qps / qps if qps > 0 else 0
            results["scaling_factor"].append({
                "doc_count": doc_count,
                "qps": qps,
                "scaling_factor": scaling_factor
            })
            
            print(f"  Avg Latency: {np.mean(latencies) if latencies else 0:.2f}ms")
            print(f"  QPS: {qps:.2f}")
            print(f"  Memory: {resources['memory_mb']:.2f}MB")
            
            # Cleanup
            await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "scalability_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    async def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE AND SCALABILITY TESTING")
        print("="*80)
        
        results = {}
        
        # Latency tests
        results["latency"] = await self.benchmark_latency()
        
        # Throughput tests
        results["throughput"] = await self.benchmark_throughput()
        
        # Scalability tests
        results["scalability"] = await self.benchmark_scalability()
        
        # Save combined results
        combined_file = self.results_dir / "performance_comprehensive.json"
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nCombined performance results saved to {combined_file}")
        
        return results


async def main():
    """Run performance benchmarks."""
    dataset_dir = Path(__file__).parent / "data" / "test_dataset"
    results_dir = Path(__file__).parent / "results" / "performance"
    
    benchmark = PerformanceBenchmark(
        dataset_dir=dataset_dir,
        results_dir=results_dir
    )
    
    await benchmark.run_all_performance_tests()


if __name__ == "__main__":
    asyncio.run(main())
