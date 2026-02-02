"""
Retrieval Quality Benchmarks for CASR.

Evaluates:
1. Standard RAG vs CASR comparison
2. Contextual enhancement impact
3. Multi-provider comparison (OpenAI, Anthropic, Gemini, Groq)
4. Cross-domain retrieval performance
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import asyncio

from src.models.documents import Document
from src.models.users import User, UserRole
from src.models.queries import Query, QueryIntent
from src.indexing.chunker import Chunker
from src.indexing.contextualizer import Contextualizer
from src.indexing.embedder import Embedder, EmbeddingProvider
from src.indexing.index_manager import IndexManager
from src.storage.vector_store import VectorStoreConfig
from src.storage.chroma_store import ChromaVectorStore
from src.retrieval.analyzer import QueryAnalyzer
from src.retrieval.retriever import Retriever
from src.retrieval.secure_retriever import SecureRetriever
from benchmarks.metrics import MetricsCalculator, BenchmarkResult, RetrievalMetrics, PerformanceMetrics, ContextualMetrics
from benchmarks.dataset_generator import DatasetGenerator, GroundTruthQuery


class RetrievalBenchmark:
    """Benchmark retrieval quality with and without contextualization."""
    
    def __init__(
        self,
        dataset_dir: Path,
        results_dir: Path,
        providers: List[str] = None
    ):
        """
        Initialize retrieval benchmark.
        
        Args:
            dataset_dir: Directory containing test dataset
            results_dir: Directory to save results
            providers: List of LLM providers to test (default: all)
        """
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.providers = providers or ["openai", "anthropic", "gemini", "groq"]
        self.metrics_calculator = MetricsCalculator()
        
        # Load dataset
        print("Loading dataset...")
        self.documents, self.queries = DatasetGenerator.load_dataset(dataset_dir)
        print(f"Loaded {len(self.documents)} documents and {len(self.queries)} queries")
    
    async def benchmark_standard_rag(
        self,
        provider: str,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    ) -> BenchmarkResult:
        """
        Benchmark standard RAG (no contextualization).
        
        Args:
            provider: LLM provider
            embedding_provider: Embedding provider
            
        Returns:
            BenchmarkResult with metrics
        """
        print(f"\n=== Benchmarking Standard RAG ({provider}) ===")
        
        # Initialize components WITHOUT contextualizer
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        embedder = Embedder(provider=embedding_provider)
        
        # Create temporary vector store
        store_config = VectorStoreConfig(
            store_type="chroma",
            collection_name=f"benchmark_standard_{provider}",
            persist_directory=str(self.results_dir / "temp_stores")
        )
        vector_store = ChromaVectorStore(store_config)
        
        # Index documents without context
        print("Indexing documents (no contextualization)...")
        start_index = time.time()
        
        for doc in self.documents:
            chunks = chunker.chunk_document(doc)
            
            # Embed chunks directly (no context added)
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.embed_texts(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding
            
            await vector_store.add_documents(chunks)
        
        index_time = time.time() - start_index
        print(f"Indexing completed in {index_time:.2f}s")
        
        # Run queries
        print("Running queries...")
        all_retrieved = []
        all_relevant = []
        latencies = []
        
        for query in self.queries:
            start = time.time()
            
            # Embed query
            query_embedding = await embedder.embed_query(query.query)
            
            # Retrieve
            results = await vector_store.search(
                query_vector=query_embedding,
                limit=20,
                filter=None
            )
            
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            
            # Extract document IDs from chunks
            retrieved_ids = []
            for result in results:
                # Get parent document ID from chunk metadata
                doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                if doc_id not in retrieved_ids:
                    retrieved_ids.append(doc_id)
            
            all_retrieved.append(retrieved_ids)
            all_relevant.append(query.relevant_doc_ids)
        
        # Calculate metrics
        retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
            all_retrieved, all_relevant
        )
        
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            latencies, total_time=sum(latencies) / 1000
        )
        
        # Create result
        result = BenchmarkResult(
            config_name=f"standard_rag_{provider}",
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            security_metrics=None,  # Not testing security here
            performance_metrics=performance_metrics,
            provider=provider,
            model="none",
            dataset=str(self.dataset_dir),
            notes="Standard RAG without contextualization"
        )
        
        # Cleanup
        await vector_store.delete_collection()
        
        return result
    
    async def benchmark_casr(
        self,
        provider: str,
        model: str,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    ) -> BenchmarkResult:
        """
        Benchmark CASR with contextualization.
        
        Args:
            provider: LLM provider for contextualization
            model: Model name
            embedding_provider: Embedding provider
            
        Returns:
            BenchmarkResult with metrics
        """
        print(f"\n=== Benchmarking CASR ({provider}/{model}) ===")
        
        # Initialize components WITH contextualizer
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        contextualizer = Contextualizer(provider=provider, model=model)
        embedder = Embedder(provider=embedding_provider)
        
        # Create temporary vector store
        store_config = VectorStoreConfig(
            store_type="chroma",
            collection_name=f"benchmark_casr_{provider}",
            persist_directory=str(self.results_dir / "temp_stores")
        )
        vector_store = ChromaVectorStore(store_config)
        
        # Index documents with contextualization
        print("Indexing documents (with contextualization)...")
        start_index = time.time()
        context_lengths = []
        context_times = []
        
        for i, doc in enumerate(self.documents):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(self.documents)} documents...")
            
            chunks = chunker.chunk_document(doc)
            
            # Add context to each chunk
            contextualized_chunks = []
            for chunk in chunks:
                ctx_start = time.time()
                context = await contextualizer.generate_context(doc, chunk)
                ctx_time = (time.time() - ctx_start) * 1000
                
                context_times.append(ctx_time)
                context_lengths.append(len(context))
                
                # Combine chunk with context for embedding
                contextualized_content = f"{context}\n\n{chunk.content}"
                contextualized_chunks.append(contextualized_content)
            
            # Embed contextualized chunks
            embeddings = await embedder.embed_texts(contextualized_chunks)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding
            
            await vector_store.add_documents(chunks)
        
        index_time = time.time() - start_index
        print(f"Indexing completed in {index_time:.2f}s")
        
        # Run queries
        print("Running queries...")
        all_retrieved = []
        all_relevant = []
        latencies = []
        
        for query in self.queries:
            start = time.time()
            
            # Embed query
            query_embedding = await embedder.embed_query(query.query)
            
            # Retrieve
            results = await vector_store.search(
                query_vector=query_embedding,
                limit=20,
                filter=None
            )
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Extract document IDs from chunks
            retrieved_ids = []
            for result in results:
                doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                if doc_id not in retrieved_ids:
                    retrieved_ids.append(doc_id)
            
            all_retrieved.append(retrieved_ids)
            all_relevant.append(query.relevant_doc_ids)
        
        # Calculate metrics
        retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
            all_retrieved, all_relevant
        )
        
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(
            latencies, total_time=sum(latencies) / 1000
        )
        
        # Create result with contextual metrics
        result = BenchmarkResult(
            config_name=f"casr_{provider}_{model}",
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            security_metrics=None,
            performance_metrics=performance_metrics,
            contextual_metrics=ContextualMetrics(
                with_context_recall=retrieval_metrics.recall_at_k.get(10, 0.0),
                without_context_recall=0.0,  # Will be filled by comparison
                context_improvement=0.0,
                avg_context_length=sum(context_lengths) / len(context_lengths) if context_lengths else 0,
                context_generation_time_ms=sum(context_times) / len(context_times) if context_times else 0,
                context_relevance_score=0.0
            ),
            provider=provider,
            model=model,
            dataset=str(self.dataset_dir),
            notes="CASR with contextualization"
        )
        
        # Cleanup
        await vector_store.delete_collection()
        
        return result
    
    async def compare_all_providers(self) -> Dict[str, Any]:
        """
        Compare all providers for both standard RAG and CASR.
        
        Returns:
            Comparison results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PROVIDER COMPARISON")
        print("="*80)
        
        results = {
            "standard_rag": {},
            "casr": {},
            "comparison": {}
        }
        
        # Model mapping for each provider
        provider_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.3-70b-specdec"
        }
        
        for provider in self.providers:
            try:
                # Benchmark standard RAG
                standard_result = await self.benchmark_standard_rag(provider)
                results["standard_rag"][provider] = standard_result.to_dict()
                
                # Benchmark CASR
                model = provider_models.get(provider, "default")
                casr_result = await self.benchmark_casr(provider, model)
                results["casr"][provider] = casr_result.to_dict()
                
                # Calculate improvement
                standard_recall = standard_result.retrieval_metrics.recall_at_k.get(10, 0.0)
                casr_recall = casr_result.retrieval_metrics.recall_at_k.get(10, 0.0)
                
                improvement = 0.0
                if standard_recall > 0:
                    improvement = ((casr_recall - standard_recall) / standard_recall) * 100
                
                results["comparison"][provider] = {
                    "standard_recall@10": standard_recall,
                    "casr_recall@10": casr_recall,
                    "improvement_percent": improvement,
                    "standard_latency_p95": standard_result.performance_metrics.p95_latency_ms,
                    "casr_latency_p95": casr_result.performance_metrics.p95_latency_ms,
                    "context_overhead_ms": casr_result.contextual_metrics.context_generation_time_ms
                }
                
                print(f"\n{provider.upper()} Results:")
                print(f"  Standard Recall@10: {standard_recall:.4f}")
                print(f"  CASR Recall@10: {casr_recall:.4f}")
                print(f"  Improvement: {improvement:+.2f}%")
                
            except Exception as e:
                print(f"Error benchmarking {provider}: {e}")
                results["comparison"][provider] = {"error": str(e)}
        
        # Save results
        results_file = self.results_dir / "retrieval_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    async def cross_domain_benchmark(self) -> Dict[str, Any]:
        """
        Benchmark cross-domain retrieval performance.
        
        Returns:
            Cross-domain results
        """
        print("\n=== Cross-Domain Retrieval Benchmark ===")
        
        domains = ["finance", "healthcare", "legal", "technology"]
        results = {}
        
        for domain in domains:
            domain_queries = [q for q in self.queries if q.domain == domain]
            if not domain_queries:
                continue
            
            print(f"\nTesting {domain} domain ({len(domain_queries)} queries)...")
            
            # Use CASR with best provider (based on initial tests)
            casr_result = await self.benchmark_casr("openai", "gpt-4o")
            
            results[domain] = {
                "num_queries": len(domain_queries),
                "recall@10": casr_result.retrieval_metrics.recall_at_k.get(10, 0.0),
                "precision@10": casr_result.retrieval_metrics.precision_at_k.get(10, 0.0),
                "mrr": casr_result.retrieval_metrics.mrr,
                "avg_latency_ms": casr_result.performance_metrics.avg_latency_ms
            }
        
        # Save results
        results_file = self.results_dir / "cross_domain_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nCross-domain results saved to {results_file}")
        
        return results


async def main():
    """Run retrieval benchmarks."""
    dataset_dir = Path(__file__).parent / "data" / "test_dataset"
    results_dir = Path(__file__).parent / "results" / "retrieval"
    
    benchmark = RetrievalBenchmark(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        providers=["openai", "anthropic", "gemini", "groq"]
    )
    
    # Run comprehensive comparison
    await benchmark.compare_all_providers()
    
    # Run cross-domain tests
    await benchmark.cross_domain_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
