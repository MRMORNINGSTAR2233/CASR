"""
End-to-End Benchmark Runner for CASR.

Orchestrates all benchmarks:
1. Dataset generation
2. Retrieval quality benchmarks
3. Security enforcement tests
4. Performance benchmarks
5. Adversarial tests
6. Results aggregation and reporting
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset_generator import DatasetGenerator
from benchmarks.retrieval_benchmark import RetrievalBenchmark
from benchmarks.security_benchmark import SecurityBenchmark
from benchmarks.performance_benchmark import PerformanceBenchmark
from benchmarks.adversarial_benchmark import AdversarialBenchmark


class BenchmarkRunner:
    """Orchestrate all CASR benchmarks."""
    
    def __init__(
        self,
        results_base_dir: Path,
        skip_dataset_generation: bool = False
    ):
        """
        Initialize benchmark runner.
        
        Args:
            results_base_dir: Base directory for all results
            skip_dataset_generation: Skip dataset generation if already exists
        """
        self.results_base_dir = results_base_dir
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.skip_dataset_generation = skip_dataset_generation
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_base_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset directory
        self.dataset_dir = Path(__file__).parent / "data" / "test_dataset"
        
        # Results will go in run-specific directory
        self.results = {
            "run_id": self.run_timestamp,
            "start_time": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {}
        }
    
    def _print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")
    
    def _print_progress(self, step: int, total: int, description: str):
        """Print progress."""
        print(f"\n[{step}/{total}] {description}")
        print("-" * 80)
    
    async def step1_generate_dataset(self) -> bool:
        """Generate test dataset."""
        self._print_progress(1, 6, "Dataset Generation")
        
        if self.skip_dataset_generation and self.dataset_dir.exists():
            print("  ⏭️  Skipping dataset generation (using existing dataset)")
            try:
                docs, queries = DatasetGenerator.load_dataset(self.dataset_dir)
                print(f"  ✅ Loaded existing dataset: {len(docs)} docs, {len(queries)} queries")
                return True
            except Exception as e:
                print(f"  ⚠️  Error loading existing dataset: {e}")
                print("  Generating new dataset...")
        
        try:
            generator = DatasetGenerator(seed=42)
            
            print("  📝 Generating documents (100 per domain)...")
            documents = generator.generate_documents(num_docs_per_domain=100)
            
            print("  🔍 Generating queries (50 per domain)...")
            queries = generator.generate_queries(num_queries_per_domain=50)
            
            print("  💾 Saving dataset...")
            generator.save_dataset(self.dataset_dir)
            
            self.results["benchmarks"]["dataset_generation"] = {
                "num_documents": len(documents),
                "num_queries": len(queries),
                "domains": ["finance", "healthcare", "legal", "technology"],
                "status": "completed"
            }
            
            print(f"  ✅ Dataset generated successfully")
            return True
        
        except Exception as e:
            print(f"  ❌ Dataset generation failed: {e}")
            self.results["benchmarks"]["dataset_generation"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    async def step2_retrieval_benchmarks(self) -> bool:
        """Run retrieval quality benchmarks."""
        self._print_progress(2, 6, "Retrieval Quality Benchmarks")
        
        try:
            results_dir = self.run_dir / "retrieval"
            benchmark = RetrievalBenchmark(
                dataset_dir=self.dataset_dir,
                results_dir=results_dir,
                providers=["openai", "anthropic", "gemini", "groq"]
            )
            
            print("  🔄 Running provider comparison...")
            comparison_results = await benchmark.compare_all_providers()
            
            print("  🌐 Running cross-domain tests...")
            cross_domain_results = await benchmark.cross_domain_benchmark()
            
            self.results["benchmarks"]["retrieval"] = {
                "provider_comparison": comparison_results.get("comparison", {}),
                "cross_domain": cross_domain_results,
                "status": "completed"
            }
            
            print(f"  ✅ Retrieval benchmarks completed")
            return True
        
        except Exception as e:
            print(f"  ❌ Retrieval benchmarks failed: {e}")
            self.results["benchmarks"]["retrieval"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    async def step3_security_benchmarks(self) -> bool:
        """Run security enforcement benchmarks."""
        self._print_progress(3, 6, "Security Enforcement Benchmarks")
        
        try:
            results_dir = self.run_dir / "security"
            benchmark = SecurityBenchmark(
                dataset_dir=self.dataset_dir,
                results_dir=results_dir
            )
            
            print("  🔒 Running comprehensive security tests...")
            security_results = await benchmark.run_all_security_tests()
            
            self.results["benchmarks"]["security"] = {
                "rbac_accuracy": security_results["rbac"]["overall"]["average_rbac_accuracy"],
                "unauthorized_leaks": security_results["rbac"]["overall"]["total_unauthorized_leaks"],
                "audit_coverage": security_results["audit"]["coverage"],
                "status": "completed"
            }
            
            print(f"  ✅ Security benchmarks completed")
            return True
        
        except Exception as e:
            print(f"  ❌ Security benchmarks failed: {e}")
            self.results["benchmarks"]["security"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    async def step4_performance_benchmarks(self) -> bool:
        """Run performance and scalability benchmarks."""
        self._print_progress(4, 6, "Performance and Scalability Benchmarks")
        
        try:
            results_dir = self.run_dir / "performance"
            benchmark = PerformanceBenchmark(
                dataset_dir=self.dataset_dir,
                results_dir=results_dir
            )
            
            print("  ⚡ Running performance tests...")
            performance_results = await benchmark.run_all_performance_tests()
            
            self.results["benchmarks"]["performance"] = {
                "avg_latency_ms": performance_results["latency"]["total_latency"]["avg_ms"],
                "p95_latency_ms": performance_results["latency"]["total_latency"]["p95_ms"],
                "throughput_qps": performance_results["throughput"]["by_concurrency"].get("10", {}).get("queries_per_second", 0),
                "status": "completed"
            }
            
            print(f"  ✅ Performance benchmarks completed")
            return True
        
        except Exception as e:
            print(f"  ❌ Performance benchmarks failed: {e}")
            self.results["benchmarks"]["performance"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    async def step5_adversarial_benchmarks(self) -> bool:
        """Run adversarial security tests."""
        self._print_progress(5, 6, "Adversarial Security Tests")
        
        try:
            results_dir = self.run_dir / "adversarial"
            benchmark = AdversarialBenchmark(
                dataset_dir=self.dataset_dir,
                results_dir=results_dir
            )
            
            print("  🛡️  Running adversarial tests...")
            adversarial_results = await benchmark.run_all_adversarial_tests()
            
            self.results["benchmarks"]["adversarial"] = {
                "privilege_escalation_leaks": adversarial_results["privilege_escalation"]["failed_blocks"],
                "cross_domain_leakage": adversarial_results["cross_domain_leakage"]["total_leakage"],
                "injection_bypasses": adversarial_results["injection_attacks"]["total_attempts"] - adversarial_results["injection_attacks"]["blocked"],
                "status": "completed"
            }
            
            print(f"  ✅ Adversarial tests completed")
            return True
        
        except Exception as e:
            print(f"  ❌ Adversarial tests failed: {e}")
            self.results["benchmarks"]["adversarial"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def step6_generate_report(self) -> bool:
        """Generate comprehensive benchmark report."""
        self._print_progress(6, 6, "Generating Comprehensive Report")
        
        try:
            # Calculate summary statistics
            self.results["end_time"] = datetime.now().isoformat()
            
            # Create summary
            summary = {
                "overall_status": "completed",
                "failed_benchmarks": [],
                "key_metrics": {}
            }
            
            # Check for failures
            for bench_name, bench_results in self.results["benchmarks"].items():
                if bench_results.get("status") == "failed":
                    summary["failed_benchmarks"].append(bench_name)
                    summary["overall_status"] = "partial"
            
            # Extract key metrics
            if "retrieval" in self.results["benchmarks"]:
                retrieval = self.results["benchmarks"]["retrieval"]
                if "provider_comparison" in retrieval:
                    best_provider = max(
                        retrieval["provider_comparison"].items(),
                        key=lambda x: x[1].get("casr_recall@10", 0) if isinstance(x[1], dict) else 0
                    )
                    summary["key_metrics"]["best_retrieval_provider"] = {
                        "provider": best_provider[0],
                        "recall@10": best_provider[1].get("casr_recall@10", 0) if isinstance(best_provider[1], dict) else 0
                    }
            
            if "security" in self.results["benchmarks"]:
                security = self.results["benchmarks"]["security"]
                summary["key_metrics"]["security"] = {
                    "rbac_accuracy": security.get("rbac_accuracy", 0),
                    "unauthorized_leaks": security.get("unauthorized_leaks", 0),
                    "audit_coverage": security.get("audit_coverage", 0)
                }
            
            if "performance" in self.results["benchmarks"]:
                perf = self.results["benchmarks"]["performance"]
                summary["key_metrics"]["performance"] = {
                    "avg_latency_ms": perf.get("avg_latency_ms", 0),
                    "p95_latency_ms": perf.get("p95_latency_ms", 0),
                    "throughput_qps": perf.get("throughput_qps", 0)
                }
            
            if "adversarial" in self.results["benchmarks"]:
                adv = self.results["benchmarks"]["adversarial"]
                total_critical_issues = (
                    adv.get("privilege_escalation_leaks", 0) +
                    adv.get("cross_domain_leakage", 0) +
                    adv.get("injection_bypasses", 0)
                )
                summary["key_metrics"]["adversarial"] = {
                    "total_critical_issues": total_critical_issues,
                    "security_status": "PASS" if total_critical_issues == 0 else "FAIL"
                }
            
            self.results["summary"] = summary
            
            # Save comprehensive results
            results_file = self.run_dir / "comprehensive_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"  💾 Results saved to {results_file}")
            
            # Generate markdown report
            self._generate_markdown_report()
            
            print(f"  ✅ Report generation completed")
            return True
        
        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")
            return False
    
    def _generate_markdown_report(self):
        """Generate markdown summary report."""
        report_file = self.run_dir / "BENCHMARK_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# CASR Comprehensive Benchmark Report\n\n")
            f.write(f"**Run ID:** {self.results['run_id']}\n\n")
            f.write(f"**Start Time:** {self.results['start_time']}\n\n")
            f.write(f"**End Time:** {self.results.get('end_time', 'N/A')}\n\n")
            
            summary = self.results.get("summary", {})
            f.write(f"**Overall Status:** {summary.get('overall_status', 'unknown').upper()}\n\n")
            
            if summary.get("failed_benchmarks"):
                f.write("**Failed Benchmarks:**\n")
                for bench in summary["failed_benchmarks"]:
                    f.write(f"- {bench}\n")
                f.write("\n")
            
            f.write("## Key Metrics\n\n")
            
            key_metrics = summary.get("key_metrics", {})
            
            # Retrieval metrics
            if "best_retrieval_provider" in key_metrics:
                f.write("### Retrieval Quality\n\n")
                best = key_metrics["best_retrieval_provider"]
                f.write(f"- **Best Provider:** {best['provider']}\n")
                f.write(f"- **Recall@10:** {best['recall@10']:.4f}\n\n")
            
            # Security metrics
            if "security" in key_metrics:
                f.write("### Security Enforcement\n\n")
                sec = key_metrics["security"]
                f.write(f"- **RBAC Accuracy:** {sec['rbac_accuracy']:.4f}\n")
                f.write(f"- **Unauthorized Leaks:** {sec['unauthorized_leaks']} ⚠️ (should be 0)\n")
                f.write(f"- **Audit Coverage:** {sec['audit_coverage']:.4f}\n\n")
            
            # Performance metrics
            if "performance" in key_metrics:
                f.write("### Performance\n\n")
                perf = key_metrics["performance"]
                f.write(f"- **Average Latency:** {perf['avg_latency_ms']:.2f}ms\n")
                f.write(f"- **P95 Latency:** {perf['p95_latency_ms']:.2f}ms\n")
                f.write(f"- **Throughput:** {perf['throughput_qps']:.2f} QPS\n\n")
            
            # Adversarial metrics
            if "adversarial" in key_metrics:
                f.write("### Adversarial Testing\n\n")
                adv = key_metrics["adversarial"]
                f.write(f"- **Total Critical Issues:** {adv['total_critical_issues']}\n")
                f.write(f"- **Security Status:** {adv['security_status']}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("Detailed results for each benchmark are available in the following files:\n\n")
            f.write("- Retrieval: `retrieval/retrieval_comparison.json`\n")
            f.write("- Security: `security/security_comprehensive.json`\n")
            f.write("- Performance: `performance/performance_comprehensive.json`\n")
            f.write("- Adversarial: `adversarial/adversarial_comprehensive.json`\n")
        
        print(f"  📄 Markdown report saved to {report_file}")
    
    async def run_all(self):
        """Run all benchmarks in sequence."""
        self._print_header("CASR COMPREHENSIVE BENCHMARK SUITE")
        
        print(f"Run ID: {self.run_timestamp}")
        print(f"Results Directory: {self.run_dir}")
        print()
        
        start_time = time.time()
        
        # Step 1: Generate dataset
        if not await self.step1_generate_dataset():
            print("\n❌ Benchmark suite failed at dataset generation")
            return
        
        # Step 2: Retrieval benchmarks
        await self.step2_retrieval_benchmarks()
        
        # Step 3: Security benchmarks
        await self.step3_security_benchmarks()
        
        # Step 4: Performance benchmarks
        await self.step4_performance_benchmarks()
        
        # Step 5: Adversarial tests
        await self.step5_adversarial_benchmarks()
        
        # Step 6: Generate report
        self.step6_generate_report()
        
        # Final summary
        duration = time.time() - start_time
        
        self._print_header("BENCHMARK SUITE COMPLETED")
        
        print(f"Total Duration: {duration/60:.2f} minutes")
        print(f"Results Directory: {self.run_dir}")
        print()
        
        summary = self.results.get("summary", {})
        
        if summary.get("overall_status") == "completed":
            print("✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        else:
            print("⚠️  SOME BENCHMARKS FAILED")
            if summary.get("failed_benchmarks"):
                print(f"   Failed: {', '.join(summary['failed_benchmarks'])}")
        
        print()
        print("Key Results:")
        
        key_metrics = summary.get("key_metrics", {})
        
        if "best_retrieval_provider" in key_metrics:
            best = key_metrics["best_retrieval_provider"]
            print(f"  🎯 Best Retrieval: {best['provider']} (Recall@10: {best['recall@10']:.4f})")
        
        if "security" in key_metrics:
            sec = key_metrics["security"]
            print(f"  🔒 RBAC Accuracy: {sec['rbac_accuracy']:.4f}")
            print(f"  ⚠️  Unauthorized Leaks: {sec['unauthorized_leaks']}")
        
        if "performance" in key_metrics:
            perf = key_metrics["performance"]
            print(f"  ⚡ P95 Latency: {perf['p95_latency_ms']:.2f}ms")
            print(f"  📈 Throughput: {perf['throughput_qps']:.2f} QPS")
        
        if "adversarial" in key_metrics:
            adv = key_metrics["adversarial"]
            status_emoji = "✅" if adv["security_status"] == "PASS" else "❌"
            print(f"  {status_emoji} Security Status: {adv['security_status']}")
        
        print()
        print(f"📊 View full report: {self.run_dir / 'BENCHMARK_REPORT.md'}")
        print()


async def main():
    """Main entry point for benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CASR comprehensive benchmarks")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Base directory for results (default: benchmarks/results)"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation if it already exists"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    runner = BenchmarkRunner(
        results_base_dir=results_dir,
        skip_dataset_generation=args.skip_dataset
    )
    
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
