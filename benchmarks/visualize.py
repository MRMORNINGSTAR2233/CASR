"""
Visualization utilities for CASR benchmark results.

Generates:
1. Provider comparison charts
2. Performance graphs
3. Security audit reports
4. Scalability plots
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
from datetime import datetime


class BenchmarkVisualizer:
    """Generate visualizations from benchmark results."""
    
    def __init__(self, results_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        self.viz_dir = results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def load_results(self, benchmark_type: str) -> Dict[str, Any]:
        """Load results for a specific benchmark type."""
        results_file = self.results_dir / f"{benchmark_type}_comprehensive.json"
        
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_provider_comparison(self, retrieval_results: Dict[str, Any]):
        """Plot provider comparison for retrieval quality."""
        if "comparison" not in retrieval_results:
            print("No comparison data found")
            return
        
        comparison = retrieval_results["comparison"]
        
        # Extract data
        providers = []
        standard_recall = []
        casr_recall = []
        improvement = []
        
        for provider, metrics in comparison.items():
            if "error" in metrics:
                continue
            
            providers.append(provider.upper())
            standard_recall.append(metrics.get("standard_recall@10", 0) * 100)
            casr_recall.append(metrics.get("casr_recall@10", 0) * 100)
            improvement.append(metrics.get("improvement_percent", 0))
        
        if not providers:
            print("No valid provider data")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Recall comparison
        x = np.arange(len(providers))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, standard_recall, width, label='Standard RAG', color='#FF6B6B')
        bars2 = ax1.bar(x + width/2, casr_recall, width, label='CASR', color='#4ECDC4')
        
        ax1.set_xlabel('Provider', fontweight='bold')
        ax1.set_ylabel('Recall@10 (%)', fontweight='bold')
        ax1.set_title('Retrieval Quality: Standard RAG vs CASR', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(providers)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Improvement
        colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvement]
        bars3 = ax2.bar(providers, improvement, color=colors)
        
        ax2.set_xlabel('Provider', fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontweight='bold')
        ax2.set_title('CASR Improvement over Standard RAG', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars3, improvement):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%',
                    ha='center', va='bottom' if imp > 0 else 'top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "provider_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Provider comparison chart saved")
    
    def plot_performance_metrics(self, performance_results: Dict[str, Any]):
        """Plot performance metrics."""
        if "latency" not in performance_results:
            print("No latency data found")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Latency distribution
        latency = performance_results["latency"]["total_latency"]
        metrics = ['avg', 'p50', 'p95', 'p99', 'max']
        values = [latency[f"{m}_ms"] for m in metrics]
        
        bars = ax1.bar(metrics, values, color='#3498DB')
        ax1.set_xlabel('Metric', fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontweight='bold')
        ax1.set_title('Query Latency Distribution', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.1f}ms',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Throughput by concurrency
        if "throughput" in performance_results and "summary" in performance_results["throughput"]:
            summary = performance_results["throughput"]["summary"]
            concurrent_users = [s["concurrent_users"] for s in summary]
            qps = [s["qps"] for s in summary]
            
            ax2.plot(concurrent_users, qps, marker='o', linewidth=2, markersize=8, color='#2ECC71')
            ax2.set_xlabel('Concurrent Users', fontweight='bold')
            ax2.set_ylabel('Throughput (QPS)', fontweight='bold')
            ax2.set_title('Throughput Scalability', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            # Add value labels
            for x, y in zip(concurrent_users, qps):
                ax2.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Scalability by document count
        if "scalability" in performance_results and "scaling_factor" in performance_results["scalability"]:
            scaling = performance_results["scalability"]["scaling_factor"]
            doc_counts = [s["doc_count"] for s in scaling]
            qps_values = [s["qps"] for s in scaling]
            
            ax3.plot(doc_counts, qps_values, marker='s', linewidth=2, markersize=8, color='#E67E22')
            ax3.set_xlabel('Document Count', fontweight='bold')
            ax3.set_ylabel('Throughput (QPS)', fontweight='bold')
            ax3.set_title('Document Count Scalability', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)
            
            for x, y in zip(doc_counts, qps_values):
                ax3.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Latency components
        if "embedding_time" in performance_results["latency"] and "retrieval_time" in performance_results["latency"]:
            embed_time = performance_results["latency"]["embedding_time"]["avg_ms"]
            retrieval_time = performance_results["latency"]["retrieval_time"]["avg_ms"]
            
            components = ['Embedding', 'Retrieval', 'Total']
            times = [embed_time, retrieval_time, embed_time + retrieval_time]
            colors = ['#9B59B6', '#1ABC9C', '#34495E']
            
            bars = ax4.bar(components, times, color=colors)
            ax4.set_ylabel('Time (ms)', fontweight='bold')
            ax4.set_title('Latency Breakdown', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            for bar, time in zip(bars, times):
                ax4.text(bar.get_x() + bar.get_width()/2., time,
                        f'{time:.1f}ms',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Performance metrics chart saved")
    
    def plot_security_metrics(self, security_results: Dict[str, Any]):
        """Plot security enforcement metrics."""
        if "rbac" not in security_results:
            print("No RBAC data found")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: RBAC accuracy by role
        rbac_by_role = security_results["rbac"]["by_role"]
        roles = list(rbac_by_role.keys())
        accuracies = [rbac_by_role[role]["rbac_accuracy"] * 100 for role in roles]
        
        bars = ax1.barh(roles, accuracies, color='#27AE60')
        ax1.set_xlabel('RBAC Accuracy (%)', fontweight='bold')
        ax1.set_ylabel('Role', fontweight='bold')
        ax1.set_title('RBAC Enforcement Accuracy by Role', fontsize=12, fontweight='bold')
        ax1.set_xlim([0, 105])
        ax1.grid(axis='x', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            ax1.text(acc + 1, bar.get_y() + bar.get_height()/2.,
                    f'{acc:.1f}%',
                    va='center', fontsize=9)
        
        # Plot 2: Security violations
        violation_types = ['Unauthorized\nLeaked', 'False\nPositives', 'False\nNegatives']
        violation_counts = [
            sum(rbac_by_role[role]["unauthorized_leaked"] for role in roles),
            sum(rbac_by_role[role]["false_positives"] for role in roles),
            sum(rbac_by_role[role]["false_negatives"] for role in roles)
        ]
        
        colors = ['#E74C3C', '#F39C12', '#E67E22']
        bars = ax2.bar(violation_types, violation_counts, color=colors)
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Security Violations by Type', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, violation_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., count,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Audit logging coverage
        if "audit" in security_results:
            audit = security_results["audit"]
            coverage = audit.get("coverage", 0) * 100
            completeness = audit.get("log_completeness", 0) * 100
            
            metrics = ['Coverage', 'Completeness']
            values = [coverage, completeness]
            
            bars = ax3.bar(metrics, values, color='#3498DB')
            ax3.set_ylabel('Percentage (%)', fontweight='bold')
            ax3.set_title('Audit Logging Metrics', fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 105])
            ax3.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Overall security score
        overall_accuracy = security_results["rbac"]["overall"]["average_rbac_accuracy"] * 100
        total_leaks = security_results["rbac"]["overall"]["total_unauthorized_leaks"]
        
        # Security score (100 - leak penalty)
        security_score = max(0, overall_accuracy - (total_leaks * 5))  # -5% per leak
        
        ax4.pie([security_score, 100 - security_score],
               labels=['Secure', 'Risk'],
               colors=['#2ECC71', '#E74C3C'],
               autopct='%1.1f%%',
               startangle=90)
        ax4.set_title(f'Overall Security Score\n(RBAC Accuracy: {overall_accuracy:.1f}%, Leaks: {total_leaks})',
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "security_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Security metrics chart saved")
    
    def plot_adversarial_results(self, adversarial_results: Dict[str, Any]):
        """Plot adversarial testing results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Privilege escalation attacks
        if "privilege_escalation" in adversarial_results:
            priv_esc = adversarial_results["privilege_escalation"]
            attacks = priv_esc.get("attacks", [])
            
            if attacks:
                attack_types = [a["attack_type"].replace("_", "\n") for a in attacks]
                success_rates = [a["success_rate"] * 100 for a in attacks]
                
                colors = ['#2ECC71' if rate > 95 else '#E74C3C' for rate in success_rates]
                bars = ax1.barh(attack_types, success_rates, color=colors)
                ax1.set_xlabel('Block Success Rate (%)', fontweight='bold')
                ax1.set_title('Privilege Escalation Defense', fontsize=12, fontweight='bold')
                ax1.set_xlim([0, 105])
                ax1.grid(axis='x', alpha=0.3)
                
                for bar, rate in zip(bars, success_rates):
                    ax1.text(rate + 1, bar.get_y() + bar.get_height()/2.,
                            f'{rate:.1f}%',
                            va='center', fontsize=9)
        
        # Plot 2: Cross-domain isolation
        if "cross_domain_leakage" in adversarial_results:
            cross_domain = adversarial_results["cross_domain_leakage"]
            domain_tests = cross_domain.get("domain_tests", [])
            
            if domain_tests:
                domains = [t["domain"] for t in domain_tests]
                isolation_rates = [t["isolation_rate"] * 100 for t in domain_tests]
                
                colors = ['#2ECC71' if rate > 95 else '#E74C3C' for rate in isolation_rates]
                bars = ax2.bar(domains, isolation_rates, color=colors)
                ax2.set_ylabel('Isolation Rate (%)', fontweight='bold')
                ax2.set_title('Cross-Domain Isolation', fontsize=12, fontweight='bold')
                ax2.set_ylim([0, 105])
                ax2.grid(axis='y', alpha=0.3)
                
                for bar, rate in zip(bars, isolation_rates):
                    ax2.text(bar.get_x() + bar.get_width()/2., rate,
                            f'{rate:.1f}%',
                            ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Injection attack defense
        if "injection_attacks" in adversarial_results:
            injection = adversarial_results["injection_attacks"]
            total = injection.get("total_attempts", 0)
            blocked = injection.get("blocked", 0)
            executed = injection.get("executed", 0)
            
            if total > 0:
                categories = ['Blocked', 'Executed\n(Safe)', 'Bypassed']
                counts = [blocked, executed - (total - blocked), max(0, total - blocked - executed)]
                colors = ['#2ECC71', '#F39C12', '#E74C3C']
                
                bars = ax3.bar(categories, counts, color=colors)
                ax3.set_ylabel('Count', fontweight='bold')
                ax3.set_title(f'Injection Attack Defense ({total} attacks)', fontsize=12, fontweight='bold')
                ax3.grid(axis='y', alpha=0.3)
                
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., count,
                                f'{int(count)}',
                                ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Edge case handling
        if "edge_cases" in adversarial_results:
            edge = adversarial_results["edge_cases"]
            handled = edge.get("handled_correctly", 0)
            errors = edge.get("errors", 0)
            total = handled + errors
            
            if total > 0:
                labels = ['Handled\nCorrectly', 'Errors']
                sizes = [handled, errors]
                colors = ['#2ECC71', '#E74C3C']
                explode = (0.1, 0)
                
                ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
                ax4.set_title(f'Edge Case Handling ({total} cases)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "adversarial_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Adversarial results chart saved")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n=== Generating Visualizations ===\n")
        
        # Retrieval visualizations
        retrieval_results = self.load_results("retrieval/retrieval_comparison")
        if retrieval_results:
            self.plot_provider_comparison(retrieval_results)
        
        # Performance visualizations
        performance_results = self.load_results("performance/performance")
        if performance_results:
            self.plot_performance_metrics(performance_results)
        
        # Security visualizations
        security_results = self.load_results("security/security")
        if security_results:
            self.plot_security_metrics(security_results)
        
        # Adversarial visualizations
        adversarial_results = self.load_results("adversarial/adversarial")
        if adversarial_results:
            self.plot_adversarial_results(adversarial_results)
        
        print(f"\n✅ All visualizations saved to {self.viz_dir}")


def main():
    """Generate visualizations from latest results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CASR benchmark visualizations")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing benchmark results"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    visualizer = BenchmarkVisualizer(results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
