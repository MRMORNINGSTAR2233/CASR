# CASR Comprehensive Benchmarking Suite

This directory contains a complete end-to-end benchmarking and evaluation framework for the CASR (Context-Aware Secure Retrieval) system.

## Overview

The benchmarking suite evaluates CASR across multiple dimensions:

1. **Retrieval Quality** - Accuracy metrics (Recall@K, Precision@K, MRR, NDCG)
2. **Security Enforcement** - RBAC/ABAC correctness, policy violations, audit logging
3. **Performance** - Latency, throughput, scalability
4. **Contextual Enhancement** - Impact of contextualization on retrieval
5. **Adversarial Robustness** - Resistance to attacks and edge cases
6. **Provider Comparison** - OpenAI, Anthropic, Gemini, Groq

## Structure

```
benchmarks/
├── __init__.py                  # Package initialization
├── README.md                    # This file
├── metrics.py                   # Comprehensive metrics calculation
├── dataset_generator.py         # Realistic test dataset generation
├── retrieval_benchmark.py       # Retrieval quality benchmarks
├── security_benchmark.py        # Security enforcement tests
├── performance_benchmark.py     # Performance and scalability tests
├── adversarial_benchmark.py     # Adversarial security testing
├── run_benchmarks.py            # Main orchestrator
├── visualize.py                 # Results visualization
├── data/                        # Test datasets
│   └── test_dataset/            # Generated test data
│       ├── documents.jsonl      # Multi-domain documents
│       ├── queries.jsonl        # Ground truth queries
│       └── dataset_stats.json   # Dataset statistics
└── results/                     # Benchmark results
    └── run_YYYYMMDD_HHMMSS/     # Timestamped run directory
        ├── comprehensive_results.json
        ├── BENCHMARK_REPORT.md
        ├── retrieval/           # Retrieval benchmark results
        ├── security/            # Security benchmark results
        ├── performance/         # Performance benchmark results
        ├── adversarial/         # Adversarial test results
        └── visualizations/      # Generated charts
```

## Quick Start

### 1. Install Dependencies

```bash
# Install additional benchmarking dependencies
pip install matplotlib seaborn psutil
```

### 2. Run Complete Benchmark Suite

```bash
# Run all benchmarks (will take 30-60 minutes)
python benchmarks/run_benchmarks.py

# Run with existing dataset (faster)
python benchmarks/run_benchmarks.py --skip-dataset

# Specify custom results directory
python benchmarks/run_benchmarks.py --results-dir ./my_results
```

### 3. Generate Visualizations

```bash
# Generate charts from results
python benchmarks/visualize.py benchmarks/results/run_YYYYMMDD_HHMMSS
```

## Individual Benchmarks

You can also run individual benchmark modules:

### Dataset Generation

```bash
python benchmarks/dataset_generator.py
```

Generates:
- 400 multi-domain documents (100 per domain: finance, healthcare, legal, tech)
- 200 ground truth queries (50 per domain)
- Security classifications (PUBLIC to TOP_SECRET)
- RBAC/ABAC attributes

### Retrieval Quality Benchmark

```bash
python benchmarks/retrieval_benchmark.py
```

Tests:
- **Standard RAG vs CASR** - Measures contextual enhancement impact
- **Multi-Provider Comparison** - OpenAI, Anthropic, Gemini, Groq
- **Cross-Domain Performance** - Retrieval quality per domain

Metrics:
- Recall@K (K = 1, 3, 5, 10, 20)
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)

### Security Enforcement Benchmark

```bash
python benchmarks/security_benchmark.py
```

Tests:
- **RBAC Enforcement** - Role-based access control across 7 user roles
- **ABAC Enforcement** - Attribute-based policies (department, region, clearance)
- **Audit Logging** - Completeness and coverage

Validates:
- Zero unauthorized document leakage
- Correct role-based filtering
- Policy condition enforcement
- Comprehensive audit trails

### Performance Benchmark

```bash
python benchmarks/performance_benchmark.py
```

Tests:
- **Latency Distribution** - Average, P50, P95, P99, Max
- **Throughput** - Queries per second with 1, 5, 10, 20, 50 concurrent users
- **Scalability** - Performance with 100, 500, 1K, 2K documents
- **Resource Usage** - Memory and CPU consumption

### Adversarial Security Tests

```bash
python benchmarks/adversarial_benchmark.py
```

Tests:
- **Privilege Escalation** - Public user accessing SECRET/TOP_SECRET
- **Cross-Domain Leakage** - Domain isolation enforcement
- **Injection Attacks** - SQL injection, XSS, command injection patterns
- **Edge Cases** - Empty queries, unicode, extremely long inputs

## Metrics Reference

### Retrieval Metrics

| Metric | Description | Range | Ideal |
|--------|-------------|-------|-------|
| **Recall@K** | Fraction of relevant docs in top-K | 0.0-1.0 | ≥0.8 |
| **Precision@K** | Fraction of top-K that are relevant | 0.0-1.0 | ≥0.7 |
| **MRR** | Mean reciprocal rank of first relevant doc | 0.0-1.0 | ≥0.6 |
| **NDCG@K** | Normalized discounted cumulative gain | 0.0-1.0 | ≥0.7 |
| **MAP** | Mean average precision | 0.0-1.0 | ≥0.6 |

### Security Metrics

| Metric | Description | Range | Ideal |
|--------|-------------|-------|-------|
| **RBAC Accuracy** | Correct access decisions | 0.0-1.0 | 1.0 |
| **Unauthorized Leaks** | Documents leaked to unauthorized users | 0-∞ | 0 |
| **False Positives** | Authorized users incorrectly blocked | 0-∞ | 0 |
| **False Negatives** | Unauthorized users granted access | 0-∞ | 0 |
| **Audit Coverage** | % of actions logged | 0.0-1.0 | 1.0 |

### Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Avg Latency** | Mean query response time | <200ms |
| **P95 Latency** | 95th percentile latency | <500ms |
| **P99 Latency** | 99th percentile latency | <1000ms |
| **Throughput** | Queries per second | >50 QPS |

## Expected Results

### Retrieval Quality

- **CASR Improvement**: 15-30% higher Recall@10 vs Standard RAG
- **Best Provider**: OpenAI/Gemini typically achieve highest accuracy
- **Context Impact**: Contextual embeddings improve retrieval by 20-40%

### Security

- **Zero Leakage**: System should block ALL unauthorized access attempts
- **RBAC Accuracy**: Should achieve 100% correct access decisions
- **Audit Completeness**: 100% coverage of security events

### Performance

- **Average Latency**: 100-200ms for typical queries
- **P95 Latency**: 200-400ms
- **Throughput**: 50-200 QPS depending on hardware
- **Scalability**: Near-linear scaling up to 10K documents

### Adversarial

- **Privilege Escalation**: 0 successful attacks
- **Cross-Domain Leakage**: >99% isolation
- **Injection Defense**: >95% blocked/safely handled

## Real-World Scenarios Tested

### 1. Enterprise Document Retrieval
- Multi-department access control
- Security classification enforcement
- Audit compliance requirements

### 2. Healthcare Information Systems
- HIPAA compliance scenarios
- Patient data isolation
- Role-based access (doctors, nurses, admin)

### 3. Financial Services
- Regulatory document access
- Client confidentiality
- Insider trading prevention

### 4. Legal Document Management
- Attorney-client privilege
- Case-based access control
- Confidential filing protection

## Interpreting Results

### Retrieval Report

```json
{
  "comparison": {
    "openai": {
      "standard_recall@10": 0.65,
      "casr_recall@10": 0.82,
      "improvement_percent": 26.15
    }
  }
}
```

**Interpretation**: CASR with OpenAI achieves 82% recall, a 26% improvement over standard RAG.

### Security Report

```json
{
  "overall": {
    "average_rbac_accuracy": 1.0,
    "total_unauthorized_leaks": 0,
    "critical_violations": 0
  }
}
```

**Interpretation**: Perfect security - no unauthorized access or data leakage.

### Performance Report

```json
{
  "latency": {
    "avg_ms": 156.3,
    "p95_ms": 287.5,
    "p99_ms": 421.8
  }
}
```

**Interpretation**: System meets <200ms average latency target, acceptable P95/P99.

## Visualization Outputs

The visualization module generates:

1. **provider_comparison.png** - Bar charts comparing Standard RAG vs CASR across providers
2. **performance_metrics.png** - Latency distribution, throughput, and scalability graphs
3. **security_metrics.png** - RBAC accuracy, violations, audit coverage
4. **adversarial_results.png** - Attack defense success rates, isolation metrics

## Customization

### Add New Domains

Edit `dataset_generator.py`:

```python
DOMAIN_TEMPLATES = {
    "your_domain": {
        "topics": [...],
        "entities": [...],
        "templates": [...]
    }
}
```

### Add New Metrics

Edit `metrics.py`:

```python
def calculate_custom_metric(self, data):
    # Your metric calculation
    return score
```

### Test Different Providers

Edit benchmark files:

```python
providers = ["openai", "anthropic", "gemini", "groq", "your_provider"]
```

## Troubleshooting

### Out of Memory

Reduce dataset size in `run_benchmarks.py`:

```python
documents = generator.generate_documents(num_docs_per_domain=50)  # Instead of 100
```

### API Rate Limits

Add delays between requests:

```python
import asyncio
await asyncio.sleep(0.5)  # 500ms delay
```

### Slow Execution

Run individual benchmarks instead of full suite:

```bash
# Just retrieval
python benchmarks/retrieval_benchmark.py

# Just security
python benchmarks/security_benchmark.py
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: CASR Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: |
          pip install -e .
          python benchmarks/run_benchmarks.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{casr_benchmarks_2025,
  title={CASR Comprehensive Benchmarking Suite},
  author={CASR Development Team},
  year={2025},
  url={https://github.com/MRMORNINGSTAR2233/CASR}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: https://github.com/MRMORNINGSTAR2233/CASR/issues
- Documentation: See main README.md

---

**Last Updated**: February 2026
**Version**: 1.0.0
