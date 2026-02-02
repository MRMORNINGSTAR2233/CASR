# CASR Benchmarking - Execution Guide

This guide provides step-by-step instructions to run the comprehensive CASR evaluation suite.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Ensure you're in the CASR project directory
   cd /Users/akshaykumar/Documents/Projects/CASR
   
   # Install all dependencies including benchmarking tools
   pip install -e .
   ```

2. **API Keys Configuration**
   
   Make sure your `.env` file contains all required API keys:
   ```bash
   # LLM Providers (at least one required for each benchmark)
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=...
   GROQ_API_KEY=gsk_...
   
   # Optional: Embedding providers
   COHERE_API_KEY=...
   ```

## Quick Start (Full Suite)

Run all benchmarks with a single command:

```bash
python benchmarks/run_benchmarks.py
```

**Expected Duration**: 30-60 minutes (depending on API rate limits and hardware)

**Output**: 
- Results in `benchmarks/results/run_YYYYMMDD_HHMMSS/`
- Comprehensive report: `BENCHMARK_REPORT.md`
- JSON results: `comprehensive_results.json`

## Step-by-Step Execution

### Step 1: Generate Test Dataset

```bash
python benchmarks/dataset_generator.py
```

**Output**:
- 400 documents (100 per domain: finance, healthcare, legal, tech)
- 200 queries (50 per domain)
- Files in `benchmarks/data/test_dataset/`

**Duration**: ~2 minutes

### Step 2: Retrieval Quality Benchmarks

```bash
python benchmarks/retrieval_benchmark.py
```

**Tests**:
- Standard RAG vs CASR comparison
- Multi-provider evaluation (OpenAI, Anthropic, Gemini, Groq)
- Cross-domain performance

**Duration**: ~15-20 minutes

**Key Metrics**:
- Recall@10, Precision@10
- MRR, NDCG@10
- Context improvement percentage

### Step 3: Security Enforcement Tests

```bash
python benchmarks/security_benchmark.py
```

**Tests**:
- RBAC enforcement (7 user roles)
- ABAC policy evaluation
- Audit logging coverage

**Duration**: ~10-15 minutes

**Critical Metrics**:
- Unauthorized leaks (should be 0)
- RBAC accuracy (should be 100%)
- Audit coverage (should be 100%)

### Step 4: Performance Benchmarks

```bash
python benchmarks/performance_benchmark.py
```

**Tests**:
- Latency distribution (avg, p50, p95, p99)
- Throughput with concurrent users (1, 5, 10, 20, 50)
- Scalability with document count (100, 500, 1K, 2K)

**Duration**: ~10-15 minutes

**Target Metrics**:
- Avg latency: <200ms
- P95 latency: <500ms
- Throughput: >50 QPS

### Step 5: Adversarial Security Tests

```bash
python benchmarks/adversarial_benchmark.py
```

**Tests**:
- Privilege escalation attempts
- Cross-domain information leakage
- Injection attack resistance (13 patterns)
- Edge case handling

**Duration**: ~10-15 minutes

**Success Criteria**:
- 0 privilege escalation successes
- >99% domain isolation
- >95% injection defense rate

### Step 6: Generate Visualizations

```bash
# Replace with your actual results directory
python benchmarks/visualize.py benchmarks/results/run_20260203_123456
```

**Output**: 
- `visualizations/provider_comparison.png`
- `visualizations/performance_metrics.png`
- `visualizations/security_metrics.png`
- `visualizations/adversarial_results.png`

**Duration**: ~1 minute

## Running Specific Benchmark Categories

### Only Retrieval Tests

```bash
python -c "
import asyncio
from pathlib import Path
from benchmarks.retrieval_benchmark import RetrievalBenchmark

async def main():
    benchmark = RetrievalBenchmark(
        dataset_dir=Path('benchmarks/data/test_dataset'),
        results_dir=Path('benchmarks/results/retrieval_only'),
        providers=['openai', 'gemini']  # Subset of providers
    )
    await benchmark.compare_all_providers()

asyncio.run(main())
"
```

### Only Security Tests

```bash
python -c "
import asyncio
from pathlib import Path
from benchmarks.security_benchmark import SecurityBenchmark

async def main():
    benchmark = SecurityBenchmark(
        dataset_dir=Path('benchmarks/data/test_dataset'),
        results_dir=Path('benchmarks/results/security_only')
    )
    await benchmark.run_all_security_tests()

asyncio.run(main())
"
```

## Expected Results Summary

### Retrieval Quality
```
Provider Comparison:
├── OpenAI:      Recall@10: 0.75-0.85, Improvement: +20-30%
├── Anthropic:   Recall@10: 0.70-0.80, Improvement: +15-25%
├── Gemini:      Recall@10: 0.72-0.82, Improvement: +18-28%
└── Groq:        Recall@10: 0.68-0.78, Improvement: +15-25%
```

### Security
```
RBAC Enforcement:
├── Accuracy:         100%
├── Unauthorized Leaks: 0
├── False Positives:   0-5
└── Audit Coverage:    100%
```

### Performance
```
Latency:
├── Average:    150-200ms
├── P95:        250-400ms
└── P99:        350-600ms

Throughput:
├── 1 user:     80-120 QPS
├── 10 users:   50-80 QPS
└── 50 users:   20-40 QPS
```

### Adversarial
```
Security Defense:
├── Privilege Escalation: 0/40 successful (100% blocked)
├── Cross-Domain Leakage: <1% (>99% isolation)
├── Injection Attacks:    13/13 blocked (100%)
└── Edge Cases:          10/10 handled (100%)
```

## Interpreting Results

### 1. Check BENCHMARK_REPORT.md

```bash
cat benchmarks/results/run_YYYYMMDD_HHMMSS/BENCHMARK_REPORT.md
```

Look for:
- Overall Status: Should be "COMPLETED"
- Key Metrics summary
- Critical security issues (should be 0)

### 2. Review JSON Results

```bash
# View comprehensive results
cat benchmarks/results/run_YYYYMMDD_HHMMSS/comprehensive_results.json | jq .summary
```

### 3. Examine Visualizations

Open PNG files in `visualizations/` directory to see:
- Provider performance comparison
- Latency and throughput graphs
- Security metrics dashboard
- Adversarial test results

## Troubleshooting

### Issue: API Rate Limits

**Symptom**: Requests failing with 429 errors

**Solution**: 
```bash
# Run with delays
export SLEEP_BETWEEN_REQUESTS=1.0

# Or reduce test size in run_benchmarks.py:
# num_docs_per_domain=50  # instead of 100
# num_queries_per_domain=25  # instead of 50
```

### Issue: Out of Memory

**Symptom**: Process killed or MemoryError

**Solution**:
```bash
# Reduce document count
python benchmarks/dataset_generator.py  # Edit num_docs_per_domain=50

# Or run benchmarks individually instead of full suite
```

### Issue: Missing Dependencies

**Symptom**: ImportError for matplotlib, seaborn, or psutil

**Solution**:
```bash
pip install matplotlib seaborn psutil
```

### Issue: ChromaDB Persistence Errors

**Symptom**: "Collection already exists" or persistence errors

**Solution**:
```bash
# Clean up temporary stores
rm -rf benchmarks/results/*/temp_stores/
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: CASR Benchmarks

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install matplotlib seaborn psutil
      
      - name: Run benchmarks
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          python benchmarks/run_benchmarks.py
      
      - name: Generate visualizations
        run: |
          LATEST_RUN=$(ls -td benchmarks/results/run_* | head -1)
          python benchmarks/visualize.py $LATEST_RUN
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results-${{ github.run_number }}
          path: benchmarks/results/run_*/
          retention-days: 90
      
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        run: |
          LATEST_RUN=$(ls -td benchmarks/results/run_* | head -1)
          cat $LATEST_RUN/BENCHMARK_REPORT.md >> $GITHUB_STEP_SUMMARY
```

## Performance Optimization Tips

1. **Parallel Provider Testing**: The benchmark runs providers sequentially. For faster execution, modify `retrieval_benchmark.py` to use `asyncio.gather()`.

2. **Reduce Test Dataset**: For quick validation:
   ```python
   # In dataset_generator.py
   num_docs_per_domain=25  # instead of 100
   num_queries_per_domain=10  # instead of 50
   ```

3. **Skip Dataset Generation**: If dataset already exists:
   ```bash
   python benchmarks/run_benchmarks.py --skip-dataset
   ```

4. **Local LLMs**: For unlimited testing, use local models:
   ```python
   # In benchmark configs
   providers = ["ollama"]  # Add Ollama support
   ```

## Next Steps After Benchmarking

1. **Analyze Results**: Identify bottlenecks and security issues
2. **Optimize**: Focus on areas with poor performance
3. **Re-benchmark**: Validate improvements
4. **Document**: Update research paper with findings
5. **Publish**: Share results and visualizations

## Questions?

- Check `benchmarks/README.md` for detailed documentation
- Review individual benchmark files for implementation details
- Open an issue on GitHub for support

---

**Created**: February 2026  
**Last Updated**: February 2026  
**Version**: 1.0.0
