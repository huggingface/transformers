# Benchmarking v2

A comprehensive benchmarking framework for transformer models that supports multiple execution modes (eager, compiled, kernelized), detailed performance metrics collection, and structured output format.


## Quick Start

### Running All Benchmarks

```bash
# Run all benchmarks with default settings
python run_benchmarks.py

# Specify output directory
python run_benchmarks.py --output-dir my_results

# Run with custom parameters
python run_benchmarks.py \
    --warmup-iterations 5 \
    --measurement-iterations 10 \
    --num-tokens-to-generate 200
```

### Uploading Results to HuggingFace Dataset

You can automatically upload benchmark results to a HuggingFace Dataset for tracking and analysis:

```bash
# Upload to a public dataset with auto-generated run ID
python run_benchmarks.py --upload-to-hub username/benchmark-results

# Upload with a custom run ID for easy identification
python run_benchmarks.py --upload-to-hub username/benchmark-results --run-id experiment_v1

# Upload with custom HuggingFace token (if not set in environment)
python run_benchmarks.py --upload-to-hub username/benchmark-results --token hf_your_token_here
```

**Dataset Directory Structure:**
```
dataset_name/
├── 2025-01-15/
│   ├── runs/                       # Non-scheduled runs (manual, PR, etc.)
│   │   └── 123-1245151651/         # GitHub run number and ID
│   │       └── benchmark_results/
│   │           ├── benchmark_summary_20250115_143022.json
│   │           └── model-name/
│   │               └── model-name_benchmark_20250115_143022.json
│   └── benchmark_results_abc123de/ # Scheduled runs (daily CI)
│       ├── benchmark_summary_20250115_143022.json
│       └── model-name/
│           └── model-name_benchmark_20250115_143022.json
└── 2025-01-16/
    └── ...
```

**Authentication for Uploads:**

For uploading results, you need a HuggingFace token with write permissions to the target dataset. You can provide the token in several ways (in order of precedence):

1. Command line: `--token hf_your_token_here`
3. Environment variable: `HF_TOKEN`

### Running Specific Benchmarks

```bash
# Include only specific benchmarks
python run_benchmarks.py --include llama

# Exclude specific benchmarks
python run_benchmarks.py --exclude old_benchmark
```

### Comparing Against a Baseline

`compare_results.py` compares a candidate run to a baseline, scenario by scenario, on end-to-end latency, time to first
token, inter-token latency and throughput. A metric that is worse by more than `--threshold` percent is a regression,
and the script then exits with status 1, so it can gate CI.

```bash
# Baseline on main, candidate on a branch, same hardware and arguments
python run_benchmarks.py --model-id meta-llama/Llama-2-7b-hf -b 1 -s 128 -n 128 -w 5 -i 30 --output-dir results_main
python run_benchmarks.py --model-id meta-llama/Llama-2-7b-hf -b 1 -s 128 -n 128 -w 5 -i 30 --output-dir results_pr

python compare_results.py --baseline results_main --candidate results_pr --threshold 1.0
```

Each side is a results file or a directory of them: with several repeated runs, each metric is the median of the
per-run medians. The report also shows the noise for each metric (the largest within-run or between-run coefficient of
variation) and flags it when it is above the threshold: the comparison cannot resolve a change that small. Pick a
threshold above the noise of your setup, and run the baseline and the candidate on the same machine, GPU and arguments.
Scenarios that failed or only exist on one side are listed separately.

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "model_name": "llama_2_7b",
  "benchmark_scenarios": [
    {
      "scenario_name": "eager_variant",
      "metadata": {
        "timestamp": "2025-01-XX...",
        "commit_id": "abc123...",
        "hardware_info": {
          "gpu_name": "NVIDIA A100",
          "gpu_memory_total": 40960,
          "cpu_count": 64
        },
        "config": {
          "variant": "eager",
          "warmup_iterations": 3,
          "measurement_iterations": 5
        }
      },
      "measurements": {
        "latency": {
          "mean": 2.45,
          "median": 2.43,
          "std": 0.12,
          "min": 2.31,
          "max": 2.67,
          "p95": 2.61,
          "p99": 2.65
        },
        "time_to_first_token": {
          "mean": 0.15,
          "std": 0.02
        },
        "tokens_per_second": {
          "mean": 87.3,
          "unit": "tokens/sec"
        }
      },
      "gpu_metrics": {
        "gpu_utilization_mean": 85.2,
        "gpu_memory_used_mean": 12450
      }
    }
  ]
}
```

### Debug Mode

```bash
python run_benchmarks.py --log-level DEBUG
```

## Contributing

To add new benchmarks:

1. Create a new file in `benches/`
2. Implement the `ModelBenchmark` interface
3. Add a runner function (`run_<benchmark_name>` or `run_benchmark`)
4. run_benchmarks.py