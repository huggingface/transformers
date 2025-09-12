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

### Running Specific Benchmarks

```bash
# Include only specific benchmarks
python run_benchmarks.py --include llama

# Exclude specific benchmarks
python run_benchmarks.py --exclude old_benchmark

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