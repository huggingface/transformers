# Benchmarking v2

A comprehensive benchmarking framework for transformer models that supports multiple execution modes (eager, compiled, kernelized), detailed performance metrics collection, and structured output format.


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support, ensure you have CUDA-compatible PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

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

# Enable mock benchmark (skipped by default)
python run_benchmarks.py --include mock_benchmark --enable-mock
```

## Execution Modes

### Eager Mode
Standard PyTorch execution without compilation:
```python
config.variant = "eager"
```

### Compiled Mode
Uses `torch.compile` for optimization:
```python
config.variant = "compiled"
config.compile_mode = "max-autotune"  # or "default", "reduce-overhead"
```

### Kernelized Mode
Optimized kernel execution (experimental):
```python
config.variant = "kernelized"
config.compile_mode = "max-autotune"
```

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

## Mock Benchmark

The framework includes a mock benchmark for testing and demonstration purposes:

```bash
# Run mock benchmark (demonstrates framework without ML dependencies)
python run_benchmarks.py --include mock_benchmark --enable-mock

# Test framework with fast settings
python run_benchmarks.py --include mock_benchmark --enable-mock \
    --warmup-iterations 1 --measurement-iterations 2 --num-tokens-to-generate 10
```

The mock benchmark:
- **No Dependencies**: Runs without torch/transformers
- **Realistic Simulation**: Generates realistic timing data for different variants
- **Fast Testing**: Quick way to test the framework
- **Performance Simulation**: Shows expected performance differences between eager/compiled/kernelized
- **Skipped by Default**: Won't run unless explicitly enabled with `--enable-mock`

## Advanced Usage

### Custom Hardware Monitoring

```python
from framework import GPUMonitor

monitor = GPUMonitor(sample_interval=0.05)  # 50ms sampling
monitor.start()
# ... run benchmark ...
metrics = monitor.stop()
```

### Custom Timing Measurements

```python
from framework import TimingResult, flush_memory
import time

def custom_measurement():
    flush_memory()  # Clear GPU cache
    
    start_time = time.perf_counter()
    # ... your code ...
    end_time = time.perf_counter()
    
    return TimingResult(
        latency=end_time - start_time,
        tokens_per_second=tokens_generated / (end_time - start_time)
    )
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `num_tokens_to_generate`
2. **Import errors**: Ensure all dependencies are installed
3. **No benchmarks found**: Check that benchmark files have proper runner functions
4. **Compilation failures**: Some models don't support all compile modes

### Debug Mode

```bash
python run_benchmarks.py --log-level DEBUG
```

## Contributing

To add new benchmarks:

1. Create a new file in `benches/`
2. Implement the `ModelBenchmark` interface
3. Add a runner function (`run_<benchmark_name>` or `run_benchmark`)
4. Test with the framework

The framework will automatically discover and run your benchmark! 