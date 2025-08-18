# Benchmarking Framework v2

A comprehensive benchmarking framework for transformer models that supports multiple execution modes (eager, compiled, kernelized), detailed performance metrics collection, and structured output format.

## Features

- **Multiple Execution Modes**: Support for eager, compiled (torch.compile), and kernelized execution
- **Comprehensive Metrics**: Time-to-first-token, latency, throughput (tokens/sec), GPU utilization
- **Statistical Analysis**: Mean, median, percentiles (25th, 75th, 90th, 95th, 99th), min/max values
- **Hardware Info Collection**: GPU type, memory, CPU info, software versions
- **Metadata Tracking**: Timestamps, commit IDs, configuration details
- **Structured Output**: JSON format with organized benchmark scenarios
- **Configurable Parameters**: Warmup iterations, measurement iterations, token counts
- **Auto-discovery**: Automatically finds and runs all benchmarks in `./benches` directory

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
python run_benchmarks.py --include llama_benchmark

# Exclude specific benchmarks
python run_benchmarks.py --exclude old_benchmark

# Run with specific model
python run_benchmarks.py --model-id gpt2

# Enable mock benchmark (skipped by default)
python run_benchmarks.py --include mock_benchmark --enable-mock
```

### Running Individual Benchmarks

```bash
# Run LLaMA benchmark directly
cd benches
python llama_benchmark.py
```

## Framework Architecture

### Core Components

1. **`framework.py`**: Core framework with base classes and utilities
2. **`run_benchmarks.py`**: Top-level script for running all benchmarks
3. **`benches/`**: Directory containing benchmark implementations

### Key Classes

- **`BenchmarkConfig`**: Configuration for benchmark scenarios
- **`ModelBenchmark`**: Abstract base class for model-specific benchmarks
- **`BenchmarkRunner`**: Coordinates benchmark execution and metrics collection
- **`GPUMonitor`**: Collects GPU utilization metrics during benchmarking

## Creating New Benchmarks

### 1. Create Benchmark Implementation

Create a new file in `benches/your_model_benchmark.py`:

```python
from framework import ModelBenchmark, BenchmarkConfig, TimingResult
import logging

class YourModelBenchmark(ModelBenchmark):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        # Initialize your benchmark
        
    def setup_model(self, config: BenchmarkConfig) -> None:
        # Setup model based on configuration
        # Handle eager/compiled/kernelized variants
        pass
        
    def cleanup_model(self) -> None:
        # Cleanup resources
        pass
        
    def measure_time_to_first_token(self, config: BenchmarkConfig) -> float:
        # Measure time to first token
        return 0.0
        
    def measure_latency(self, config: BenchmarkConfig) -> TimingResult:
        # Measure full generation latency
        return TimingResult(latency=0.0, tokens_per_second=0.0)

def run_your_model_benchmark(logger, output_dir="benchmark_results", **kwargs):
    """Entry point function for your benchmark."""
    from framework import BenchmarkRunner, create_config_variants
    
    benchmark = YourModelBenchmark(logger)
    runner = BenchmarkRunner(logger, output_dir)
    
    # Create configurations
    configs = create_config_variants(base_config, variants)
    
    # Run and save results
    results = runner.run_benchmark(benchmark, configs)
    return runner.save_results("your_model", results)
```

### 2. Configuration Variants

Use `create_config_variants` to test multiple configurations:

```python
from framework import BenchmarkConfig, create_config_variants

base_config = BenchmarkConfig(
    name="my_benchmark",
    model_id="my-model",
    warmup_iterations=3,
    measurement_iterations=5
)

configs = create_config_variants(base_config, {
    "variant": ["eager", "compiled", "kernelized"],
    "compile_mode": [None, "default", "max-autotune"],
    "batch_size": [1, 2, 4]
})
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
```

## Command Line Options

```bash
python run_benchmarks.py --help
```

Key options:
- `--output-dir`: Base output directory
- `--model-id`: Specific model to benchmark
- `--warmup-iterations`: Number of warmup runs
- `--measurement-iterations`: Number of measurement runs
- `--num-tokens-to-generate`: Tokens to generate per run
- `--include`/`--exclude`: Filter benchmarks to run
- `--enable-mock`: Enable mock benchmark (skipped by default)
- `--log-level`: Logging verbosity

## Example Benchmark Results

After running benchmarks, you'll find:

```
benchmark_results/
├── llama_2_7b/
│   └── llama_2_7b_benchmark_20250101_120000.json
├── gpt2/
│   └── gpt2_benchmark_20250101_120500.json
├── benchmark_summary.json
└── benchmark_run_20250101_120000.log
```

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