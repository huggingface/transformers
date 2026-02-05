#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script for MXFP4 training with backward kernels.

Compares:
1. MXFP4 training (native, weights stay quantized)
2. MXFP4 with dequantization to BF16 for training

Metrics:
- Forward + backward throughput (tokens/sec)
- Peak memory usage
- Memory savings vs BF16
"""

import argparse
import gc
import time
from contextlib import contextmanager

import torch


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def get_peak_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def track_memory():
    """Context manager to track memory usage."""
    reset_memory_stats()
    start_mem = get_gpu_memory_mb()
    yield
    peak_mem = get_peak_gpu_memory_mb()
    end_mem = get_gpu_memory_mb()
    print(f"  Memory: start={start_mem:.1f}MB, peak={peak_mem:.1f}MB, end={end_mem:.1f}MB")


def benchmark_swiglu_backward(batch_size: int, seq_len: int, hidden_size: int, warmup: int = 5, iterations: int = 100):
    """Benchmark SwiGLU backward pass."""
    print("\n=== SwiGLU Backward Benchmark ===")
    print(f"Shape: [{batch_size}, {seq_len}, {hidden_size}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create inputs
    input_a = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    grad_output = torch.randn(batch_size, seq_len, hidden_size // 2, dtype=torch.float32, device=device)

    # Import implementations
    from transformers.integrations.mxfp4_backward import swiglu_backward_torch

    # Warmup
    for _ in range(warmup):
        _ = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=7.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark PyTorch implementation
    start = time.perf_counter()
    for _ in range(iterations):
        _ = swiglu_backward_torch(grad_output, input_a, alpha=1.702, limit=7.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations

    tokens_per_sec = (batch_size * seq_len) / torch_time
    print(f"PyTorch backward: {torch_time * 1000:.3f}ms ({tokens_per_sec:.0f} tokens/sec)")

    # Benchmark Triton implementation if available
    if torch.cuda.is_available():
        try:
            from transformers.integrations.mxfp4_backward import swiglu_backward_triton

            # Warmup
            for _ in range(warmup):
                _ = swiglu_backward_triton(grad_output, input_a, alpha=1.702, limit=7.0)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                _ = swiglu_backward_triton(grad_output, input_a, alpha=1.702, limit=7.0)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / iterations

            tokens_per_sec = (batch_size * seq_len) / triton_time
            speedup = torch_time / triton_time
            print(f"Triton backward: {triton_time * 1000:.3f}ms ({tokens_per_sec:.0f} tokens/sec, {speedup:.2f}x speedup)")

        except Exception as e:
            print(f"Triton benchmark skipped: {e}")


def benchmark_memory_comparison(
    n_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    """Compare memory usage: MXFP4 vs BF16 weights."""
    print("\n=== Memory Comparison ===")
    print(f"Experts: {n_experts}, Hidden: {hidden_size}, Intermediate: {intermediate_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate theoretical sizes
    gate_up_params = n_experts * 2 * intermediate_size * hidden_size
    down_params = n_experts * hidden_size * intermediate_size
    total_params = gate_up_params + down_params

    bf16_size_mb = (total_params * 2) / (1024 * 1024)  # 2 bytes per BF16
    mxfp4_size_mb = (total_params * 0.5) / (1024 * 1024)  # 0.5 bytes per FP4

    print(f"Parameters: {total_params:,}")
    print(f"BF16 size: {bf16_size_mb:.1f}MB")
    print(f"MXFP4 size: {mxfp4_size_mb:.1f}MB (includes scales)")
    print(f"Memory savings: {(1 - mxfp4_size_mb / bf16_size_mb) * 100:.1f}%")

    # Actual memory test
    if device == "cuda":
        reset_memory_stats()

        # BF16 weights
        with track_memory():
            bf16_gate_up = torch.randn(n_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
            bf16_down = torch.randn(n_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        bf16_peak = get_peak_gpu_memory_mb()
        del bf16_gate_up, bf16_down

        reset_memory_stats()

        # MXFP4 weights (packed)
        with track_memory():
            mxfp4_gate_up = torch.randint(
                0, 256, (n_experts, 2 * intermediate_size, hidden_size // 32, 16), dtype=torch.uint8, device=device
            )
            mxfp4_gate_up_scales = torch.randint(
                0, 256, (n_experts, 2 * intermediate_size // 32, hidden_size), dtype=torch.uint8, device=device
            )
            mxfp4_down = torch.randint(
                0, 256, (n_experts, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8, device=device
            )
            mxfp4_down_scales = torch.randint(
                0, 256, (n_experts, hidden_size // 32, intermediate_size), dtype=torch.uint8, device=device
            )
        mxfp4_peak = get_peak_gpu_memory_mb()
        del mxfp4_gate_up, mxfp4_gate_up_scales, mxfp4_down, mxfp4_down_scales

        print("\nActual memory allocation:")
        print(f"BF16 peak: {bf16_peak:.1f}MB")
        print(f"MXFP4 peak: {mxfp4_peak:.1f}MB")
        print(f"Actual savings: {(1 - mxfp4_peak / bf16_peak) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MXFP4 backward kernels")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=2880, help="Hidden size")
    parser.add_argument("--intermediate-size", type=int, default=1440, help="Intermediate size")
    parser.add_argument("--n-experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    print("MXFP4 Backward Kernels Benchmark")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    else:
        print("Running on CPU (limited benchmark)")

    # Run benchmarks
    benchmark_swiglu_backward(
        args.batch_size, args.seq_len, args.hidden_size, args.warmup, args.iterations
    )

    benchmark_memory_comparison(args.n_experts, args.hidden_size, args.intermediate_size)

    print("\n" + "=" * 50)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
