#!/usr/bin/env python
"""
Microbenchmark for FA2 _get_unpad_data + _index_first_axis performance.

Tests the performance difference between passing max_seqlen_in_batch as:
- Python int (via .item())
- 0-dim CUDA tensor (no .item())

This isolates the kernel dispatch / sync overhead under large KV cache / 
bandwidth-saturated conditions.
"""
import argparse
import torch
import torch.nn.functional as F
import time
import sys

# Add transformers to path
sys.path.insert(0, "src")

from transformers.modeling_flash_attention_utils import (
    _get_unpad_data,
    _index_first_axis,
    _unpad_input,
)


def create_attention_mask(batch_size, seq_len, device, pad_ratio=0.0):
    """Create attention mask with optional padding."""
    if pad_ratio > 0:
        # Create variable length sequences
        lengths = torch.randint(
            int(seq_len * (1 - pad_ratio)), seq_len + 1, (batch_size,), device=device
        )
        mask = torch.arange(seq_len, device=device).expand(batch_size, -1) < lengths.unsqueeze(1)
    else:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return mask


def benchmark_unpad_data(attention_mask, num_iters=100, warmup=10):
    """Benchmark _get_unpad_data + downstream _index_first_axis."""
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    
    # Create dummy hidden states
    hidden_dim = 128
    num_heads = 32
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden_states = torch.randn(
        batch_size, seq_len, num_heads, hidden_dim, 
        device=device, dtype=dtype
    )
    
    # Warmup
    for _ in range(warmup):
        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
        _ = _index_first_axis(hidden_states, indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark with scalar max_seqlen (via .item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
        # Simulate downstream usage: pass max_seqlen to flash_attn_varlen_func
        # Here we just use it in a dummy op to prevent optimization
        _ = _index_first_axis(hidden_states, indices)
        _ = max_seqlen + 1  # Use the scalar
    if device.type == "cuda":
        torch.cuda.synchronize()
    scalar_time = time.perf_counter() - start
    
    return scalar_time / num_iters * 1000  # ms per iter


def benchmark_unpad_data_tensor_max_seqlen(attention_mask, num_iters=100, warmup=10):
    """Benchmark _get_unpad_data WITHOUT .item() - keeping max_seqlen as tensor."""
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    
    hidden_dim = 128
    num_heads = 32
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden_states = torch.randn(
        batch_size, seq_len, num_heads, hidden_dim, 
        device=device, dtype=dtype
    )
    
    # Monkey-patch to remove .item() call
    def patched_get_unpad_data(attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max()  # NO .item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch
    
    # Warmup
    for _ in range(warmup):
        indices, cu_seqlens, max_seqlen = patched_get_unpad_data(attention_mask)
        _ = _index_first_axis(hidden_states, indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark with tensor max_seqlen
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        indices, cu_seqlens, max_seqlen = patched_get_unpad_data(attention_mask)
        _ = _index_first_axis(hidden_states, indices)
        _ = max_seqlen + 1  # Use the tensor
    if device.type == "cuda":
        torch.cuda.synchronize()
    tensor_time = time.perf_counter() - start
    
    return tensor_time / num_iters * 1000  # ms per iter


def benchmark_with_profiler(attention_mask, num_iters=20):
    """Run with torch.profiler to analyze kernel dispatch differences."""
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    hidden_dim = 128
    num_heads = 32
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden_states = torch.randn(
        batch_size, seq_len, num_heads, hidden_dim, 
        device=device, dtype=dtype
    )
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    # Profile scalar path
    print("\n=== Profiling SCALAR max_seqlen path ===")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iters):
            indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
            _ = _index_first_axis(hidden_states, indices)
            _ = max_seqlen + 1
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))
    
    # Profile tensor path
    def patched_get_unpad_data(attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max()  # NO .item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch
    
    print("\n=== Profiling TENSOR max_seqlen path ===")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iters):
            indices, cu_seqlens, max_seqlen = patched_get_unpad_data(attention_mask)
            _ = _index_first_axis(hidden_states, indices)
            _ = max_seqlen + 1
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description="FA2 unpad microbenchmark")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--pad-ratio", type=float, default=0.0, help="Padding ratio (0-1)")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--profile", action="store_true", help="Run with profiler")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch}, Seq len: {args.seq_len}, Pad ratio: {args.pad_ratio}")
    
    attention_mask = create_attention_mask(args.batch, args.seq_len, device, args.pad_ratio)
    
    print(f"\nMask shape: {attention_mask.shape}, Non-zero: {attention_mask.sum().item()}")
    
    if args.profile:
        benchmark_with_profiler(attention_mask, num_iters=min(args.iters, 20))
    else:
        scalar_ms = benchmark_unpad_data(attention_mask, args.iters, args.warmup)
        tensor_ms = benchmark_unpad_data_tensor_max_seqlen(attention_mask, args.iters, args.warmup)
        
        print(f"\nResults (avg over {args.iters} iters):")
        print(f"  Scalar max_seqlen (.item()):  {scalar_ms:.3f} ms/iter")
        print(f"  Tensor max_seqlen (no .item()): {tensor_ms:.3f} ms/iter")
        print(f"  Slowdown: {tensor_ms / scalar_ms:.2f}x ({(tensor_ms - scalar_ms) / scalar_ms * 100:.1f}%)")


if __name__ == "__main__":
    main()