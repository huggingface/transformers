

# Apple Silicon

Apple Silicon (M-series) chips have a unified memory architecture where the CPU and GPU share the same memory pool. Shared memory eliminates the data transfer overhead of GPUs, making it practical to train large models locally. Transformers uses the [Metal Performance Shaders (MPS)](https://pytorch.org/docs/stable/notes/mps.html) backend to accelerate training on this hardware.

This requires macOS 12.3 or later and PyTorch built with MPS support.

> [!WARNING]
> MPS doesn't support all PyTorch operations yet (see this [GitHub issue](https://github.com/pytorch/pytorch/issues/77764) for more details about missing ops). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU kernels for unsupported operations. Open an issue in the [PyTorch](https://github.com/pytorch/pytorch/issues) repository for any other unexpected behavior.



## Model loading and device selection

MPS requires the entire model to fit in unified memory, so `device_map="auto"` can't offload layers to the CPU like CUDA. In this case, try using a smaller model.

Loading weights to MPS is faster and uses less memory with safetensors `0.8.0` and PyTorch 2.9 or later. When `device_map="mps"` or `"auto"`, weights are mapped into Metal buffers without an intermediate copy, which roughly halves the memory footprint during loading and makes it about 5-6x faster. On older PyTorch versions, loading will fall back to the standard load path.

[`Trainer`] detects MPS automatically with `torch.backends.mps.is_available` and sets the device to `mps` without any configuration changes.

## Mixed precision

MPS supports both bf16 and fp16 mixed precision (bf16 requires macOS 14.0 or later).

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,  # requires macOS 14.0+
)
```



## Graph cache

MPS compiles a separate Metal kernel for each unique tensor shape and stores them in a graph cache with no eviction policy. Training with variable-length inputs (padded sequences, dynamic batches) grows the cache on every new shape and can eventually exhaust unified memory.

Set `torch_empty_cache_steps` in [`TrainingArguments`] to bound this growth. On MPS, [`Trainer`] clears the graph cache alongside the device cache every `torch_empty_cache_steps` steps, at a throughput cost. You need to opt-in to clear both caches. When `torch_empty_cache_steps` is unset (the default), neither cache is cleared and behavior is unchanged.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    torch_empty_cache_steps=1,  # clear caches every step
)
```

Graph cache clearing requires PyTorch 2.13 or later. On older versions, [`Trainer`] skips the graph cache call and only clears the device cache.

## Next steps

- Read the [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) blog post for background on the MPS backend.

