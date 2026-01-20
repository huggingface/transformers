# Kernels

The Transformers Kernels integration provides high-performance, optimized kernel implementations for common transformer operations. By enabling kernels with a single `use_kernels=True` flag, you can achieve significant speedups for model inference and training with minimal code changes. The system leverages specialized CUDA, Triton, ROCm, Metal, and XPU kernels distributed through the Hugging Face Hub, automatically replacing standard PyTorch operations while maintaining full compatibility. Kernels are mode-aware (automatically switching between training and inference optimizations), support multiple hardware backends (NVIDIA, AMD, Apple Silicon, Intel), and are fully customizable via `KernelConfig` for advanced use cases.

For more information on optimizing transformer performance, see the [Performance and Optimization guide](../performance).

## Requirements

### Software Dependencies

Install the `kernels` package to enable this feature:

```bash
pip install kernels>=0.9.0
```

Upgrade to the latest version for the newest features and hardware support:

```bash
pip install --upgrade kernels
```

Please note the 0.9.0 is the minimum version required for this feature.
We do recommend using the most recent version to get the best performances and bug fixes.

### Hardware Compatibility

Kernels support multiple hardware platforms with varying levels of optimization:

- **NVIDIA GPUs (CUDA)**: Modern architectures with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell)
- **AMD GPUs (ROCm)**: Compatible with ROCm-supported devices
- **Apple Silicon (Metal)**: M-series chips (M1, M2, M3, M4 and newer)
- **Intel GPUs (XPU)**: Intel Data Center GPU Max Series and compatible devices

Individual kernel implementations may have specific requirements. Consult the kernel repository documentation for detailed compatibility information.

## Quick Start

### Basic Usage

Let `kernels` pull and replace supported operations with optimized kernels when loading any model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    device_map="cuda"
)

# Model now uses optimized kernels automatically
output = model.generate(input_ids, max_new_tokens=50)
```

### Custom Kernel Repositories

Specify kernels from different Hub repositories using various formats:

```python
from transformers import AutoModelForCausalLM

# Use latest version from a repository
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="kernels-community/flash-attn2",
    device_map="cuda"
)

# Pin to a specific version
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="kernels-community/flash-attn2@v2.1.0",
    device_map="cuda"
)

# Use semantic versioning constraints
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="kernels-community/flash-attn2@>=2.0,<3.0",
    device_map="cuda"
)
```

### Mode-Aware Kernels

Kernels automatically adapt to training and inference modes:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    use_kernels=True,
    device_map="cuda"
)

# Switch to inference mode - uses inference-optimized kernels
model.eval()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=50)

# Switch to training mode - uses training-optimized kernels with gradient support
model.train()
loss = model(input_ids, labels=labels).loss
loss.backward()
```

## How It Works

### Automatic Kernel Replacement

When `use_kernels=True`, the library automatically optimizes your model:

1. **Scans the model**: Identifies layers that have optimized kernel implementations available
2. **Loads kernels from Hub**: Downloads and caches kernels from Hugging Face repositories
3. **Replaces forward methods**: Swaps standard PyTorch operations with optimized kernels

The process is transparent and requires no code changes to your model implementation.

### Kernel Sources

Kernels are distributed through Hugging Face Hub repositories with flexible versioning:

- `org/repo:layer_name` - Latest version
- `org/repo@v1.2.3:layer_name` - Specific version
- `org/repo@>=1.0,<2.0:layer_name` - Semantic versioning constraints

**Some popular Kernel Repositories:**
- [`kernels-community/flash-attn2`](https://huggingface.co/kernels-community/flash-attn2) - Flash attention implementations
- [`kernels-community/flash-attn2`](https://huggingface.co/kernels-community/vllm-flash-attn3) - Flash Attention 3 with support for attention sinks
- [`kernels-community/megablocks`](https://huggingface.co/kernels-community/megablocks) - MoE optimizations
- [`kernels-community/moe`](https://huggingface.co/RedHatAI/moe) - Llama 4 MoE layers
- [`kernels-community/liger_kernels`](https://huggingface.co/kernels-community/liger_kernels) - RMSNorm, activation functions

Browse available kernels at [huggingface.co/kernels-community](https://huggingface.co/kernels-community).

## Advanced Configuration

### Custom Kernel Mappings

Use `KernelConfig` to specify different kernel implementations:

```python
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": "kernels-community/liger_kernels:LigerRMSNorm",
        "LlamaAttention": "kernels-community/flash-attn2:FlashAttention2",
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    kernel_config=kernel_config,
    device_map="cuda"
)
```

### Device-Specific Kernels

Specify different kernel implementations per device type:

```python
from transformers import KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": {
            "cuda": "kernels-community/liger_kernels:LigerRMSNorm",
            "rocm": "kernels-community/rocm-kernels:RocmRMSNorm",
            "metal": "kernels-community/metal-kernels:MetalRMSNorm",
            "xpu": "kernels-community/xpu-kernels:XpuRMSNorm"
        }
    }
)
```

### Disabling Kernels

You can disable kernels for specific layers using empty kernel mappings:

```python
from transformers import AutoModelForCausalLM, KernelConfig

# Disable kernels for specific layers
kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": "",  # Empty string disables kernel for this layer
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda"
)
```

To globally disable kernels, set the environment variable:

```bash
export USE_HUB_KERNELS=0  # or OFF or NO
```

### Operational Modes

Kernels support different operational modes for various workflows:

- **Mode.INFERENCE**: Optimized for inference workloads (batch processing, reduced memory)
- **Mode.TRAINING**: Optimized for training (gradient computation, mixed precision training)
- **Mode.TORCH_COMPILE**: Compatible with `torch.compile()` for JIT optimization

Modes are automatically selected based on `model.training` state, but can be explicitly configured via `KernelConfig`.

## Important Notes

### Determinism

Some kernel implementations may produce slightly different numerical results than standard PyTorch operations due to optimizations like operation reordering or different accumulation strategies. While functionally equivalent, this may affect reproducibility in some scenarios.

For deterministic behavior when required:
- Check kernel repository documentation for determinism guarantees
- Consider disabling specific kernels that affect your use case
- Set appropriate random seeds and PyTorch deterministic flags

### Compatibility

- **Lazy Loading**: Kernels are downloaded and cached only when needed, reducing startup time
- **Backward Compatibility**: Models work identically with or without kernels enabled
- **Dynamic Replacement**: Kernel replacement happens at model load time and persists for the model's lifetime

## Troubleshooting

### Installation Issues

If you encounter import errors:

```bash
pip install kernels
```

### Kernel Loading Failures

If specific kernels fail to load:
- Check your hardware compatibility with the kernel requirements
- Verify your CUDA/ROCm/Metal drivers are up to date
- Consult the kernel repository documentation for known issues

### Device Compatibility

Not all kernels support all devices. The library will fall back to standard PyTorch operations if a kernel is unavailable for your hardware. Check kernel repository documentation for device-specific support.

## Additional Resources

- **Kernels Library**: [github.com/huggingface/kernels](https://github.com/huggingface/kernels) - Core kernels implementation and kernel builder tools
- **Community Kernels**: [huggingface.co/kernels-community](https://huggingface.co/kernels-community) - Browse available kernel implementations
- **Performance Guide**: See the [Performance and Optimization documentation](../performance) for comprehensive optimization strategies
- **API Reference**: Detailed `KernelConfig` documentation for advanced configuration options