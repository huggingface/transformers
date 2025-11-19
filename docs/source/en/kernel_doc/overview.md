# Hugging Face Transformers Kernels

The Transformers Kernels integration provides high-performance, optimized kernel implementations for common transformer operations. By leveraging specialized CUDA, Triton, ROCm, and XPU kernels from the Hugging Face Hub, you can significantly accelerate model inference and training with minimal code changes.

## Key Benefits

- **Drop-in Performance Boost**: Enable optimized kernels with a single `use_kernels=True` flag
- **Hub-Based Distribution**: Access community-maintained kernels directly from Hugging Face Hub
- **Multi-Backend Support**: CUDA, ROCm, and XPU backends for different hardware platforms
- **Mode-Aware Optimization**: Automatically switches between training and inference optimizations
- **Zero Code Changes**: Existing models automatically benefit from kernel acceleration
- **Customizable**: Override default kernels with your own implementations via `KernelConfig`

## Supported Operations

The `kernels` library provides optimized implementations for:

### Normalization Layers
- **RMSNorm**: Root Mean Square Layer Normalization

### Activation Functions
- **FastGELU, NewGELU, QuickGELU**: GELU variants
- **SiLU**: Sigmoid Linear Unit
- **GeluTanh**: GELU with Tanh approximation

### MLP and MoE Layers
- **MLP**: Standard Multi-Layer Perceptron layers
- **MegaBlocksMoeMLP**: Optimized Mixture-of-Experts implementations
- **Llama4TextMoe**: Llama 4 MoE layers

### Attention Mechanisms
- **Flash Attention**: Fast attention implementations
- **MultiScaleDeformableAttention**: For vision transformers
- **Custom Attention**: Load community kernels for specialized attention patterns

### Specialized Operations
- **Mamba Selective Scan**: Built-in optimized kernels for Mamba SSM models
- **Causal Convolution 1D**: Efficient causal convolutions

## Quick Start

### Basic Usage

Enable kernels when loading any model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    use_kernels=True,
    device_map="cuda"
)

# Model now uses optimized kernels automatically
output = model.generate(input_ids, max_new_tokens=50)
```

### Custom Attention Kernels

Use specialized attention implementations:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    attn_implementation="kernels-community/flash-attn",
    device_map="cuda"
)
```

## How It Works

### Automatic Kernel Replacement

When `use_kernels=True`, the library:

1. **Scans the model**: Identifies layers that have optimized kernel implementations
2. **Loads kernels from Hub**: Downloads and caches kernels from Hugging Face repositories
3. **Replaces forward methods**: Swaps standard PyTorch operations with optimized kernels
4. **Maintains compatibility**: Ensures identical outputs while improving performance

### Mode-Aware Optimization

Kernels automatically adapt to your training/inference workflow:

```python
model.eval()   # Uses inference-optimized kernels
model.train()  # Switches to training-optimized kernels
```

### Kernel Sources

Kernels are distributed through Hugging Face Hub repositories in the format:
- `org/repo:layer_name` - Latest version
- `org/repo@v0.1.0:layer_name` - Specific version
- Supports semantic versioning constraints

**Popular Kernel Repositories:**
- [`kernels-community/flash-attn`](https://huggingface.co/kernels-community/flash-attn2) - Flash attention implementations
- [`kernels-community/megablocks`](https://huggingface.co/kernels-community/megablocks) - MoE optimizations
- [`kernels-community/moe`](https://huggingface.co/RedHatAI/moe) - Llama 4 MoE layers
- [`kernels-community/liger_kernels`](https://huggingface.co/kernels-community/liger_kernels) - RMSNorm, activation functions

## Requirements

- **kernels** package: `pip install kernels`
- **Recommended**: `kernels>=0.10.2` for XPU support

## Advanced Features

### Training and Inference Modes

Kernels support different operational modes:

- **Mode.INFERENCE**: Optimized for inference workloads (batch size optimization, reduced memory)
- **Mode.TRAINING**: Optimized for training (gradient computation, mixed precision)
- **Mode.TORCH_COMPILE**: Compatible with `torch.compile` for JIT optimization

### Device-Specific Kernels

Specify different kernel implementations per device:

```python
kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": {
            "cuda": "kernels-community/cuda-norm:FastRMSNorm",
            "rocm": "kernels-community/rocm-norm:RocmRMSNorm",
            "xpu": "kernels-community/xpu-norm:XpuRMSNorm"
        }
    }
)
```

### Built-in Kernels

Transformers includes built-in CUDA kernels for specific models:

- **Falcon Mamba**: Selective scan operations with layer normalization fusion
- Located in: `transformers.kernels.falcon_mamba`

## Important Notes

- **No Unkernelization**: Once kernels are enabled, they cannot be disabled during the session
- **Lazy Loading**: Kernels are downloaded and cached only when needed
- **Backward Compatibility**: Models work identically with or without kernels enabled
- **Hardware Requirements**: CUDA kernels require compatible NVIDIA GPUs; ROCm requires AMD GPUs; XPU requires Intel GPUs

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'kernels'`:

```bash
pip install kernels
```

### Disabling the use of `kernels` globally

You can disable the use of `kernels` globally by setting the environment variable `USE_HUB_KERNELS=0|OFF|NO`.

### Device Compatibility

Not all kernels support all devices. Check the kernel repository documentation for device support.

## Additional Resources

- **Kernels Library**: [github.com/huggingface/kernels](https://github.com/huggingface/kernels)
- **Community Kernels**: [huggingface.co/kernels-community](https://huggingface.co/kernels-community)
- **API Reference**: See `KernelConfig` documentation for advanced configuration options
