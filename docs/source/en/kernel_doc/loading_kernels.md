<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Loading kernels

A kernel works as a drop-in replacement for standard PyTorch operations. It swaps the `forward` method with the optimized kernel implementation without breaking model code.

This guide shows how to load kernels to accelerate inference.

Install the kernels package. We recommend the latest version which provides the best performance and bug fixes.

> [!NOTE]
> kernels >=0.11.0 is the minimum required version for working with Transformers.

```bash
pip install -U kernels
```

Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to load a matching kernel variant for your platform and environment. This replaces supported PyTorch operations with the kernel implementation.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    device_map="cuda"
)
```

## Attention kernels

Load attention kernels from the Hub with the `attn_implementation` argument.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="kernels-community/flash-attn2",
    device_map="cuda"
)
```

Specific kernels, like attention, accept several formats.

- `@v2.1.0` pins to a specific tag or branch.
- `@>=2.0,<3.0` sets semantic versioning constraints.

```py
from transformers import AutoModelForCausalLM

# pin to a specific version
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="kernels-community/flash-attn2@v2.1.0",
    device_map="cuda"
)
# use semantic versioning constraints
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="kernels-community/flash-attn2@>=2.0,<3.0",
    device_map="cuda"
)
```

## Mode-awareness

Kernels automatically adapt to [training](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) and [inference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) modes based on PyTorch's `model.training` state.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
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

Explicitly enable training and inference modes with the `mode` argument in the [`~kernels.kernelize`] function. Training mode also supports an additional torch.compile mode.

```py
from kernels import Mode

# inference optimized kernels
model.kernelize(mode=Mode.INFERENCE)

# training optimized kernels
model.kernelize(mode=Mode.TRAINING)

# training and torch-compile friendly kernels
model.kernelize(mode=Mode.TRAINING | Mode.TORCH_COMPILE)
```

## KernelConfig

[`KernelConfig`] customizes which kernels are used in a model.

The `:` separator names a specific kernel entry inside the repository and maps it to a layer.

```py
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": "kernels-community/liger_kernels:LigerRMSNorm",
        "LlamaAttention": "kernels-community/flash-attn2:FlashAttention2",
    }
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda"
)
```

Specify different kernel implementations for each device type.

```py
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

## Local kernels

Load kernels from local file paths with `use_local_kernel=True` in [`KernelConfig`]. This loads from a local filesystem path instead of a Hub repository.

Local kernels use `/abs/path:layer_name` instead of the Hub format `org/repo:layer_name`.

```py
from transformers import KernelConfig, AutoModelForCausalLM

kernel_mapping = {
    "RMSNorm": "/path/to/liger_kernels:LigerRMSNorm",
}
kernel_config = KernelConfig(kernel_mapping, use_local_kernel=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype="auto",
    device_map="auto",
    use_kernels=True,
    kernel_config=kernel_config
)
```

## Disabling kernels

Disable kernels for specific layers with an empty kernel mapping in [`KernelConfig`].

```py
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": "",
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda"
)
```

Set the environment variable to disable kernels globally.

```bash
export USE_HUB_KERNELS=0  # or OFF or NO
```

## Troubleshooting

Kernel integration depends on hardware, drivers, and package versions working together. The following sections cover common failures.

### Installation issues

Import errors indicate the kernels library isn't installed.

```bash
pip install -U kernels
```

### Kernel loading failures

If specific kernels fail to load, try the following.

- Check your hardware compatibility with the kernel requirements.
- Verify your CUDA/ROCm/Metal drivers are up to date.
- Consult the kernel repository documentation for known issues.

### Device compatibility

Not all kernels support all devices. The library falls back to standard PyTorch operations if a kernel is unavailable for your hardware. Check kernel repository documentation for device-specific support.

## Resources

- [Kernels](https://github.com/huggingface/kernels) repository
- [Enhance Your Models in 5 Minutes with the Hugging Face Kernel Hub](https://huggingface.co/blog/hello-hf-kernels) blog post
- Discover kernels in the [kernels-community](https://huggingface.co/kernels-community) org
