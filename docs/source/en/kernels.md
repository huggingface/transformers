<!---Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Kernels

Custom kernels target specific ops like matrix multiplications, attention, and normalization to run them faster. Fusing multiple ops into a single kernel reduces memory bandwidth usage by reading and writing GPU memory fewer times, and cuts per-op launch overhead.

## Hub kernels

The [Hub](https://huggingface.co/kernels-community) hosts community kernels you can load with [`KernelConfig`]. Pass the config to `kernel_config` in [`~AutoModelForCausalLM.from_pretrained`]. Once the kernel is loaded, it's active for training. Read the [Loading kernels](./kernel_doc/loading_kernels#kernelconfig) guide for all available options.

```py
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "RMSNorm": "kernels-community/rmsnorm",
    }
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
)
```

## Liger

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) fuses layers like RMSNorm, RoPE, SwiGLU, CrossEntropy, and FusedLinearCrossEntropy into single Triton kernels. It's compatible with FlashAttention, FSDP, and DeepSpeed, and improves multi-GPU training throughput while reducing memory usage, making larger vocabularies, batch sizes, and context lengths more feasible.

```bash
pip install liger-kernel
```

Set `use_liger_kernel=True` in [`TrainingArguments`] to patch the corresponding model layers with Liger's kernels.

> [!TIP]
> See the [patching](https://github.com/linkedin/Liger-Kernel#patching) page for a complete list of supported models.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...,
    use_liger_kernel=True
)
```

To control which layers are patched, pass `liger_kernel_config` as a dict. Available options vary by model and include: `rope`, `swiglu`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, etc.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...,
    use_liger_kernel=True,
    liger_kernel_config={
        "rope": True,
        "cross_entropy": True,
        "rms_norm": False,
        "swiglu": True,
    }
)
```

## Next steps

- See the [Attention backends](./attention_interface) guide for details on kernels like FlashAttention that reduce memory usage.
- See the [torch.compile](./torch_compile) guide to learn how to compile the forward and backward pass for your entire training step.
