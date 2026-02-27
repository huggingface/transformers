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



## Liger

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of layers such as RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy, and more that have been fused into a single Triton kernel for training LLMs. These kernels are also compatible with FlashAttention, FSDP, and DeepSpeed. As a result, Liger Kernel can increase multi-GPU training throughput and reduce memory usage. This is useful for multi-head training and supporting larger vocabulary sizes, larger batch sizes, and longer context lengths.

```bash
pip install liger-kernel
```

Enable Liger Kernel for training by setting `use_liger_kernel=True` in [`TrainingArguments`]. This patches the corresponding layers in the model with Ligers kernels.

> [!TIP]
> Liger Kernel supports Llama, Gemma, Mistral, and Mixtral models. Refer to the [patching](https://github.com/linkedin/Liger-Kernel#patching) list for the latest list of supported models.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...,
    use_liger_kernel=True
)
```

You can also configure which specific kernels to apply using the `liger_kernel_config` parameter. This dict is passed as keyword arguments to the `_apply_liger_kernel_to_instance` function, allowing fine-grained control over kernel usage. Available options vary by model but typically include: `rope`, `swiglu`, `cross_entropy`, `fused_linear_cross_entropy`, `rms_norm`, etc.

```py
from transformers import TrainingArguments

# Apply only specific kernels
training_args = TrainingArguments(
    ...,
    use_liger_kernel=True,
    liger_kernel_config={
        "rope": True,
        "cross_entropy": True,
        "rms_norm": False,  # Don't apply Liger's RMSNorm kernel
        "swiglu": True,
    }
)
```