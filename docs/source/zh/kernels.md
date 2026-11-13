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


# Kernels（自定义内核）

自定义内核针对矩阵乘法、注意力计算和归一化等特定算子进行优化，使其运行更快。将多个算子融合到单个内核中可以减少对 GPU 显存的读写次数，降低内存带宽使用，同时消除逐算子的启动开销。

## Hub 内核

[Hub](https://huggingface.co/kernels-community) 上托管了社区内核，你可以通过 [`KernelConfig`] 加载它们。将配置传入 [`~AutoModelForCausalLM.from_pretrained`] 的 `kernel_config` 参数即可。内核加载后，会在训练过程中自动激活。有关所有可用选项，请参阅[加载内核](./kernel_doc/loading_kernels#kernelconfig)指南。

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

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 将 RMSNorm、RoPE、SwiGLU、CrossEntropy 和 FusedLinearCrossEntropy 等层融合为单个 Triton 内核。它与 FlashAttention、FSDP 和 DeepSpeed 兼容，能够提升多 GPU 训练的吞吐量，同时降低显存占用，让更大的词汇量、批次大小和上下文长度变得更加可行。

```bash
pip install liger-kernel
```

在 [`TrainingArguments`] 中设置 `use_liger_kernel=True`，即可用 Liger 内核替换对应的模型层。

> [!TIP]
> 请参阅 [patching](https://github.com/linkedin/Liger-Kernel#patching) 页面获取支持的模型完整列表。

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...,
    use_liger_kernel=True
)
```

要控制哪些层被替换，可以通过 `liger_kernel_config` 字典来指定。可选参数因模型而异，包括：`rope`、`swiglu`、`cross_entropy`、`fused_linear_cross_entropy`、`rms_norm` 等。

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

## 下一步

- 参阅[注意力后端](./attention_interface)指南，了解 FlashAttention 等降低显存占用的内核详情。
- 参阅 [torch.compile](./torch_compile) 指南，了解如何编译整个训练步骤的前向和反向传播。
