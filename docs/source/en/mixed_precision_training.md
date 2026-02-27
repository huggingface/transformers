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

# Mixed precision training

Full precision (fp32) training stores and computes everything in 32-bits. Mixed precision uses a lower precision (fp16 or bf16) data type for the compute intensive computations in the forward and backward passes, while keeping a "main copy" of the fp32 weights for the optimizer update. This makes compute faster and halves weights and activation memory, and preserves training stability. 

```text
┌─────────────────────────────────────────────────────┐
│           MIXED PRECISION TRAINING LOOP             │
│                                                     │
│  fp32 master weights ──cast──▶ fp16/bf16            │
│         ▲                          │                │
│         │                    FORWARD (autocast)     │
│         │                    matmuls in fp16/bf16   │
│         │                    reductions stay fp32   │
│         │                          │ loss           │
│         │                    LOSS SCALE ×S  ──fp16  │
│         │                          │                │
│         │                    BACKWARD               │
│         │                    grads in fp16/bf16     │
│         │                          │                │
│         │                    UNSCALE ÷S    ──fp16   │
│         │                    check inf/nan          │
│         │                    cast grads → fp32      │
│         └────────────────────────── optimizer.step  │
└─────────────────────────────────────────────────────┘
```

Set [`~TrainingArguments.bf16`] or [`~TrainingArguments.fp16`]  to `True` to enable mixed precision training. These are both 16-bit data types but bf16 has the same exponent range as fp32, so it almost never runs into overflow issues. Use bf16 on more modern hardware like A100, H100, or Ampere GPUs and fallback to fp16 on hardware like V100 or T4.

```py
from transformers import TrainingArguments

args = TrainingArguments(..., bf16=True)
args = TrainingArguments(..., fp16=True)
```

## tf32

[tf32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) is a compute mode on Ampere GPUs that uses 10-bit mantissa for matmuls instead of 23-bits. This can give you a speedup, especially when paired with bf16 or fp16. PyTorch enables tf32 for matmuls by default on Ampere and newer GPUs, but setting it explicitly in [`TrainingArguments`] ensures it's active regardless of the PyTorch version or environment defaults.

```py
from transformers import TrainingArguments

args = TrainingArguments(..., bf16=True, tf32=True)
```

## Next steps

- See the [Kernels](./kernels) guide to learn how to speed up training with custom fused kernels.
- See the [torch.compile](./torch_compile) guide to learn how to compile the forward and backward pass for additional throughput.
