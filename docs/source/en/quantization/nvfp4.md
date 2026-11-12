<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# NVFP4

> [!WARNING]
> NVFP4 support is limited: Blackwell GPUs only, no tensor parallelism, no CPU or disk offload, no saving or training,
> and MoE experts only from pre-quantized checkpoints. See [Current limitations](#current-limitations).

NVFP4 quantization packs full-precision linear weights into NVIDIA's 4-bit floating-point format while a model is
loaded. [`NVFP4Config`] replaces eligible bias-free `torch.nn.Linear` modules, whose `in_features` and `out_features` are both divisible by 16, with an NVFP4 linear implementation from
the [NVFP4 Hub kernel](https://huggingface.co/kernels-community/nvfp4-gemm). The model's attention and MLP interfaces are
not replaced.

> [!TIP]
> NVFP4 requires a Blackwell GPU with compute capability 10.0 or newer, a compatible CUDA-enabled PyTorch build, and
> the [kernels](https://github.com/huggingface/kernels) package.

Install Accelerate and a compatible version of `kernels`.

```bash
pip install --upgrade accelerate kernels
```

Pass [`NVFP4Config`] to [`~PreTrainedModel.from_pretrained`] with a single CUDA device. Weights are quantized as they
are loaded, so the source checkpoint should contain floating-point weights.

```py
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, NVFP4Config


model_id = "meta-llama/Llama-3.2-1B"
quantization_config = NVFP4Config()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("NVFP4 is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use `modules_to_not_convert` to keep selected modules in their original precision.

```py
quantization_config = NVFP4Config(modules_to_not_convert=["vision", "lm_head"])
```

## Pre-quantized checkpoints

Pre-quantized NVFP4 checkpoints store each linear as `weight` (e2m1 pairs), `weight_scale` (fp8 scales per 16
values), `weight_scale_2` (a per-tensor scale) and `input_scale`, MoE experts included. This is the layout NVIDIA's
[ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) exports, e.g.
[nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4). Such checkpoints load without a
`quantization_config`. ModelOpt's config names the format with a `quant_algo`, one for the whole model or, with
`MIXED_PRECISION`, one per layer (e.g. NVFP4 experts with FP8 attention), which loads as a
[quantization plan](./overview#quantization-plans).

```py
model = AutoModelForCausalLM.from_pretrained("nvidia/Qwen3.6-35B-A3B-NVFP4", device_map="cuda")
```

Pass `NVFP4Config(dequantize=True)` to load such a checkpoint in full precision. Dequantizing is pure PyTorch, so it
needs neither the kernel nor a Blackwell GPU.

NVFP4 linear modules support `torch.compile`. The first compiled invocation includes graph compilation time, so warm up
the model before measuring generation throughput.

## Current limitations

- Quantizing while loading runs on one CUDA device. A pre-quantized checkpoint can span several with a `device_map`.
- Expert parallelism shards NVFP4 experts. Tensor parallelism of NVFP4 layers is rejected, since their scales have no
  sharding plan yet.
- CPU and disk offload are not supported.
- MoE experts are only quantized from pre-quantized checkpoints, not on the fly.
- Weight-only (`W4A16_NVFP4`) layers dequantize their weights in each forward, without the kernels.
- NVFP4 models cannot currently be serialized with [`~PreTrainedModel.save_pretrained`] or trained.
