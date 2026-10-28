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

NVFP4 is NVIDIA's 4-bit floating-point format. Weights are E2M1 values with an E4M3 scale per 16 values and an fp32
global per tensor. Transformers runs NVFP4 linears and MoE experts on the
[finegrained-kernels](https://huggingface.co/kernels-community/finegrained-kernels) Hub kernel.

> [!TIP]
> NVFP4 needs the [kernels](https://github.com/huggingface/kernels) package and a GPU with FP8 support: an NVIDIA GPU
> with compute capability 8.9 or newer, an AMD GPU or an Intel XPU. Without one, a pre-quantized checkpoint loads
> dequantized to the model dtype.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

NVIDIA ModelOpt exports load without a quantization config. NVFP4 layers keep the checkpoint's activation scales, so
their activations quantize to NVFP4 too. Layers exported as `W4A16_NVFP4` keep their activations in the model dtype.

```py
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "nvidia/Qwen3-30B-A3B-NVFP4"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("NVFP4 is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

A `MIXED_PRECISION` export may quantize some layers to FP8, block FP8 or MXFP8, for example its attention. Those layers
load dequantized to the model dtype, and a warning says so.

An MXFP8 checkpoint with `"expert_dtype": "nvfp4"` in its quantization config also loads with NVFP4. Its experts load
in NVFP4, and its other MXFP8 layers load dequantized.

The tensor names say how a checkpoint stores its NVFP4 tensors. ModelOpt names the globals `weight_scale_2` and
`input_scale`. compressed-tensors (`nvfp4-pack-quantized`) names the packed weight `weight_packed` and ships the
reciprocal of each global. Pass `checkpoint_format` to [`NVFP4Config`] to set the layout yourself.

## Quantize while loading

Pass [`NVFP4Config`] to quantize a full-precision checkpoint as it loads. The linears and MoE experts are quantized.
MoE routers stay in the model dtype. Use `modules_to_not_convert` to keep other modules in their original precision.

```py
import torch

from transformers import AutoModelForCausalLM, NVFP4Config


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    device_map="cuda",
    quantization_config=NVFP4Config(modules_to_not_convert=["lm_head"]),
)
```

## MoE experts

Pick the experts' kernels with `experts_implementation`: `"grouped_mm"` or `"batched_mm"` run the fused MoE chain, and
`"eager"` runs one call per expert. On Blackwell GPUs, scales load swizzled into the layout the scaled MMA reads.

[`DistributedConfig`] shards the experts in one of two ways:

- With `ep_size` equal to `tp_size`, each rank holds whole experts, with masked expert parallelism. Each rank runs its own
  experts on every token, so there is no all-to-all dispatch.
- With `tp_size` alone, each rank holds a slice of every expert.

```py
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(tp_size=8, ep_size=8)
```

## Saving

[`~PreTrainedModel.save_pretrained`] writes the NVFP4 tensors in the checkpoint layout, under the modules' own names.
A model quantized on the fly saves quantized, so it reloads without quantizing again.

```py
model.save_pretrained("Qwen3-30B-A3B-NVFP4-local")
```

## Current limitations

- CPU and disk offload are not supported.
- NVFP4 models cannot be trained.
