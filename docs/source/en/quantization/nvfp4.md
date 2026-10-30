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

NVFP4 is NVIDIA's 4-bit floating-point format. Transformers runs NVFP4 linears and MoE experts on the
[finegrained-kernels](https://huggingface.co/kernels-community/finegrained-kernels) Hub kernel.

> [!TIP]
> NVFP4 needs the [kernels](https://github.com/huggingface/kernels) package and a GPU with FP8 support: an NVIDIA GPU
> with compute capability 8.9 or newer, an AMD GPU or an Intel XPU.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

NVIDIA ModelOpt exports load without a quantization config.

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

## Quantize while loading

Pass [`NVFP4Config`] to quantize a full-precision checkpoint as it loads.

```py
import torch

from transformers import AutoModelForCausalLM, NVFP4Config


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, device_map="cuda", quantization_config=NVFP4Config()
)
```

## MoE experts

`experts_implementation="grouped_mm"` runs the experts with the fused MoE kernels, and is the fastest for prefill.
`"batched_mm"` is the fastest for decode.

```py
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Qwen3-30B-A3B-NVFP4", dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm"
)
```

On several GPUs, use expert parallelism: each GPU holds whole experts. Launch the script with
`torchrun --nproc_per_node 8 script.py`.

```py
import torch

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig


model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Qwen3-30B-A3B-NVFP4",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=8, ep_size=8),
)
```

## Saving

An NVFP4 model saves and reloads like any other, including one quantized while loading.

```py
model.save_pretrained("Qwen3-30B-A3B-NVFP4-local")
```

## Current limitations

- CPU and disk offload are not supported.
- NVFP4 models cannot be trained.
