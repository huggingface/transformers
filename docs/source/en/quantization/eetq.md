<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EETQ

[Easy and Efficient Quantization for Transformers (EETQ)](https://github.com/NetEase-FuXi/EETQ) quantizes weights to int8 with one scale per output channel, and runs the matmul with int8 weights and 16-bit activations. Use it for 8-bit weight-only quantization on NVIDIA GPUs without calibration data.

The table below summarizes what EETQ supports in Transformers.

| | |
|---|---|
| Config | [`EetqConfig`] |
| `quant_method` | `eetq` |
| Workflow | Quantize while loading |
| Bits | 8 |
| Hardware | CUDA |
| Save | ✅ |
| Fine-tune | ✅ |
| Auto-compile | ❌ |

## Install the EETQ kernels

Transformers downloads the EETQ kernels from the [kernels-community/quantization-eetq](https://huggingface.co/kernels-community/quantization-eetq) repository on the Hub, so you don't need to build EETQ. Install the [kernels](https://huggingface.co/docs/kernels/index) and [Accelerate](https://huggingface.co/docs/accelerate/index) libraries.

```bash
pip install -U "transformers[kernels]" accelerate
```

## Quantize a model with EETQ

Create an [`EetqConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. `weights` sets the weight dtype and only accepts `"int8"`, which is the default.

```py
import torch
from transformers import AutoModelForCausalLM, EetqConfig

quantization_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    dtype=torch.float16,
    device_map="cuda",
)
```

Load the model with `dtype=torch.float16` for the best efficiency. EETQ stores its scales in fp16.

## Limitations

- EETQ needs a CUDA GPU. Loading without one raises a `RuntimeError`.
- EETQ can't offload to the CPU or disk. A `device_map` that includes `"cpu"` or `"disk"` raises a `ValueError`.

## Resources

- [EETQ](https://github.com/NetEase-FuXi/EETQ) repository
