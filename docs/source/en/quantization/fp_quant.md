<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FP-Quant

[FP-Quant](https://github.com/IST-DASLab/FP-Quant) is a family of quantization algorithms tailored for the Blackwell generation of Nvidia GPUs. The goal is to allow for efficient post-training quantization (PTQ) and quantization-aware training (QAT) of LLMs in the [MXFP4 and NVFP4 data-types](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

Currently, only PTQ with MXFP4 is supported. Models can either be quantized on the fly with `quantization_config=FPQuantConfig()`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(),
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

or pre-processed with GPTQ for better quality (see [FP Format Quantization Harness](https://github.com/IST-DASLab/FP-Quant)).

A **Blackwell-generation GPU is required** to run the kernels. Runtime support for FP-Quant is implemented through the [QuTLASS](https://github.com/IST-DASLab/qutlass) library and a lightweight PyTorch interface lib [`fp_quant`](https://github.com/IST-DASLab/FP-Quant/tree/master/inference_lib). We recommend installing the former **from source** and the latter with  `pip install fp_quant`.

Users **without a Blackwell-generation GPU** , can use the method with `quantization_config=FPQuantConfig(pseudoquant=True)` without having to install [QuTLASS](https://github.com/IST-DASLab/qutlass). This would provide no speedups but would fully emulate the effect of quantization.

> [!TIP]
> Find models pre-quantized with FP-Quant in the official ISTA-DASLab [collection](https://huggingface.co/collections/ISTA-DASLab/fp-quant-6877c186103a21d3a02568ee).

## torch.compile

FP-Quant is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(),
    device_map="cuda",
    dtype=torch.bfloat16,
)

model.forward = torch.compile(model.forward, mode="max-autotune", fullgraph=True)
```

## Speedups

FP-Quant currently performs best for very large batch size processing.

See [QuTLASS README](https://github.com/IST-DASLab/qutlass/blob/main/README.md) for speedups.
