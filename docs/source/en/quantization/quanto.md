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

# Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a PyTorch quantization library from [Optimum](https://huggingface.co/docs/optimum/index). It replaces linear layers with quantized versions built on PyTorch, so it works on any device PyTorch supports and with any model that has `torch.nn.Linear` layers, including speech and vision models. Use it to quantize weights to 2, 4, or 8 bits on hardware other methods don't support, like MPS or CPU.

The table below summarizes what Quanto supports in Transformers.

| | |
|---|---|
| Config | [`QuantoConfig`] |
| `quant_method` | `quanto` |
| Workflow | Quantize while loading |
| Bits | 2, 4, 8 |
| Hardware | CUDA, MPS, XPU, CPU |
| Save | ❌ |
| Fine-tune | ✅ |
| Auto-compile | ❌ |

## Install Quanto

Install optimum-quanto and [Accelerate](https://huggingface.co/docs/accelerate/index).

```bash
pip install -U optimum-quanto accelerate
```

## Quantize a model with Quanto

Create a [`QuantoConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. `weights` sets the weight dtype and accepts `"int8"` (the default), `"float8"`, `"int4"`, or `"int2"`.

```py
from transformers import AutoModelForCausalLM, QuantoConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=QuantoConfig(weights="int4"),
    device_map="auto",
)
```

## Limitations

- Transformers quantizes only the weights. Setting `activations` in [`QuantoConfig`] raises a `ValueError`. Use the [Quanto](https://github.com/huggingface/optimum-quanto) library directly for activation quantization, calibration, or quantization-aware training.
- Transformers can't save Quanto models. The [Quanto](https://github.com/huggingface/optimum-quanto) library has its own way to save and reload quantized models.
- Quantizing while loading can't offload to the CPU or disk. A `device_map` that places part of the model on `"cpu"` or `"disk"` raises a `ValueError`.

## Resources

- [Quanto: a PyTorch quantization backend for Optimum](https://huggingface.co/blog/quanto-introduction) blog post
- [optimum-quanto](https://github.com/huggingface/optimum-quanto) repository
