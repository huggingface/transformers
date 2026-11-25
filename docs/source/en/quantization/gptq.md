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

# GPTQ

[GPTQ](https://huggingface.co/papers/2210.17323) quantizes a model one layer at a time and uses a small calibration dataset to update the remaining weights so they compensate for each layer's rounding error. Transformers runs GPTQ through [Optimum](https://huggingface.co/docs/optimum/index) and the [GPT-QModel](https://github.com/ModelCloud/GPTQModel) library. Use it to quantize a model to 2 to 8 bits with calibration while it loads, or to load one of the many GPTQ checkpoints on the Hub.

The table below summarizes what GPTQ supports in Transformers.

| | |
|---|---|
| Config | [`GPTQConfig`] |
| `quant_method` | `gptq` |
| Workflow | Quantize while loading, with calibration |
| Bits | 2, 3, 4, 8 |
| Hardware | CUDA, XPU, CPU |
| Save | ✅ |
| Fine-tune | ✅ |
| Auto-compile | ❌ |

## Install GPTQ

Install Optimum 1.24.0 or later and GPT-QModel 1.4.3 or later.

```bash
pip install -U "optimum>=1.24.0"
pip install -U "gptqmodel>=1.4.3" --no-build-isolation
```

## Quantize a model with GPTQ

Create a [`GPTQConfig`] with a `dataset` to calibrate on, and pass it to [`~PreTrainedModel.from_pretrained`]. The weights are quantized after they load. GPTQ recommends fp16, so the example below sets `dtype=torch.float16`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    dtype=torch.float16,
    device_map="auto",
)
```

`dataset` is required to quantize a model. Pass the name of a dataset from the GPTQ paper, `"wikitext2"`, `"c4"`, or `"c4-new"`, or a list of strings. The `tokenizer` prepares the dataset. If you don't pass one, GPTQ loads the tokenizer from the model's repository.

Calibration runs the model block by block, so it takes much longer than loading. Before you quantize a model, check the Hub for an existing GPTQ checkpoint.

## Load a GPTQ checkpoint

Pass the checkpoint to [`~PreTrainedModel.from_pretrained`] without a [`GPTQConfig`]. Transformers reads the quantization settings from the checkpoint's config.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    device_map="auto",
)
```

To change the kernel or buffer size, pass a [`GPTQConfig`] with the new `backend` or `max_input_length`. Transformers overrides only these two loading attributes and keeps the rest of the checkpoint's settings.

## Configure GPTQ

These [`GPTQConfig`] parameters change how the weights are quantized. See [`GPTQConfig`] for the rest.

| Parameter | Default | Effect |
|---|---|---|
| `bits` | required | Number of bits per weight. Supports 2, 3, 4, and 8. |
| `group_size` | `128` | Number of weights that share a scale. `-1` uses one scale per column. |
| `damp_percent` | `0.1` | Fraction of the average Hessian diagonal added for numerical stability. Must be between 0 and 1. |
| `desc_act` | `False` | Quantizes columns in order of decreasing activation size (act-order). Can improve accuracy and slow down inference. Setting it to `True` turns off `act_group_aware`. |
| `sym` | `True` | Uses symmetric quantization. Set to `False` for asymmetric quantization. |
| `modules_in_block_to_quantize` | `None` | Lists of module names in each block to quantize, in order. Linear layers not listed stay unquantized. With the default, every linear layer in the block is quantized. Requires Optimum 1.15.0 or later. |

## Choose a kernel backend

GPT-QModel runs the quantized layers through one of several kernels. The `backend` parameter picks one and defaults to `"auto"`, which lets GPT-QModel choose a kernel for your hardware and checkpoint. The example below loads a checkpoint with the ExLlamaV2 kernel.

```py
from gptqmodel import BACKEND
from transformers import AutoModelForCausalLM, GPTQConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    quantization_config=GPTQConfig(bits=4, backend=BACKEND.EXLLAMA_V2),
    device_map="auto",
)
```

See the GPT-QModel [backend list](https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py) for the available values.

## Limitations

- GPTQ only supports text models. Loading a model whose main input isn't `input_ids` raises a `RuntimeError`.
- Quantizing a model with disk offloading in its `device_map` raises a `ValueError`.
- Loading without GPT-QModel installed raises an `ImportError`. AutoGPTQ isn't supported.
- Without a `device_map`, the model loads on the CPU.

## Resources

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://huggingface.co/papers/2210.17323) paper
- [GPT-QModel](https://github.com/ModelCloud/GPTQModel) repository
- [Optimum GPTQ](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) guide
- [GPTQ quantization with PEFT](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) notebook
