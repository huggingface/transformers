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

# HIGGS

[HIGGS](https://huggingface.co/papers/2411.17525) applies a Hadamard transform to each weight matrix and then rounds groups of weights to a grid that minimizes the mean squared error, without calibration data. Use it for 2- to 4-bit weights on NVIDIA GPUs, with the [FLUTE](https://github.com/HanGuo97/flute) kernels running the matmuls.

The table below summarizes what HIGGS supports in Transformers.

| | |
|---|---|
| Config | [`HiggsConfig`] |
| `quant_method` | `higgs` |
| Workflow | Quantize while loading |
| Bits | 2, 3, 4 |
| Hardware | CUDA |
| Save | ✅ |
| Fine-tune | ❌ |
| Auto-compile | ❌ |

## Install HIGGS

Install FLUTE 0.3.0 or later, fast-hadamard-transform, and [Accelerate](https://huggingface.co/docs/accelerate/index).

```bash
pip install -U "flute-kernel>=0.3.0" fast_hadamard_transform accelerate
```

## Quantize a model with HIGGS

Create a [`HiggsConfig`] and pass it to [`~PreTrainedModel.from_pretrained`]. Set `device_map` to your CUDA devices, and load the model in fp16 or bf16.

```py
from transformers import AutoModelForCausalLM, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=HiggsConfig(bits=4),
    device_map="cuda",
)
```

After the weights load, FLUTE tunes its kernels for each layer on your GPU and repacks the weights. The tuning results are stored in the config's `tune_metadata`, so a saved model doesn't need to tune again.

## Configure HIGGS

These [`HiggsConfig`] parameters change how the weights are quantized.

| Parameter | Default | Effect |
|---|---|---|
| `bits` | `4` | Number of bits per weight. Supports 2, 3, and 4. |
| `p` | `2` | Dimension of the quantization grid, either 1 or 2. A grid with `p=2` rounds pairs of weights together and is more accurate. |
| `group_size` | `256` | Number of weights that share a scale. Supports 64, 128, and 256. Must divide `hadamard_size`. |
| `hadamard_size` | `512` | Size of the Hadamard transform. Each layer's input dimension is padded to a multiple of this size. Smaller values reduce accuracy. |

## Speed up inference with torch.compile

Without compilation, HIGGS runs slower than bf16. [`~GenerationMixin.generate`] doesn't compile HIGGS models automatically, so compile the model yourself with [torch.compile](../perf_torch_compile).

```py
model = torch.compile(model)
```

The table below shows forward passes per second for Llama-3.1-8B-Instruct on an RTX 4090.

| Batch size | bf16 (compiled) | HIGGS 4-bit | HIGGS 4-bit (compiled) |
|---|---|---|---|
| 1 | 59 | 41 | 124 |
| 4 | 57 | 42 | 123 |
| 16 | 56 | 41 | 120 |

## Limitations

- HIGGS needs a CUDA GPU. Loading without one raises a `NotImplementedError`.
- Loading without a `device_map`, or with one that includes `"cpu"` or `"disk"`, raises a `ValueError`.
- Loading in a dtype other than fp16 or bf16 raises a `ValueError`.

## Resources

- [Pushing the Limits of Large Language Model Quantization via the Linearity Theorem](https://huggingface.co/papers/2411.17525) paper
- [FLUTE](https://github.com/HanGuo97/flute) repository
- [HIGGS models](https://huggingface.co/collections/ISTA-DASLab/higgs-675308e432fd56b7f6dab94e) collection from ISTA-DASLab
