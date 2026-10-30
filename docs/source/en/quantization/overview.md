<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Overview

Quantization stores a model's weights, and sometimes its activations, in fewer bits so the model needs less memory. Models are usually loaded in bf16, which takes 2 bytes per parameter. An 8B model needs about 16GB for its weights in bf16, about 8GB at 8 bits, and about 4–5GB at 4 bits. Fewer bits cost some accuracy, and each quantization method makes a different tradeoff between memory, speed, and accuracy.

Transformers integrates more than 20 quantization methods through [`~PreTrainedModel.from_pretrained`], so you use the same loading code whichever method you choose.

## Two ways to use a quantized model

How you use quantization depends on whether your checkpoint is already quantized.

| You have | What you do | Methods |
|---|---|---|
| A full-precision checkpoint | Pass a quantization config to `from_pretrained`, and the weights are quantized as they load. | [Quantize a model while loading](#quantize-a-model-while-loading) |
| A checkpoint that's already quantized | Call `from_pretrained` without a quantization config. Transformers reads the method from the checkpoint's `config.json`, or from the GGUF file you name. | [Load a pre-quantized checkpoint](#load-a-pre-quantized-checkpoint) |

Some methods, such as AWQ, AutoRound, AQLM, compressed-tensors, and GGUF, can only be loaded from a checkpoint. You quantize the model with the method's own library first, and then load the result in Transformers.

## Quantize a model while loading

Create the method's config class and pass it to `quantization_config`. The example below loads an 8B model in 4 bits with [bitsandbytes](./bitsandbytes).

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
```

Every method in this group has its own config class, like [`TorchAoConfig`] or [`BitsAndBytesConfig`]. Most of them don't need a calibration dataset. [GPTQ](./gptq) is the exception because it calibrates on a dataset while the model loads.

## Load a pre-quantized checkpoint

A pre-quantized checkpoint stores a `quantization_config` in its `config.json`. When `from_pretrained` finds it, the matching quantizer is selected automatically.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-alpha-AWQ",
    device_map="auto",
)
print(model.config.quantization_config.quant_method)
```

Install the library that the method depends on before loading the checkpoint. Each method page lists what to install. To change loading options, like which kernel backend runs, pass a config of the same type. Its options are merged with the ones stored in the checkpoint.

### GGUF checkpoints

A GGUF checkpoint is a single file that holds the weights, config, and tokenizer, and it has no `config.json` for `from_pretrained` to read. Name the file with `gguf_file`.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3.5-4B-GGUF",
    gguf_file="Qwen3.5-4B-Q4_K_M.gguf",
)
```

Most GGUF models are dequantized to full precision as they load, so they use as much memory as an unquantized model. Weights stay packed only on MPS with the [kernels](https://huggingface.co/docs/kernels/index) package installed, for Qwen3.5 and Qwen3.5 MoE. See [GGUF](./gguf) for details.

## Save a quantized model

Call [`~PreTrainedModel.save_pretrained`] or [`~PreTrainedModel.push_to_hub`] to save the quantized weights and their `quantization_config`. Not every method can save its quantized weights.

## Choose a method

[Choose a method](./selecting) recommends a starting point for common goals and compares every method by hardware, bit width, saving, fine-tuning, and `torch.compile` support. Use the [quantization picker](https://huggingface.co/spaces/stevhliu/quantization-picker) to check whether a model fits in your hardware's memory at each precision.

## Other tools

These tools quantize models outside Transformers.

- [Optimum](https://huggingface.co/docs/optimum/index) quantizes models for specific hardware and runtimes, like Intel CPUs and ONNX Runtime.
- Hub Spaces quantize a model and push the result to your account: [bitsandbytes](https://huggingface.co/spaces/bnb-community/bnb-my-repo), [GGUF](https://huggingface.co/spaces/ggml-org/gguf-my-repo), and [MLX](https://huggingface.co/spaces/mlx-community/mlx-my-repo).

## Resources

- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) course
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth) course

## Next steps

- Read [Quantization concepts](./concept_guide) for how quantization schemes, granularity, and techniques work.
- Quantize the KV cache to save memory during long generations with the [quantized cache](../kv_cache#quantized-cache).
- Quantize a model while exporting it with [Quantizing exports](../exporters_quantization).
- Contribute a new method with [Contribute a quantization method](./contribute).
