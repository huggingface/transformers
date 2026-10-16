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

NVFP4 is NVIDIA's 4-bit floating-point format: weights in E2M1 (two values per byte) with two levels of scale, an FP8
(E4M3) scale per group of 16 values and one FP32 global scale per tensor. [`NVFP4Config`] loads NVFP4 checkpoints as
they ship and quantizes full-precision checkpoints on the fly. Checkpoints declare it as `quant_method="nvfp4"`.

> [!TIP]
> Quantized compute needs a CUDA GPU with compute capability 8.9 or newer or an Intel XPU, and the
> [kernels](https://github.com/huggingface/kernels) package, which loads the
> [finegrained kernels](https://huggingface.co/kernels-community/finegrained-kernels) from the Hub. Without one, a
> pre-quantized checkpoint is dequantized to full precision as it loads. The 4-bit activation scheme uses the NVFP4
> MMA of Blackwell GPUs.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

The NVFP4 exports of [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (`quant_method="modelopt"`,
`quant_algo="NVFP4"`) and of [compressed-tensors](./compressed_tensors) (`format="nvfp4-pack-quantized"`) load as
[`NVFP4Config`], with their calibrated activation scales, and save back in their own layout. A ModelOpt export
labelled `quant_algo="NVFP4"` whose `config_groups` give different modules different formats loads as a
[`FineGrainedConfig`]. ModelOpt `quant_algo="MIXED_PRECISION"` uses its explicit `quantized_layers` mapping; see
[Fine-grained](./finegrained) for supported formats and restrictions.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("NVFP4 is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Pass `NVFP4Config(dequantize=True)` to load linear and expert weights dequantized to bf16 instead.
Checkpoints with explicit NVFP4 embedding targets keep those tables packed and decode only selected rows during
lookup; they currently reject `dequantize=True`, including the automatic CPU fallback.

## Quantize on the fly

```py
from transformers import AutoModelForCausalLM, NVFP4Config

quantization_config = NVFP4Config(modules_to_not_convert=["lm_head"])
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
model.save_pretrained("llama-3.1-8b-nvfp4")
```

## Activation formats

`activation_format` picks what the activations are quantized to before each matmul.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"nvfp4"` | W4A4 | NVFP4, against the checkpoint's calibrated global scale when it ships one |
| `"bf16"` | W4A16 (weight-only) | kept in full precision |

With a pre-quantized checkpoint, pass the config with the `activation_format` you want; the weights load as they
ship.

`modules_to_not_convert` keeps the modules it names in full precision.
