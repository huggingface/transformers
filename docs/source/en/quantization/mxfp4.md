<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MXFP4

MXFP4 is the OCP microscaling 4-bit format: weights in 4-bit floating point (E2M1, two values per byte) with one
power-of-two scale (UE8M0) per group of 32 values along a row. It cuts weight memory about 4x against bf16, so
GPT-OSS-120B fits on a single 80GB GPU and GPT-OSS-20B needs about 16GB.

[`Mxfp4Config`] loads MXFP4 checkpoints, such as OpenAI's GPT-OSS, as they ship and quantizes full-precision
checkpoints on the fly. Checkpoints declare it as `quant_method="mxfp4"`. DeepSeek-V4 ships its routed experts in MXFP4
beside block-FP8 weights, which loads through [Fine-grained](./finegrained).

> [!TIP]
> Quantized compute needs a CUDA GPU with compute capability 8.9 or newer or an Intel XPU, and the
> [kernels](https://github.com/huggingface/kernels) package, which loads the
> [finegrained kernels](https://huggingface.co/kernels-community/finegrained-kernels) from the Hub. Without one, a
> pre-quantized checkpoint is dequantized to full precision as it loads.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

A checkpoint whose quantization config says `"quant_method": "mxfp4"` loads in MXFP4 with no config.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Pass `Mxfp4Config(dequantize=True)` to load the weights dequantized to bf16 instead, for example to fine-tune them.

## Quantize on the fly

```py
from transformers import AutoModelForCausalLM, Mxfp4Config

quantization_config = Mxfp4Config(modules_to_not_convert=["lm_head"])
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
model.save_pretrained("qwen3-30b-a3b-mxfp4")
```

## Activation formats

`activation_format` picks what the activations are quantized to before each matmul.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `"bf16"` (default) | W4A16 (weight-only) | kept in full precision, as GPT-OSS runs |
| `"mxfp8"` | W4A8 | MXFP8 |
| `None` or `"mxfp4"` | W4A4 | MXFP4 |

With a pre-quantized checkpoint, pass the config with the `activation_format` you want; the weights load as they
ship.

The 4-bit activation scheme uses the 4-bit block-scaled MMA of Blackwell GPUs.

## Resources

Learn more about MXFP4 and how blockwise scaling works in this
[blog post](https://huggingface.co/blog/faster-transformers#mxfp4-quantization).
