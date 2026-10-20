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

# MXFP8

MXFP8 is the OCP microscaling FP8 format: weights in 8-bit floating point (`float8_e4m3fn`) with one power-of-two
scale (UE8M0) per group of 32 values along a row. [`Mxfp8Config`] loads MXFP8 checkpoints, such as `MiniMaxAI/MiniMax-M3-MXFP8`, as
they ship and quantizes full-precision checkpoints on the fly. Checkpoints declare it as `quant_method="mxfp8"`.

> [!TIP]
> Quantized compute needs a CUDA GPU with compute capability 8.9 or newer or an Intel XPU, and the
> [kernels](https://github.com/huggingface/kernels) package. Without one, a pre-quantized checkpoint is dequantized to
> full precision as it loads. On Blackwell, the group scales feed the tensor cores' block-scaled MMA directly.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

```py
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("MiniMaxAI/MiniMax-M3-MXFP8", dtype="auto", device_map="auto")
```

## Quantize on the fly

```py
from transformers import AutoModelForCausalLM, Mxfp8Config

quantization_config = Mxfp8Config(modules_to_not_convert=["lm_head"])
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
model.save_pretrained("qwen3-30b-a3b-mxfp8")
```

## Activation formats

`activation_format` picks what the activations are quantized to before each matmul.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"mxfp8"` | W8A8 | MXFP8 |
| `"mxfp4"` | W8A4 | MXFP4 |
| `"bf16"` | W8A16 (weight-only) | kept in full precision |

With a pre-quantized checkpoint, pass the config with the `activation_format` you want; the weights load as they
ship.

Other options are shared with the other formats: `modules_to_not_convert` keeps modules in full precision, and
`dequantize=True` loads a pre-quantized checkpoint dequantized.

## Related formats

- [FP8](./fp8) holds FP8 weights with one scale per 2D block or per tensor.
- [MXFP4](./mxfp4) is the 4-bit microscaling format.
- [Fine-grained](./finegrained) mixes MXFP8 with other formats in one model.
