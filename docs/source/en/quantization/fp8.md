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

# FP8

FP8 stores weights in 8-bit floating point (`float8_e4m3fn`) with one scale per 2D block of the weight matrix
(`weight_block_size=(128, 128)` by default), or one scale per tensor (`weight_block_size=None`). Activations are
quantized to FP8 too, per token and block at run time (`activation_scheme="dynamic"`), or against a calibrated scale
shipped with the checkpoint (`activation_scheme="static"`).

It is the format of [DeepSeek-V3](https://hf.co/papers/2412.19437) and DeepSeek-R1, Qwen3-FP8, and Mistral's static
FP8 checkpoints. [`Fp8Config`] loads these checkpoints as they ship and quantizes full-precision checkpoints on the
fly. Checkpoints declare it as `quant_method="fp8"`.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/b7b3b34bf826a6423ea82ffc57ecac80c46c3c76/transformers/quantization/quantization_deepseek.png">
</div>

> [!TIP]
> Quantized compute needs a CUDA GPU with compute capability 8.9 or newer (Ada, Hopper, Blackwell) or an Intel XPU,
> and the [kernels](https://github.com/huggingface/kernels) package, which loads the
> [finegrained kernels](https://huggingface.co/kernels-community/finegrained-kernels) from the Hub. Without one, a
> pre-quantized checkpoint is dequantized to full precision as it loads.

```bash
pip install --upgrade accelerate kernels
```

## Load a pre-quantized checkpoint

A pre-quantized checkpoint carries its quantization config, so no config is needed.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B-FP8"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("What are we having for dinner?", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Checkpoints exported by [compressed-tensors](./compressed_tensors) in block FP8 (`format="float-quantized"`) load as
[`Fp8Config`] as well.

## Quantize on the fly

Pass a [`Fp8Config`] to [`~PreTrainedModel.from_pretrained`] to quantize a full-precision checkpoint as it loads.
Set `dtype="auto"` so the unquantized modules keep the checkpoint's dtype.

```py
from transformers import AutoModelForCausalLM, Fp8Config

quantization_config = Fp8Config(modules_to_not_convert=["lm_head"])
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
```

[`~PreTrainedModel.save_pretrained`] writes the quantized model back in the same layout, and
[`~PreTrainedModel.from_pretrained`] reloads it.

```py
model.save_pretrained("llama-3.1-8b-fp8")
model = AutoModelForCausalLM.from_pretrained("llama-3.1-8b-fp8", device_map="auto")
```

## Activation formats

`activation_format` picks what the activations are quantized to before each matmul.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"fp8"` | W8A8 | FP8, per token and block at run time, or against a calibrated per-tensor scale with `activation_scheme="static"` |
| `"bf16"` | W8A16 | Full precision; block-FP8 weights are decoded inside the matmul |

## Options

- `weight_block_size` is the `(rows, columns)` block each weight scale covers; `None` gives one scale per tensor.
- `activation_scheme="static"` quantizes activations against the calibrated scale a checkpoint ships (its
  `input_scale`), instead of deriving one per token at run time.
- `scale_fmt="ue8m0"` holds the block scales as powers of two (`float8_e8m0fnu`, with a `uint8` exponent fallback when the dtype is unavailable), as DeepSeek-V4 ships them.
  Float32 scale containers, such as DeepSeek-V4-Base's, load and save as native UE8M0.
- `modules_to_not_convert` keeps the modules it names in full precision; `modules_to_convert` names embedding tables to
  hold in FP8 with one per-tensor scale (a table too large to keep in full precision).
- `dequantize=True` loads a pre-quantized checkpoint dequantized to full precision.

## DeepGEMM fast path

On Hopper GPUs, a dense FP8 linear with `weight_block_size=(128, 128)` and dynamic activations dispatches to the
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from
[kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm). On Blackwell the Triton kernels are
faster and are always used.

DeepGEMM JIT-compiles its kernels with the system `nvcc`, so a full CUDA toolkit must be installed (12.3 or later on
Hopper). Transformers finds the toolkit by checking `CUDA_HOME`, then `CUDA_PATH`, then `nvcc` on your `PATH`, then
`/usr/local/cuda`. If DeepGEMM cannot load, Transformers logs a warning once and uses the Triton kernels. Set
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` to always use them.

For MoE experts, DeepGEMM is opt-in: pass `experts_implementation="deepgemm"` (or `"deepgemm_megamoe"` on Blackwell)
to [`~PreTrainedModel.from_pretrained`]. The DeepGEMM experts kernels on Blackwell read only UE8M0 scales; with
`float32` scales, use the default `grouped_mm` and `batched_mm` experts implementations. See the
[Experts backends](../experts_interface) guide for every option.

## Related formats

- [MXFP8](./mxfp8) holds FP8 weights with one power-of-two scale per 32 values of a row.
- [Fine-grained](./finegrained) mixes block FP8 with the 4-bit formats in one model, as DeepSeek-V4 ships it.
