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
fly. Checkpoints declare it as `quant_method="fp8"`, and [compressed-tensors](./compressed_tensors) block or per-tensor
FP8 exports (`format="float-quantized"`) load as [`Fp8Config`] as well. See [Fine-grained](./finegrained) for the
requirements, loading options and parallelism every block-scaled format shares.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/b7b3b34bf826a6423ea82ffc57ecac80c46c3c76/transformers/quantization/quantization_deepseek.png">
</div>

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, Fp8Config

# a pre-quantized checkpoint loads with no config
model_id = "Qwen/Qwen3-8B-FP8"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("What are we having for dinner?", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# a full-precision one is quantized as it loads
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype="auto", device_map="auto", quantization_config=Fp8Config()
)
```

## Options

- `weight_block_size` is the `(rows, columns)` block each weight scale covers; `None` gives one scale per tensor.
- `activation_scheme="static"` quantizes activations against the calibrated scale a checkpoint ships (its
  `input_scale`), instead of deriving one per token at run time.
- `scale_fmt="ue8m0"` holds the block scales as powers of two (`float8_e8m0fnu`), as DeepSeek-V4 ships them. Float32
  scale containers, such as DeepSeek-V4-Base's, load and save as native UE8M0.
- `modules_to_convert` names the embedding tables a checkpoint ships in FP8 with one per-tensor scale, such as
  Qwen4-Exp's n-gram table. Quantizing on the fly leaves embeddings in full precision.

## DeepGEMM fast path

On Hopper GPUs, a dense FP8 linear with `weight_block_size=(128, 128)` and dynamic activations dispatches to the
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from
[kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm). Static activations, a model spread
over several GPUs, and every linear on Blackwell, where they are faster, use the Triton kernels.

For MoE experts, DeepGEMM is opt-in: pass `experts_implementation="deepgemm"` (or `"deepgemm_megamoe"` on Blackwell)
to [`~PreTrainedModel.from_pretrained`]. On Blackwell the DeepGEMM experts kernels read only UE8M0 scales; with
`float32` scales, use the default `grouped_mm` and `batched_mm` experts implementations. See the
[Experts backends](../experts_interface) guide for every option.

DeepGEMM JIT-compiles its kernels with the system `nvcc`, so a full CUDA toolkit must be installed; a runtime-only
install (CUDA libraries without `nvcc`) won't work. It needs CUDA 12.3 or later on Hopper and 12.9 or later on
Blackwell. Transformers finds the toolkit by checking `CUDA_HOME`, then `CUDA_PATH`, then `nvcc` on your `PATH`, then
`/usr/local/cuda`. If DeepGEMM cannot load, Transformers logs a warning once and uses the Triton kernels. Set
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` to always use them for linears; it leaves the `"deepgemm"` experts
backend untouched.

## Migrating from FineGrainedFP8Config

`FineGrainedFP8Config` is deprecated. Passed to [`~PreTrainedModel.from_pretrained`], it still loads through its own
frozen integration, while a checkpoint declaring `quant_method="fp8"` loads as [`Fp8Config`] when no config is passed.
New code uses the config of the checkpoint's format, with the same field names:

| `FineGrainedFP8Config` | Use |
|---|---|
| `weight_block_size=[1, 32]` with `scale_fmt="ue8m0"` | [`Mxfp8Config`] |
| any other `weight_block_size` | [`Fp8Config`] |
| on a model whose config declares `expert_dtype="fp4"` (DeepSeek-V4) | [`Fp8Config`], which loads as a [`FineGrainedConfig`] |
