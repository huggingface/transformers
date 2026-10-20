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

FP8 stores weights in 8-bit floating point (`float8_e4m3fn`). By default, each 128x128 block of the weight matrix has
one scale (`weight_block_size=(128, 128)`). Set `weight_block_size=None` for one scale per tensor. Activations are
quantized to FP8 too. With `activation_scheme="dynamic"`, they are quantized per token and block at run time. With
`activation_scheme="static"`, they use a calibrated scale shipped with the checkpoint.

FP8 is the format of [DeepSeek-V3](https://hf.co/papers/2412.19437), DeepSeek-R1, Qwen3-FP8 and the static FP8
checkpoints of Mistral. [`Fp8Config`] loads these checkpoints as they ship. It also quantizes full-precision
checkpoints on the fly. Checkpoints declare the format as `quant_method="fp8"`. Block and per-tensor FP8 exports of
[compressed-tensors](./compressed_tensors) (`format="float-quantized"`) also load as [`Fp8Config`]. See
[Fine-grained](./finegrained) for the requirements, loading options and parallelism shared by every block-scaled
format.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/b7b3b34bf826a6423ea82ffc57ecac80c46c3c76/transformers/quantization/quantization_deepseek.png">
</div>

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, Fp8Config

# a pre-quantized checkpoint loads without a config
model_id = "Qwen/Qwen3-8B-FP8"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("What are we having for dinner?", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# a full-precision checkpoint is quantized as it loads
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype="auto", device_map="auto", quantization_config=Fp8Config()
)
```

## Options

See [`Fp8Config`] for every option. `modules_to_convert` names the FP8 embedding tables of a pre-quantized checkpoint.
Quantizing on the fly keeps embedding tables in full precision.

## DeepGEMM fast path

On Hopper GPUs, a dense FP8 linear with `weight_block_size=(128, 128)` and dynamic activations runs on the
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from
[kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm). The Triton kernels serve static activations and models spread
over several GPUs. They also serve every linear on Blackwell, where they are faster.

DeepGEMM is opt-in for MoE experts. Pass `experts_implementation="deepgemm"` to [`~PreTrainedModel.from_pretrained`],
or `"deepgemm_megamoe"` on Blackwell. On Blackwell, the DeepGEMM experts kernels only read UE8M0 scales. Use the
default `grouped_mm` or `batched_mm` experts implementations for `float32` scales. See the
[Experts backends](../experts_interface) guide for every option.

DeepGEMM compiles its kernels at run time with the system `nvcc`. A full CUDA toolkit must be installed, and the CUDA
libraries alone are not enough. DeepGEMM needs CUDA 12.3 or later on Hopper and 12.9 or later on Blackwell.
Transformers looks for the toolkit in this order.

1. `CUDA_HOME`
2. `CUDA_PATH`
3. `nvcc` on your `PATH`
4. `/usr/local/cuda`

If DeepGEMM cannot load, Transformers logs a warning once and uses the Triton kernels. Set
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` to always use the Triton kernels for linears. It does not affect the
`"deepgemm"` experts backend.

## Migrating from FineGrainedFP8Config

`FineGrainedFP8Config` is deprecated. When passed to [`~PreTrainedModel.from_pretrained`], it still loads through its
own frozen integration. A checkpoint that declares `quant_method="fp8"` loads as [`Fp8Config`] when no config is
passed. New code uses the config of the checkpoint format, with the same field names.

Use [`Fp8Config`] with the same arguments.
