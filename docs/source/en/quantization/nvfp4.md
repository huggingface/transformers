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
they ship and quantizes full-precision checkpoints on the fly. Checkpoints declare it as `quant_method="nvfp4"`. See
[Fine-grained](./finegrained) for the requirements, loading options and parallelism every block-scaled format shares.

The NVFP4 exports of [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (`quant_method="modelopt"`,
`quant_algo="NVFP4"`) and of [compressed-tensors](./compressed_tensors) (`format="nvfp4-pack-quantized"`) load as
[`NVFP4Config`] with their calibrated activation scales, and save back in the same tensor layout. ModelOpt exports
that give different modules different formats load as a [`FineGrainedConfig`] (see [Fine-grained](./finegrained)).

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("NVFP4 is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Activation formats

`activation_format` picks what the activations are quantized to before each matmul. With a pre-quantized checkpoint,
pass the config with the `activation_format` you want; the weights load as they ship.

| `activation_format` | Scheme | Activations |
|---|---|---|
| `None` (default) or `"nvfp4"` | W4A4 | NVFP4, against the checkpoint's calibrated global scale when it ships one, on the NVFP4 MMA of Blackwell GPUs |
| `"bf16"` | W4A16 (weight-only) | kept in full precision |
