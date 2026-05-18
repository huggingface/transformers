<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-grained FP8

Fine-grained FP8 quantization quantizes the weights and activations to fp8.

- The weights are quantized to 8-bits for each 2D block (`weight_block_size=(128, 128)`).
- The activations are quantized to 8-bits for each group per token. The group value matches the weights in the input channel (128 by default).

FP8 quantization enables support for [DeepSeek-V3](https://hf.co/papers/2412.19437) and DeepSeek-R1.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/b7b3b34bf826a6423ea82ffc57ecac80c46c3c76/transformers/quantization/quantization_deepseek.png">
</div>

> [!TIP]
> You need a GPU with Compute Capability>=9 (H100), and install a PyTorch version compatible with the CUDA version of your GPU.

Install Accelerate and upgrade to the latest version of PyTorch.

```bash
pip install --upgrade accelerate torch
```

Create a [`FineGrainedFP8Config`] class and pass it to [`~PreTrainedModel.from_pretrained`] to quantize it. The weights are loaded in full precision (`torch.float32`) by default regardless of the actual data type the weights are stored in. Set `dtype="auto"` to load the weights in the data type defined in a models `config.json` file to automatically load the most memory-optimal data type.

```py
from transformers import FineGrainedFP8Config, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = FineGrainedFP8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(quantized_model.device.type)

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use [`~PreTrainedModel.save_pretrained`] to save the quantized model and reload it with [`~PreTrainedModel.from_pretrained`].

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```

## DeepGEMM fast path

On Hopper (SM90+) and Blackwell (SM100+) GPUs with CUDA runtime 12.3 or later, every FP8 linear automatically dispatches to the [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from [kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm) when `weight_block_size=(128, 128)` and `activation_scheme="dynamic"`. DeepGEMM is 3-6x faster than the Triton fallback. Install or upgrade the [kernels](https://github.com/huggingface/kernels) package to enable it.

```bash
pip install -U kernels
```

If the kernel cannot load (missing `kernels`, unsupported GPU, or older CUDA), Transformers logs a warning once and falls back to the Triton finegrained-fp8 kernel. Static activation quantization always stays on the Triton path.

For MoE experts, the DeepGEMM path is opt-in. Pass `experts_implementation="deepgemm"` (or `"deepgemm_megamoe"` on Blackwell) at load time to route the expert matmuls through DeepGEMM. See the [Experts backends](../experts_interface) guide for the full set of options.

## UE8M0 scale format

DeepSeek V4-style checkpoints store FP8 weight scales in the packed `float8_e8m0fnu` format instead of `float32`. Pass `scale_fmt="ue8m0"` to load these checkpoints. UE8M0 scales must take the DeepGEMM path (the Triton fallback only reads `float32` scales) so the same hardware and `kernels` requirements apply.

```py
from transformers import FineGrainedFP8Config

quantization_config = FineGrainedFP8Config(scale_fmt="ue8m0")
```
