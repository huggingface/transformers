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

NVFP4 is NVIDIA's 4-bit floating-point format. Each block of 16 weights shares an FP8 scale, and each tensor has an FP32 scale on top. Transformers quantizes the weights while the model loads and quantizes activations on the fly, so prefill runs as a W4A4 matmul on Blackwell tensor cores and decode runs as a W4A16 matmul. Use it to run 4-bit models on a Blackwell GPU.

The table below summarizes what NVFP4 supports in Transformers.

| | |
|---|---|
| Config | [`NVFP4Config`] |
| `quant_method` | `nvfp4` |
| Workflow | Quantize while loading |
| Bits | 4 |
| Hardware | One CUDA GPU with compute capability 10.0+ (Blackwell) |
| Save | ❌ |
| Fine-tune | ❌ |
| Auto-compile | ✅ |

## Install the NVFP4 kernels

Transformers downloads the NVFP4 kernels from the [kernels-community/nvfp4-gemm](https://huggingface.co/kernels-community/nvfp4-gemm) repository on the Hub. Install the [kernels](https://huggingface.co/docs/kernels/index) library. The kernel needs a build that matches your PyTorch and CUDA versions.

```bash
pip install -U "transformers[kernels]"
```

## Quantize a model with NVFP4

Create a [`NVFP4Config`] and pass it to [`~PreTrainedModel.from_pretrained`] with a `device_map` that places the whole model on one CUDA GPU. The source checkpoint must have floating-point weights.

```py
from transformers import AutoModelForCausalLM, NVFP4Config

quantization_config = NVFP4Config()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

NVFP4 replaces each `torch.nn.Linear` layer that has no bias and whose `in_features` and `out_features` are both divisible by 16. Other linear layers keep their original precision. [`NVFP4Config`] has no parameters that change how the weights are quantized.

## Limitations

- NVFP4 needs a CUDA GPU with compute capability 10.0 or later. Loading without a CUDA GPU or on an older GPU raises a `RuntimeError`.
- NVFP4 runs on one GPU. A `device_map` that spans several GPUs, or tensor parallelism, raises a `ValueError`.
- NVFP4 can't offload to the CPU or disk. A `device_map` that includes `"cpu"` or `"disk"`, or no `device_map` at all, raises a `ValueError`.
- NVFP4 can't load pre-quantized NVFP4 checkpoints. Loading one raises a `ValueError`.
- NVFP4 models can't be saved or fine-tuned.

## Resources

- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) blog post
- [kernels-community/nvfp4-gemm](https://huggingface.co/kernels-community/nvfp4-gemm) Hub kernel
