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

MXFP4 is the 4-bit floating-point format from the [Open Compute Project Microscaling (MX) specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). It stores E2M1 values in blocks that share one power-of-two (E8M0) scale. In Transformers, MXFP4 quantizes the mixture-of-experts layers of [GPT-OSS](../model_doc/gpt_oss) models, so use it to run GPT-OSS with less memory.

The table below summarizes what MXFP4 supports in Transformers.

| | |
|---|---|
| Config | [`Mxfp4Config`] |
| `quant_method` | `mxfp4` |
| Workflow | Quantize while loading, load pre-quantized |
| Bits | 4 |
| Hardware | CUDA (compute capability 7.5+), XPU, CPU |
| Save | ✅ |
| Fine-tune | ❌ |
| Auto-compile | ❌ |

## Install MXFP4

Transformers downloads the MXFP4 Triton kernels from the [kernels-community/gpt-oss-triton-kernels](https://huggingface.co/kernels-community/gpt-oss-triton-kernels) repository on the Hub. Install the [kernels](https://huggingface.co/docs/kernels/index) library, [Accelerate](https://huggingface.co/docs/accelerate/index), and Triton. CUDA needs Triton 3.4.0 or later, and XPU and CPU need Triton 3.5.0 or later.

```bash
pip install -U "transformers[kernels]" accelerate "triton>=3.4.0"
```

## Quantize a model with MXFP4

Create a [`Mxfp4Config`] and pass it to [`~PreTrainedModel.from_pretrained`] to quantize a bf16 GPT-OSS checkpoint while it loads. MXFP4 quantizes only the expert weights. The attention layers, router, embeddings, and `lm_head` keep their original precision.

```py
from transformers import AutoModelForCausalLM, Mxfp4Config

quantization_config = Mxfp4Config()
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gpt-oss-20b-BF16",
    quantization_config=quantization_config,
    device_map="cuda",
)
```

If the hardware or packages MXFP4 needs aren't available, set `dequantize=True` to load the model in bf16 instead.

## Load a pre-quantized checkpoint

The [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) and [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) checkpoints are stored in MXFP4, and their config sets `quant_method` to `mxfp4`. Load them without a [`Mxfp4Config`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="cuda")
```

If your hardware or packages can't run MXFP4, Transformers logs a warning and dequantizes a pre-quantized checkpoint to bf16 instead of raising an error.

## Choose a kernel backend

By default, MXFP4 runs the expert layers with the Triton kernels on CUDA and XPU. Pass `use_kernels=True` to [`~PreTrainedModel.from_pretrained`] to run them with the [MegaBlocks](https://huggingface.co/kernels-community/megablocks) kernel from the Hub instead.

```py
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="cpu", use_kernels=True)
```

The backend you get depends on the device.

| Device | `use_kernels=False` | `use_kernels=True` |
|---|---|---|
| CUDA | MXFP4 with Triton kernels | Dequantizes to bf16 |
| XPU | MXFP4 with Triton kernels | MXFP4 with MegaBlocks |
| CPU | Dequantizes to bf16 | MXFP4 with MegaBlocks |

## Limitations

- MXFP4 quantizes only GPT-OSS expert layers. Other models load without any quantized layers, and Transformers logs a warning.
- MXFP4 needs a CUDA GPU with compute capability 7.5 or later, an XPU, or a CPU. Quantizing a model on unsupported hardware or without Triton or kernels raises a `ValueError`. Quantizing a model on a machine with another accelerator, like MPS, raises a `RuntimeError`.
- Quantizing a model while it loads can't offload to disk. A `device_map` that includes `"disk"` raises a `ValueError`.
- MXFP4 models can't be fine-tuned. Load the model with `dequantize=True` to train it in bf16.

## Resources

- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) specification
- [MXFP4 quantization](https://huggingface.co/blog/faster-transformers#mxfp4-quantization) section of the gpt-oss tricks blog post
- [kernels-community/gpt-oss-triton-kernels](https://huggingface.co/kernels-community/gpt-oss-triton-kernels) Hub kernel
