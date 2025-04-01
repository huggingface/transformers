<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Selecting a Quantization Method

This guide helps you navigate the most common and production-ready quantization techniques available in Transformers and associated libraries. For a comprehensive overview of all supported methods and their features, refer back to the [Quantization Overview](./overview) table.

## TLDR

*   For ease of use and QLoRA fine-tuning on NVIDIA GPUs: **bitsandbytes**
*   For good 4-bit accuracy with upfront calibration: **GPTQModel** or **AWQ**
*   For speed via `torch.compile` and flexibility: **torchao**
*   For fast on-the-fly quantization without calibration: **HQQ** is worth considering.

## Production-Ready & Widely Used Methods

These methods are generally well-tested, widely adopted, and suitable for many production scenarios.

### 1. bitsandbytes (`load_in_8bit` / `load_in_4bit`)

*   **Description:** Enables 4-bit training/inference (QLoRA using NF4) and 8-bit inference (LLM.int8).
*   **Pros:**
    *   No calibration dataset required for basic inference quantization.
    *   The standard and necessary backend for **QLoRA** fine-tuning.
    *   Good community support and widely adopted.
*   **Cons:**
    *   Primarily optimized for NVIDIA GPUs (CUDA). There is [ongoing work](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx#multi-backend) to support other hardware (Intel/AMD, Apple M-series, etc.)
    *   Inference speedup compared to fp16/bf16 might not always be significant.
    *   LLM.int8 (8-bit) can sometimes have higher accuracy degradation than PTQ methods for 8-bit.

See the [bitsandbytes documentation](./bitsandbytes) for more details.

### 2. GPTQ / GPTQModel

*   **Description:** quantizes weights (commonly to 4-bit, but supports others) by processing calibration data. 
*   **Pros:**
    *   Often achieves high accuracy
    *   Can lead to inference speedups, especially when used with optimized kernels like ExLlamaV2 or Marlin (primarily NVIDIA GPUs).
    *   `GPTQModel` library offers broader hardware support (including ROCm, CPU, Metal) and features compared to the original AutoGPTQ.
    *   There are many pre-quantized GPTQ models on [Hugging Face Hub](https://huggingface.co/models?other=gptq). This means you can often skip doing the quantization yourself and directly download a GPTQ-quantized model.
*   **Cons:**
    *   Requires a calibration dataset and a separate calibration step before inference. This takes time and compute resources.
    *   Accuracy depends on the quality and relevance of the calibration data (possible to overfit on calibration data).

See the [GPTQ documentation](./gptq) for more details.

### 3. AWQ (Activation-aware Weight Quantization)

*   **Description:** Similar to GPTQ in workflow, that also typically targets 4-bit quantization.
*   **Pros:**
    *   Often achieves high accuracy at 4-bit, sometimes surpassing GPTQ for specific tasks.
    *   Fast inference kernels and offers fused modules for potential speedups on supported architectures.
    *   Shorter calibration time than GPTQ.
    *   There are many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). This means you can often skip doing the quantization yourself and directly download a AWQ-quantized model.
*   **Cons:**
    *   Requires a calibration dataset and a separate calibration step before inference.
    *   Fused modules might have limitations (e.g., incompatibility with FlashAttention).

See the [AWQ documentation](./awq) for more details.

### 4. torchao

*   **Description:** Library from PyTorch focused on architecture optimization. It's designed for strong integration with `torch.compile`.
*   **Pros:**
    *   Strong integration with `torch.compile`, potentially leading to significant inference speedups when compiled.
    *   Offers decent CPU quantization support.
    *   Flexibility in quantization schemes (int8, int4, fp8).
*   **Cons:**
    *   Newer compared to BnB or GPTQ, so the ecosystem and best practices are still evolving.
    *   Performance gains often rely heavily on `torch.compile` working effectively for the model and hardware.
    *   4-bit quantized model via torchao might not reach the same accuracy as GPTQ/AWQ’s 4-bit, because torchao might just do straightforward rounding.

See the [torchao documentation](./torchao) for more details.

### 5. HQQ (Half-Quadratic Quantization)

*   **Description:** A fast, on-the-fly quantization method (like bitsandbytes) that doesn't require calibration data.
*   **Pros:**
    *   Fast quantization process and no calibration data required.
    *   Multiple backends for fast inference.
    *   Compatible with `torch.compile` for potential inference speedups.
    *   It supports a wide range of bit depths (8, 4, 3, 2, 1-bit)
*   **Cons:**
    *   Accuracy at very low bit depths (<4-bit) can degrade significantly and requires careful evaluation for the specific task.
    *   While quantization is fast, inference speed might not be faster than other methods unless using specific backends or `torch.compile`.
    *   Saving/loading quantized models directly with Transformers `save_pretrained` might have limitations (check HQQ docs).

See the [HQQ documentation](./hqq) for more details.

## Advanced & Research-Oriented Methods

Methods like [AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs), etc., often push the boundaries of compression (e.g., < 2-bit) or explore novel techniques.

*   Consider these if:
    *   You need extreme compression (sub-4-bit).
    *   You are conducting research or require state-of-the-art results shown in their respective papers.
    *   You have significant compute resources available for potentially complex quantization procedures.
*   Recommendation: Consult their specific documentation and associated papers carefully before choosing them for production use.


Always benchmark the performance (accuracy and speed) of the quantized model on your specific task and hardware to ensure it meets your requirements. Refer to the individual documentation pages linked above for detailed usage instructions.