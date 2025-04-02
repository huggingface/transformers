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

# Selecting a quantization method

There are many quantization methods available in Transformers for inference and fine-tuning. This guide helps you choose the most common and production-ready quantization techniques depending on your use case, and presents the advantages and disadvantages of each technique.

For a comprehensive overview of all supported methods and their features, refer back to the table in the [Overview](./overview).

## Inference

Consider the quantization methods below for inference.

| quantization method | use case |
|---|---|
| bitsandbytes | ease of use and QLoRA fine-tuning on NVIDIA GPUs |
| compressed-tensors | loading specific quantized formats (FP8, Sparse) |
| GPTQModel or AWQ | good 4-bit accuracy with upfront calibration |
| HQQ | fast on the fly quantization without calibration |
| torchao | flexibility and fast inference with torch.compile |

**TLDR (Inference):**

*   For ease of use and QLoRA fine-tuning on NVIDIA GPUs: **bitsandbytes**
*   For good 4-bit accuracy with upfront calibration: **GPTQModel** or **AWQ**
*   For speed via `torch.compile` and flexibility: **torchao**
*   For fast on-the-fly quantization without calibration: **HQQ** is worth considering.
*   For loading specific formats (FP8, Sparse) and nice quantized models: **compressed-tensors** 


### bitsandbytes

*   **Description:** Enables 4-bit and 8-bit inference.
*   **Pros:**
    *   Very simple, no calibration dataset required for inference
    *   Good community support and widely adopted.
*   **Cons:**
    *   Primarily optimized for NVIDIA GPUs (CUDA).  There is [ongoing work](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx#multi-backend) to support other hardware.
    *   Inference speedup isn't guaranteed

See the [bitsandbytes documentation](./bitsandbytes) for more details.

### 2. GPTQ / GPTQModel

*   **Description:** quantizes weights (commonly to 4-bit, but supports others) by processing calibration data. 
*   **Pros:**
    *   Often achieves high accuracy
    *   Can lead to inference speedups
    *   There are many pre-quantized GPTQ models on [Hugging Face Hub](https://huggingface.co/models?other=gptq). This means you can often skip doing the quantization yourself and directly download a GPTQ-quantized model.
*   **Cons:**
    *   Requires a calibration dataset and a separate calibration step before inference. This takes time and compute resources.
    *   Accuracy depends on the quality and relevance of the calibration data (possible to overfit on calibration data).

See the [GPTQ documentation](./gptq) for more details.

### 3. AWQ (Activation-aware Weight Quantization)

*   **Description:** Similar to GPTQ in workflow, that also typically targets 4-bit quantization.
*   **Pros:**
    *   Often achieves high accuracy at 4-bit, sometimes surpassing GPTQ for specific tasks.
    *   Can lead to inference speedups
    *   Shorter calibration time than GPTQ.
    *   There are many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). This means you can often skip doing the quantization yourself and directly download a AWQ-quantized model.
*   **Cons:**
    *   Requires calibration if quantizing yourself.

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
See the [HQQ documentation](./hqq) for more details.

### 6. compressed-tensors

*   **Description:** Allows loading models saved in specific compressed formats (like FP8, INT8/4, potentially with sparsity).
*   **Pros:** 
    *   Supports flexible formats including FP8 and sparsity.
*   **Cons:** 
    *   Primarily for loading pre-quantized models, not performing the quantization within Transformers itself.

See the [compressed-tensors documentation](./compressed_tensors) for more details.

## Fine-tuning

Consider the quantization method below during fine-tuning to save memory.

### bitsandbytes[[training]]

*   **Description:** The standard method for QLoRA fine-tuning via PEFT.
*   **Pros:** Enables fine-tuning large models on consumer GPUs; widely supported and documented for PEFT.
*   **Cons:** Primarily for NVIDIA GPUs.

Other methods offer PEFT compatibility, though bitsandbytes is the most established and straightforward path for QLoRA.

See the [bitsandbytes documentation](./bitsandbytes#qlora) and [PEFT Docs](https://huggingface.co/docs/peft/developer_guides/quantization#aqlm-quantization) for more details. 

## Research

Methods like [AQLM](./aqlm), [SpQR](./spqr), [VPTQ](./vptq), [HIGGS](./higgs), etc., push the boundaries of compression (< 2-bit) or explore novel techniques.

*   Consider these if:
    *   You need extreme compression (sub-4-bit).
    *   You are conducting research or require state-of-the-art results from their respective papers.
    *   You have significant compute resources available for potentially complex quantization procedures.
We recommend consulting each methods documentation and associated papers carefully before choosing one for use in production.


> [!TIP]
> Always benchmark the performance (accuracy and speed) of the quantized model on your specific task and hardware to ensure it meets your requirements. Refer to the individual documentation pages linked above for detailed usage instructions.