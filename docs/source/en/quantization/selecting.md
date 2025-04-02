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

### [bitsandbytes](./bitsandbytes.md)

| Pros                                                         | Cons                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| Very simple, no calibration dataset required for inference.  | Primarily optimized for NVIDIA GPUs (CUDA).             |
| Good community support and widely adopted.                   | Inference speedup isn't guaranteed.                     |

### [GPTQ / GPTQModel](./gptq)

| Pros                                                                 | Cons                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Often achieves high accuracy.                                        | Requires a calibration dataset and a separate calibration step.      |
| Can lead to inference speedups.                                      | Possible to overfit on calibration data.                             |
| Many pre-quantized GPTQ models on [Hugging Face Hub](https://huggingface.co/models?other=gptq). |                                           |

### [AWQ (Activation-aware Weight Quantization)](./awq)

| Pros                                                                 | Cons                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| Often achieves high accuracy at 4-bit. (Sometimes surpasses GPTQ on specific tasks.) | Requires calibration if quantizing yourself.        |
| Can lead to inference speedups.                                      |                                                     |
| Shorter calibration time than GPTQ.                                  |                                                     |
| Many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). |                                                     |

### [torchao](./torchao)

| Pros                                                                 | Cons                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Strong integration with `torch.compile` for potential speedups.     | Newer library, ecosystem still evolving.                             |
| Offers decent CPU quantization support.                              | Performance depends on `torch.compile` working well.                 |
| Flexibility in quantization schemes (int8, int4, fp8).           | 4-bit quantization may not match GPTQ/AWQ in accuracy.              |

### [HQQ (Half-Quadratic Quantization)](./hqq)

| Pros                                                                 | Cons                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| Fast quantization process, no calibration data needed.              | Accuracy can degrade significantly at bit depths <4-bit.                     |
| Multiple backends for fast inference.                                | Inference speed may not match others unless using `torch.compile` or backends. |
| Compatible with `torch.compile`.                                     |                                                                            |
| Supports wide range of bit depths (8, 4, 3, 2, 1-bit).              |                                                                            |

### [compressed-tensors](./compressed_tensors)

| Pros                                                         | Cons                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| Supports flexible formats including FP8 and sparsity.        | Primarily for loading pre-quantized models.                 |
|                                                              | Doesn't perform quantization within Transformers directly.  |

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