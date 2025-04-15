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

### No Calibration Required (On-the-fly Quantization)

These methods are generally easier to use as they don't need a separate calibration dataset or step.

#### bitsandbytes

| Pros                                                         | Cons                                                    |
|--------------------------------------------------------------|---------------------------------------------------------|
| Very simple, no calibration dataset required for inference.  | Primarily optimized for NVIDIA GPUs (CUDA).             |
| Good community support and widely adopted.                   | Inference speedup isn't guaranteed.                     |

See the [bitsandbytes documentation](./bitsandbytes) for more details.

#### HQQ (Half-Quadratic Quantization)

| Pros                                                                 | Cons                                                                       |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| Fast quantization process, no calibration data needed.              | Accuracy can degrade significantly at bit depths <4-bit.                     |
| Multiple backends for fast inference.                                | Inference speed may not match others unless using `torch.compile` or backends. |
| Compatible with `torch.compile`.                                     |                                                                            |
| Supports wide range of bit depths (8, 4, 3, 2, 1-bit).              |                                                                            |

See the [HQQ documentation](./hqq) for more details.

#### torchao

| Pros                                                                 | Cons                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Strong integration with `torch.compile` for potential speedups.     | Newer library, ecosystem still evolving.                             |
| Offers decent CPU quantization support.                              | Performance depends on `torch.compile` working well.                 |
| Flexibility in quantization schemes (int8, int4, fp8).           | 4-bit quantization (int4wo) may not match GPTQ/AWQ in accuracy.              |

See the [torchao documentation](./torchao) for more details.

### Calibration-based Quantization

These methods require an upfront calibration step using a dataset to potentially achieve higher accuracy.

#### GPTQ/GPTQModel

Calibration for 8B model takes ~20 minutes on one A100 gpu.

| Pros                                                                 | Cons                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Often achieves high accuracy.                                        | Requires a calibration dataset and a separate calibration step.      |
| Can lead to inference speedups.                                      | Possible to overfit on calibration data.                             |
| Many pre-quantized GPTQ models on [Hugging Face Hub](https://huggingface.co/models?other=gptq). |                                           |

See the [GPTQ documentation](./gptq) for more details.

#### AWQ (Activation-aware Weight Quantization)

Calibration for 8B model takes ~10 minutes on one A100 gpu.

| Pros                                                                 | Cons                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| Often achieves high accuracy at 4-bit. (Sometimes surpasses GPTQ on specific tasks.) | Requires calibration if quantizing yourself.        |
| Can lead to inference speedups.                                      |                                                     |
| Shorter calibration time than GPTQ.                                  |                                                     |
| Many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). |                                                     |

See the [AWQ documentation](./awq) for more details.

### Loading Specific Formats

#### compressed-tensors

| Pros                                                         | Cons                                                        |
|--------------------------------------------------------------|-------------------------------------------------------------|
| Supports flexible formats including FP8 and sparsity.        | Primarily for loading pre-quantized models.                 |
|                                                              | Doesn't perform quantization within Transformers directly.  |

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

## Benchmark Comparison

To provide a quantitative comparison of different quantization methods, we benchmarked several popular techniques on the Llama 3.1 8B and 70B models. The following tables show results for accuracy (higher is better), inference throughput measured in tokens/second (higher is better), peak VRAM usage measured in GB (lower is better), and quantization time.

Performance metrics were measured on 2 NVIDIA A100 80GB GPU for Llama 3.1 70B (bfloat16), 1 NVIDIA H100 80GB GPU for FP8 methods, and 1 NVIDIA A100 80GB GPU for all other methods. Throughput was measured with a batch size of 1 and generating 64 tokens.
Results for `torch.compile` and Marlin kernels are included where applicable and supported.

<iframe
  src="https://huggingface.co/datasets/derekl35/quantization-benchmarks/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
  title="benchmarking results dataset"
></iframe>

The key takeaways are:

| Quantization & Methods                      | Memory Savings (vs bf16) | Accuracy             | Other Notes                                                        |
|-------------------------------------------- |------------------------- |--------------------- |------------------------------------------------------------------- |
| **8-bit** (bnb-int8, HQQ, Quanto, torchao, fp8) | ~2x             | Very close to baseline bf16 model   |                                                                    |
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4)    | ~4x                      | Relatively high accuracy            | AWQ/GPTQ often lead in accuracy but need calibration. HQQ/bnb-nf4 are easy on-the-fly. |
| **Sub-4-bit** (VPTQ, AQLM, 2-bit GPTQ) | Extreme (>4x)            | Noticeable drop, especially at 2-bit | Quantization times can be very long (AQLM, VPTQ). Performance varies. |

> [!TIP]
> Always benchmark the performance (accuracy and speed) of the quantized model on your specific task and hardware to ensure it meets your requirements. Refer to the individual documentation pages linked above for detailed usage instructions.