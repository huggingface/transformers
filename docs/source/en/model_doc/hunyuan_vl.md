<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was published in HF papers on 2025-11-24 and contributed to Hugging Face Transformers on 2026-07-03.*

# HunYuanVL

## Overview

HunYuanVL is a vision-language model for image-text understanding and generation
proposed in [HunyuanOCR Technical Report
](https://huggingface.co/papers/2511.19575). The open-source `hunyuan_vl` integration in Transformers is a
dense-only image-text variant tailored for OCR and document understanding style workloads such as [`tencent/HunyuanOCR`]((https://huggingface.co/tencent/HunyuanOCR)).

The abstract from the paper is the following:

*This paper presents HunyuanOCR, a commercial-grade, open-source, and lightweight (1B parameters) Vision-Language Model
(VLM) dedicated to OCR tasks. The architecture comprises a Native Vision Transformer (ViT) and a lightweight LLM
connected via an MLP adapter. HunyuanOCR demonstrates superior performance, outperforming commercial APIs, traditional
pipelines, and larger models (e.g., Qwen3-VL-4B). Specifically, it surpasses current public solutions in perception
tasks (Text Spotting, Parsing) and excels in semantic tasks (IE, Text Image Translation), securing first place in the
ICDAR 2025 DIMT Challenge (Small Model Track). Furthermore, it achieves state-of-the-art (SOTA) results on OCRBench
among VLMs with fewer than 3B parameters.*

*HunyuanOCR achieves breakthroughs in three key aspects: 1) Unifying Versatility and Efficiency: We implement
comprehensive support for core capabilities, including spotting, parsing, IE, VQA, and translation within a lightweight
framework. This addresses the limitations of narrow "OCR expert models" and inefficient "General VLMs". 2) Streamlined
End-to-End Architecture: Adopting a pure end-to-end paradigm eliminates dependencies on pre-processing modules (e.g.,
layout analysis). This fundamentally resolves error propagation common in traditional pipelines and simplifies system
deployment. 3) Data-Driven and RL Strategies: We confirm the critical role of high-quality data and, for the first time
in the industry, demonstrate that Reinforcement Learning (RL) strategies yield significant performance gains in OCR
tasks.*

*HunyuanOCR is officially open-sourced on HuggingFace. We also provide a high-performance deployment solution based on
vLLM, placing its production efficiency in the top tier. We hope this model will advance frontier research and provide a
solid foundation for industrial applications.*

## Recommended checkpoints

- [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) for OCR and document extraction workloads.

## Usage tips

This Transformers integration intentionally exposes the image-text path that is exercised by public OCR-style
checkpoints.

- Supported: dense-only text backbone, image-text prompting, OCR/document-understanding style generation.
- Not supported as part of this open-source variant: video inputs and runtime MoE execution paths.
- Compatibility note: some legacy Tencent-export configuration fields are still accepted so existing checkpoints can be
  loaded, but those fields do not imply that the open-source implementation enables extra runtime capabilities.
- For the currently validated OCR path, `attn_implementation="eager"` is the recommended starting point.
- `backend="pil"` is recommended when loading the processor for the current public OCR checkpoints.
- When batching variable-length prompts, pass `padding=True` if you need tensor outputs from the processor.

## Usage

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


model_name_or_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path, backend="pil")
model = AutoModelForImageTextToText.from_pretrained(
    model_name_or_path,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]) :]
output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
print(output)
```

## HunYuanVLProcessor

[[autodoc]] HunYuanVLProcessor
    - __call__

## HunYuanVLImageProcessor

[[autodoc]] HunYuanVLImageProcessor

## HunYuanVLImageProcessorPil

[[autodoc]] HunYuanVLImageProcessorPil

`HunYuanVLForConditionalGeneration` is the main public entrypoint for image-text generation. `HunYuanVLModel` exposes
the multimodal base model without the language modeling head, while `HunYuanVLTextModel` exposes the lower-level text
backbone.

## HunYuanVLConfig

[[autodoc]] HunYuanVLConfig

## HunYuanVLVisionConfig

[[autodoc]] HunYuanVLVisionConfig

## HunYuanVLTextConfig

[[autodoc]] HunYuanVLTextConfig

## HunYuanVLVisionTransformer

[[autodoc]] HunYuanVLVisionTransformer

## HunYuanVLTextModel

[[autodoc]] HunYuanVLTextModel
    - forward

## HunYuanVLModel

[[autodoc]] HunYuanVLModel
    - forward
    - get_image_features

## HunYuanVLForConditionalGeneration

[[autodoc]] HunYuanVLForConditionalGeneration
    - forward
    - get_image_features
