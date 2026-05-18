<!--Copyright 2026 Baidu and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-03-18 and added to Hugging Face Transformers on 2026-04-16.*

# QianfanOCR


## Overview

**Qianfan-OCR** is a 4B-parameter end-to-end document intelligence model developed by the Baidu Qianfan Team. It was proposed in [Qianfan-OCR: A Unified End-to-End Model for Document Intelligence](https://huggingface.co/papers/2603.13398) by Daxiang Dong et al.

Unlike traditional multi-stage OCR pipelines, Qianfan-OCR performs **direct image-to-text conversion** and supports a broad range of prompt-driven tasks — from structured document parsing and table extraction to chart understanding, document question answering, and key information extraction — all within one model.

The model adopts a multimodal bridging architecture consisting of three components:
- **Vision Encoder**: Qianfan-ViT with AnyResolution design (up to 4K), 256 visual tokens per 448×448 tile, max 4,096 tokens per image
- **Language Model**: Qwen3-4B with 32K context (extendable to 131K)
- **Cross-Modal Adapter**: 2-layer MLP with GELU activation

A key innovation is **Layout-as-Thought**: an optional thinking phase triggered by `<think>` tokens, where the model generates structured layout representations (bounding boxes, element types, reading order) before producing final outputs. This is particularly useful for heterogeneous pages with mixed element types (exam papers, technical reports, newspapers).

The model achieves state-of-the-art results on several benchmarks:
- **#1 end-to-end model on OmniDocBench v1.5** with an overall score of 93.12
- **#1 end-to-end model on OlmOCR Bench** with a score of 79.8
- **#1 on Key Information Extraction** with a mean score of 87.9 across five public KIE benchmarks

This model was contributed by the [Baidu Qianfan Team](https://github.com/baidubce/Qianfan-VL).

## Usage example

### Document parsing

```python
from transformers import AutoModelForImageTextToText, AutoProcessor


model = AutoModelForImageTextToText.from_pretrained("baidu/Qianfan-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Qianfan-OCR")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
messages = [{"role": "user", "content": [{"type": "image", "url": image}, {"type": "text", "text": "Parse this document to Markdown."}]}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=64)
processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

### Layout-as-Thought (thinking mode)

For documents with complex layouts, cluttered elements, or non-standard reading orders, enable thinking mode by setting `enable_thinking=True` in `apply_chat_template`. The model will first generate structured layout analysis (bounding boxes, element types, reading order), then produce the final output.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor


model = AutoModelForImageTextToText.from_pretrained("baidu/Qianfan-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Qianfan-OCR")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
messages = [{"role": "user", "content": [{"type": "image", "url": image}, {"type": "text", "text": "Parse this document to Markdown."}]}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=True).to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=128)
processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

### Batched inference

```python
from transformers import AutoModelForImageTextToText, AutoProcessor


model = AutoModelForImageTextToText.from_pretrained("baidu/Qianfan-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Qianfan-OCR")

image1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
image2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
messages = [
    [{"role": "user", "content": [{"type": "image", "url": image1}, {"type": "text", "text": "Parse this document to Markdown."}]}],
    [{"role": "user", "content": [{"type": "image", "url": image2}, {"type": "text", "text": "OCR the text in the image."}]}],
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True).to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=64)
processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

## QianfanOCRConfig

[[autodoc]] QianfanOCRConfig

## QianfanOCRVisionConfig

[[autodoc]] QianfanOCRVisionConfig

## QianfanOCRProcessor

[[autodoc]] QianfanOCRProcessor
    - __call__

## QianfanOCRVisionModel

[[autodoc]] QianfanOCRVisionModel
    - forward

## QianfanOCRModel

[[autodoc]] QianfanOCRModel
    - forward

## QianfanOCRForConditionalGeneration

[[autodoc]] QianfanOCRForConditionalGeneration
    - forward
