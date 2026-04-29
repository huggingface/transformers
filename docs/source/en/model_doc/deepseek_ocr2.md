<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-28 and added to Hugging Face Transformers on 2026-04-30.*

# DeepSeek-OCR-2


## Overview

The DeepSeek-OCR-2 model was proposed in [Visual Causal Flow: A Novel Approach to OCR-Specialized Vision-Language Models](https://huggingface.co/papers/2601.20552) by the DeepSeek team.

DeepSeek-OCR-2 is an OCR-specialized vision-language model built on a distinctive architecture: a SAM ViT-B vision encoder feeds into a Qwen2 hybrid attention encoder, which is connected through an MLP projector to a DeepSeek-V2 Mixture-of-Experts (MoE) language model. A key feature of the model is its hybrid attention mechanism, which applies bidirectional attention over image tokens and causal attention over query tokens, enabling efficient and accurate document understanding.

<img src="https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/resolve/main/assets/fig1.png" width="600">

<small> DeepSeek-OCR 2: Visual Causal Flow.</small>

This model was contributed by [thisisiron](https://huggingface.co/thisisiron).


## Usage example

### Plain OCR

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "thisisiron/DeepSeek-OCR-2-hf", device_map="auto"
)
processor = AutoProcessor.from_pretrained("thisisiron/DeepSeek-OCR-2-hf")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
inputs = processor(images=image, text="<image>\nFree OCR.", return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
# "R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext. : (...)"
```

### Grounding with markdown conversion

The `<|grounding|>` token enables coordinate-aware output with `<|ref|>` and `<|det|>` tags.

```python
inputs = processor(
    images=image,
    text="<image>\n<|grounding|>Convert the document to markdown.",
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
# "<|ref|>title<|/ref|><|det|>[[330, 198, 558, 230]]<|/det|>\n# R&D QUALITY (...)"
```

## DeepseekOcr2Config

[[autodoc]] DeepseekOcr2Config

## DeepseekOcr2VisionConfig

[[autodoc]] DeepseekOcr2VisionConfig

## DeepseekOcr2SamVisionConfig

[[autodoc]] DeepseekOcr2SamVisionConfig

## DeepseekOcr2VisionEncoderConfig

[[autodoc]] DeepseekOcr2VisionEncoderConfig

## DeepseekOcr2TextConfig

[[autodoc]] DeepseekOcr2TextConfig

## DeepseekOcr2ImageProcessor

[[autodoc]] DeepseekOcr2ImageProcessor

## DeepseekOcr2ImageProcessorPil

[[autodoc]] DeepseekOcr2ImageProcessorPil

## DeepseekOcr2Processor

[[autodoc]] DeepseekOcr2Processor

## DeepseekOcr2TextModel

[[autodoc]] DeepseekOcr2TextModel

## DeepseekOcr2VisionModel

[[autodoc]] DeepseekOcr2VisionModel

## DeepseekOcr2Model

[[autodoc]] DeepseekOcr2Model

## DeepseekOcr2ForConditionalGeneration

[[autodoc]] DeepseekOcr2ForConditionalGeneration
