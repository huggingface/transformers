<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-06-23 and contributed to Hugging Face Transformers on 2026-06-23.*


# UnlimitedOcr

## Overview

The UnlimitedOcr model was proposed in [Unlimited OCR Works](https://huggingface.co/papers/2606.23050) by Youyang Yin, Huanhuan Liu, Qunyi Xie, Chaorun Liu, Shiqi Yang, Shaohua Wang, Zhanlong Liu, Hao Zou, Jinyue Chen, Shu Wei, Jingjing Wu, Mingxin Huang, Zhen Wu, Guibin Wang, Tengyu Du, and Lei Jia from Baidu Inc.

Unlimited-OCR is an OCR-specialized vision-language model designed for one-shot long-horizon parsing of single images and multi-page documents. It extends [DeepSeek-OCR-2](deepseek_ocr2) with a two-stage vision pipeline: a SAM ViT-B vision encoder extracts spatial features, which are then fed into a CLIP ViT encoder; the concatenated outputs are projected through an MLP into a DeepSeek-V2 Mixture-of-Experts language model. The 3B-parameter model supports up to 32,768 context tokens, making it suited for parsing long or multi-page documents in a single forward pass.

Tips:

- Unlimited-OCR supports two inference configurations: the default "gundam" mode uses 640×640 tiles with dynamic cropping for high-resolution documents, and "base" mode uses a single 1024×1024 global view for standard-resolution inputs. Gundam mode is enabled by default via the image processor (`crop_to_patches=True`, `tile_size=640`).
- For multi-page documents, pass all page images together with one `<image>` token per page in the text prompt. The model processes all pages jointly within a single context window.
- The sliding-window attention applies only to generated tokens. All image and prompt tokens from the prefill remain fully visible throughout decoding, so long documents do not lose context from earlier pages.
- Use `<image>\nFree OCR.` for plain text extraction and `<image>\nDocument parsing.` for richer structured output.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/baidu/Unlimited-OCR).

## Usage examples

### Single-page OCR

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
inputs = processor(images=image, text="<image>\nFree OCR.", return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# "R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext. : (...)"
```

### Document parsing

For richer structured output such as markdown-formatted documents, use the `Document parsing.` prompt:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
inputs = processor(
    images=image,
    text="<image>\nDocument parsing.",
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

### Multi-page document OCR

Multi-page documents can be parsed jointly in a single forward pass by passing all page images together. Include one `<image>` token per page in the text prompt so the model processes all pages as a continuous document:

```python
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

page1 = Image.open("page1.png")
page2 = Image.open("page2.png")
num_pages = 2

inputs = processor(
    images=[page1, page2],
    text="<image>" * num_pages + "\nMulti page document parsing.",
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=32768)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

## UnlimitedOcrConfig

[[autodoc]] UnlimitedOcrConfig

## UnlimitedOcrTextConfig

[[autodoc]] UnlimitedOcrTextConfig

## UnlimitedOcrVisionConfig

[[autodoc]] UnlimitedOcrVisionConfig

## UnlimitedOcrVisionEncoderConfig

[[autodoc]] UnlimitedOcrVisionEncoderConfig

## UnlimitedOcrSamVisionConfig

[[autodoc]] UnlimitedOcrSamVisionConfig

## UnlimitedOcrImageProcessor

[[autodoc]] UnlimitedOcrImageProcessor

## UnlimitedOcrProcessor

[[autodoc]] UnlimitedOcrProcessor

## UnlimitedOcrPreTrainedModel

[[autodoc]] UnlimitedOcrPreTrainedModel

## UnlimitedOcrTextPreTrainedModel

[[autodoc]] UnlimitedOcrTextPreTrainedModel

## UnlimitedOcrTextModel

[[autodoc]] UnlimitedOcrTextModel
    - forward

## UnlimitedOcrVisionModel

[[autodoc]] UnlimitedOcrVisionModel
    - forward

## UnlimitedOcrModel

[[autodoc]] UnlimitedOcrModel
    - forward

## UnlimitedOcrForConditionalGeneration

[[autodoc]] UnlimitedOcrForConditionalGeneration
    - forward
