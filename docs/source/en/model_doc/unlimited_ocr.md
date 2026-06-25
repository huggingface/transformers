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
*This model was published in HF papers on 2026-06-23 and contributed to Hugging Face Transformers on 2026-06-25.*


# UnlimitedOcr

## Overview

The UnlimitedOcr model was proposed in [Unlimited OCR Works](https://huggingface.co/papers/2606.23050) by Youyang Yin, Huanhuan Liu, Qunyi Xie, Chaorun Liu, Shiqi Yang, Shaohua Wang, Zhanlong Liu, Hao Zou, Jinyue Chen, Shu Wei, Jingjing Wu, Mingxin Huang, Zhen Wu, Guibin Wang, Tengyu Du, and Lei Jia from Baidu Inc.

The abstract from the paper is the following:

*Recently, end-to-end OCR models, exemplified by DeepSeek OCR, have once again thrust OCR into the spotlight. A widely held view is that employing a large language model (LLM) as the decoder allows the model to leverage the prior distribution of language, leading to improved OCR performance. However, the downside is equally evident: as the output sequence lengthens, the accumulated KV cache drives up memory consumption and progressively slows down generation. This stands in stark contrast to humans, who exhibit no such decline in efficiency during long-horizon copying tasks. In this technical report, we propose Unlimited OCR, a model designed to emulate human parsing working memory. Taking DeepSeek OCR as the baseline, we replace all attention layers in the decoder with our proposed Reference Sliding Window Attention (R-SWA), which reduces attention computation costs while maintaining a constant KV cache throughout the entire decoding process. By combining the high compression rate of DeepSeek OCR's encoder with our constant KV cache design, Unlimited OCR can transcribe dozens of pages of documents in a single forward pass under a standard maximum length of 32K. More importantly, R-SWA is a general-purpose parsing attention mechanism — beyond OCR, it is equally applicable to tasks such as ASR, translation, etc. Codes and model weights are publicly available at http://github.com/baidu/Unlimited-OCR*

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/baidu/Unlimited-OCR).

> [!TIP]
> Unlimited-OCR supports two inference configurations: the default "gundam" mode uses 640x640 tiles with dynamic cropping for high-resolution documents, and "base" mode uses a single 1024x1024 global view for standard-resolution inputs. Gundam mode is enabled by default via the image processor (`crop_to_patches=True`).

> [!TIP]
> For multi-page documents, pass all page images together with one `<image>` token per page in the text prompt. The model processes all pages jointly within a single context window.

> [!TIP]
> The Reference Sliding Window Attention (R-SWA) applies only to generated tokens. All image and prompt tokens from the prefill remain fully visible throughout decoding, so long documents do not lose context from earlier pages.

<hfoptions id="usage">
<hfoption id="Single-page OCR">

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
inputs = processor(images=image, text="<image>\ndocument parsing.", return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# "R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext. : (...)"
```

<hfoption id="Multi-page OCR">

Multi-page documents can be parsed jointly in a single forward pass by passing all page images together. Include one `<image>` token per page in the text prompt so the model processes all pages as a continuous document.

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
    text="<image>" * num_pages + "\nMulti page parsing.",
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=32768)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

</hfoption>
</hfoptions>

## Notes

- [`UnlimitedOcrForConditionalGeneration`] extends [`DeepseekOcr2ForConditionalGeneration`](deepseek_ocr2) with a two-stage vision pipeline: a SAM ViT-B encoder feeds into a CLIP ViT encoder, and their concatenated features are projected to the language model hidden size. The 3B-parameter model supports up to 32,768 context tokens.
- The Reference Sliding Window Attention (R-SWA) in the decoder keeps the KV cache constant throughout decoding. The prefill (image tokens and prompt) is retained in full; the sliding window applies only across generated tokens.
- Image inputs are only forwarded during the first generation step. Subsequent decode steps skip `pixel_values` to avoid reprocessing the image.

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
