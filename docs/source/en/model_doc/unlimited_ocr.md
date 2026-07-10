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
*This model was published in HF papers on 2026-06-23 and contributed to Hugging Face Transformers on 2026-07-09.*


# UnlimitedOcr

## Overview

The UnlimitedOcr model was proposed in [Unlimited OCR Works](https://huggingface.co/papers/2606.23050) by Youyang Yin, Huanhuan Liu, Qunyi Xie, Chaorun Liu, Shiqi Yang, Shaohua Wang, Zhanlong Liu, Hao Zou, Jinyue Chen, Shu Wei, Jingjing Wu, Mingxin Huang, Zhen Wu, Guibin Wang, Tengyu Du, and Lei Jia from Baidu Inc.

It is a single 3B parameter model with a standard context length of 32,768 tokens.

The abstract from the paper is the following:

*Recently, end-to-end OCR models, exemplified by DeepSeek OCR, have once again thrust OCR into the spotlight. A widely held view is that employing a large language model (LLM) as the decoder allows the model to leverage the prior distribution of language, leading to improved OCR performance. However, the downside is equally evident: as the output sequence lengthens, the accumulated KV cache drives up memory consumption and progressively slows down generation. This stands in stark contrast to humans, who exhibit no such decline in efficiency during long-horizon copying tasks. In this technical report, we propose Unlimited OCR, a model designed to emulate human parsing working memory. Taking DeepSeek OCR as the baseline, we replace all attention layers in the decoder with our proposed Reference Sliding Window Attention (R-SWA), which reduces attention computation costs while maintaining a constant KV cache throughout the entire decoding process. By combining the high compression rate of DeepSeek OCR's encoder with our constant KV cache design, Unlimited OCR can transcribe dozens of pages of documents in a single forward pass under a standard maximum length of 32K. More importantly, R-SWA is a general-purpose parsing attention mechanism — beyond OCR, it is equally applicable to tasks such as ASR, translation, etc. Codes and model weights are publicly available at http://github.com/baidu/Unlimited-OCR*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/unlimited_ocr_architecture.png" width="600">

Unlimited-OCR supports two inference configurations: the default "gundam" mode uses 640x640 tiles with dynamic cropping for high-resolution documents, and "base" mode uses a single 1024x1024 global view for standard-resolution inputs. To enable base mode set `crop_to_patches=False`.

The vision tower follows the two-stage approach from [DeepSeek-OCR-2](./deepseek_ocr2): a SAM ViT-B encoder feeds into a CLIP ViT encoder. Unlike DeepSeek-OCR-2, the CLIP features are additionally concatenated with the SAM features to yield the final image tokens. Unlimited-OCR also omits the learnable patch queries from DeepSeek-OCR-2.

The text model is identical to DeepSeek-OCR-2 with the additional Reference Sliding Window Attention (R-SWA). R-SWA applies only to generated tokens. All image and prompt tokens remain fully visible throughout decoding, so long documents do not lose context from earlier pages.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/baidu/Unlimited-OCR).

> [!TIP]
> For multi-page documents, pass all page images together with one `<image>` token per page in the text prompt. The model processes all pages jointly within a single context window.

<hfoptions id="usage">
<hfoption id="Single-page OCR">

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
inputs = processor(images=image, text="<image>document parsing.", return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=4096,
    no_repeat_ngram_size=35,
    no_repeat_ngram_window_size=128,
)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# image [383, 87, 497, 171]\ntext [333, 201, 558, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

### Batch processing

For batch processing, pass multiple images and prompts at once. Set `padding=True` for the processor
if images have different sizes.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
image2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"
inputs = processor(
    images=[image1, image2],
    text=["<image>document parsing.", "<image>document parsing."],
    padding=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=4096,
    no_repeat_ngram_size=35,
    no_repeat_ngram_window_size=128,
)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# image [383, 87, 497, 171]\ntext [333, 201, 558, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

### Region detections

Set `skip_special_tokens=False` to wrap all detections and region types in `<|det|>...<|/det|>` markers. This is useful for further post-processing of the output, for example to plot the detected bounding boxes on the image. Each detection is wrapped as `<|det|>region_type [x1, y1, x2, y2]<|/det|>` with coordinates normalized to a `[0, 999]` range. Parse the markers with a regular expression and rescale the coordinates to the image size to plot the bounding boxes.

```python
import re
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
inputs = processor(images=image, text="<image>document parsing.", return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=4096,
    no_repeat_ngram_size=35,
    no_repeat_ngram_window_size=128,
)
decoded = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
# <|det|>image [383, 87, 497, 171]<|/det|>\n<|det|>text [333, 201, 558, 230]<|/det|>R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

detections = re.findall(r"<\|det\|>(\S+) \[(\d+), (\d+), (\d+), (\d+)\]<\|/det\|>", decoded)

# Visualization
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers.image_utils import load_image

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg")
width, height = image.size

figure, axis = plt.subplots(figsize=(10, 12))
axis.imshow(image)
for region_type, x1, y1, x2, y2 in detections:
    x1, y1, x2, y2 = int(x1) / 999 * width, int(y1) / 999 * height, int(x2) / 999 * width, int(y2) / 999 * height
    color = (random.random(), random.random(), random.random())
    rectangle = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none")
    axis.add_patch(rectangle)
    axis.text(x1, y1, region_type, color="white", fontsize=8, backgroundcolor=color, verticalalignment="top")
axis.axis("off")
plt.show()
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/unlimited_ocr_suggestion_form_boxes.jpg" width="600">


<hfoption id="Multi-page OCR">

Multi-page documents can be parsed jointly in a single forward pass by passing all page images together. Include one `<image>` token per page in the text prompt so the model processes all pages as a continuous document.

```python
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

page1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
page2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"
num_pages = 2

inputs = processor(
    images=[page1, page2],
    text="<image>" * num_pages + "Multi page parsing.",
    crop_to_patches=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=32768,
    no_repeat_ngram_size=35,
    no_repeat_ngram_window_size=1024,
)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# <PAGE>image [382, 87, 489, 174]\ntitle [333, 201, 556, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SCLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

</hfoption>
</hfoptions>

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

## UnlimitedOcrGenerationConfig

[[autodoc]] UnlimitedOcrGenerationConfig

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

## UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor

[[autodoc]] UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor
