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
*This model was published in HF papers on 2026-04-13 and contributed to Hugging Face Transformers on 2026-06-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TIPSv2

## Overview

[TIPSv2](https://huggingface.co/papers/2604.12012) (Text-Image Pre-training with Spatial awareness) is a family of
contrastive vision-language encoders proposed in *TIPSv2: Advancing Vision-Language Pretraining with Enhanced
Patch-Text Alignment* by Bingyi Cao et al.

The abstract from the paper is the following:

*Recent progress in vision-language pretraining has enabled significant improvements to many downstream computer vision applications, such as classification, retrieval, segmentation and depth prediction. However, a fundamental capability that these models still struggle with is aligning dense patch representations with text embeddings of corresponding concepts. In this work, we investigate this critical issue and propose novel techniques to enhance this capability in foundational vision-language models. First, we reveal that a patch-level distillation procedure significantly boosts dense patch-text alignment – surprisingly, the patch-text alignment of the distilled student model strongly surpasses that of the teacher model. This observation inspires us to consider modifications to pretraining recipes, leading us to propose iBOT++, an upgrade to the commonly-used iBOT masked image objective, where unmasked tokens also contribute directly to the loss. This dramatically enhances patch-text alignment of pretrained models. Additionally, to improve vision-language pretraining efficiency and effectiveness, we modify the exponential moving average setup in the learning recipe, and introduce a caption sampling strategy to benefit from synthetic captions at different granularities. Combining these components, we develop TIPSv2, a new family of image-text encoder models suitable for a wide range of downstream applications. Through comprehensive experiments on 9 tasks and 20 datasets, we demonstrate strong performance, generally on par with or better than recent vision encoder models. Code and models are released via our project page at https://gdm-tipsv2.github.io/.*

This model was contributed by [Ternuraz](https://huggingface.co/Ternuraz).
The original code can be found [here](https://github.com/google-deepmind/tips).

You can find all the original TIPSv2 checkpoints under the [TIPSv2](https://huggingface.co/collections/google/tipsv2) collection.

> [!TIP]
> Click on the TIPSv2 models in the right sidebar for more examples of how to apply TIPSv2 to image and text tasks.

The example below demonstrates zero-shot image classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a cat", "a dog", "a car"]

classifier = pipeline(task="zero-shot-image-classification", model="google/tipsv2-b14", device=0)
classifier(image, candidate_labels=candidate_labels)
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor


model_id = "google/tipsv2-b14"
model = AutoModel.from_pretrained(model_id, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a cat", "a dog", "a car"]
texts = [f"This is a photo of {label}." for label in candidate_labels]

inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = candidate_labels[most_likely_idx]
print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")
```

</hfoption>
</hfoptions>

## Notes

- [`Tipsv2Processor`] applies the checkpoint-compatible preprocessing defaults: lowercase text, no BOS or EOS tokens,
  text padding/truncation to length 64, image resizing to 448x448, pixel rescaling to `[0, 1]`, and no mean/std
  normalization.
- [`Tipsv2Model`] returns normalized `image_embeds` and `text_embeds`. `logits_per_image` and `logits_per_text` are
  convenience outputs computed as cosine similarity divided by the checkpoint temperature.
- Use [`~Tipsv2Model.get_image_features`] and [`~Tipsv2Model.get_text_features`] to retrieve image and text embeddings
  for retrieval or similarity scoring.
- Use [`Tipsv2VisionModel`] when you need raw vision tower outputs, including class, patch, and register tokens.

## Tipsv2Config

[[autodoc]] Tipsv2Config

## Tipsv2TextConfig

[[autodoc]] Tipsv2TextConfig

## Tipsv2VisionConfig

[[autodoc]] Tipsv2VisionConfig

## Tipsv2Tokenizer

[[autodoc]] Tipsv2Tokenizer
    - __call__

## Tipsv2ImageProcessor

[[autodoc]] Tipsv2ImageProcessor
    - preprocess

## Tipsv2Processor

[[autodoc]] Tipsv2Processor
    - __call__

## Tipsv2Model

[[autodoc]] Tipsv2Model
    - forward
    - get_text_features
    - get_image_features

## Tipsv2TextModel

[[autodoc]] Tipsv2TextModel
    - forward

## Tipsv2VisionModel

[[autodoc]] Tipsv2VisionModel
    - forward
