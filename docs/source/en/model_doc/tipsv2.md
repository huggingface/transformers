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
*This model was published in HF papers on 2026-04-13 and contributed to Hugging Face Transformers on 2026-06-17.*

# TIPSv2

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

TIPSv2 (Text-Image Pre-training with Spatial awareness) is a family of
contrastive vision-language encoders proposed in [TIPSv2: Advancing Vision-Language Pretraining with Enhanced
Patch-Text Alignment]((https://huggingface.co/papers/2604.12012)) by Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, Joshua Ainslie, Alex Bewley, Mithun Jacob, René Wagner, Washington Ramos, Krzysztof Choromanski, Mojtaba Seyedhosseini, Howard Zhou, André Araujo.

The abstract from the paper is the following:

*Recent progress in vision-language pretraining has enabled significant improvements to many downstream computer vision applications, such as classification, retrieval, segmentation and depth prediction. However, a fundamental capability that these models still struggle with is aligning dense patch representations with text embeddings of corresponding concepts. In this work, we investigate this critical issue and propose novel techniques to enhance this capability in foundational vision-language models. First, we reveal that a patch-level distillation procedure significantly boosts dense patch-text alignment – surprisingly, the patch-text alignment of the distilled student model strongly surpasses that of the teacher model. This observation inspires us to consider modifications to pretraining recipes, leading us to propose iBOT++, an upgrade to the commonly-used iBOT masked image objective, where unmasked tokens also contribute directly to the loss. This dramatically enhances patch-text alignment of pretrained models. Additionally, to improve vision-language pretraining efficiency and effectiveness, we modify the exponential moving average setup in the learning recipe, and introduce a caption sampling strategy to benefit from synthetic captions at different granularities. Combining these components, we develop TIPSv2, a new family of image-text encoder models suitable for a wide range of downstream applications. Through comprehensive experiments on 9 tasks and 20 datasets, we demonstrate strong performance, generally on par with or better than recent vision encoder models. Code and models are released via our project page at https://gdm-tipsv2.github.io/.*

This model was contributed by [Ternuraz](https://huggingface.co/Ternuraz).
The original code can be found [here](https://github.com/google-deepmind/tips).

You can find all the original TIPSv2 checkpoints under the [TIPSv2](https://huggingface.co/collections/google/tipsv2) collection.

> [!TIP]
> See [TIPSv2 DPT](./tipsv2_dpt) for depth estimation, normal estimation, and semantic segmentation on top of the TIPSv2 vision backbone.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

classifier = pipeline(task="zero-shot-image-classification", model="google/tipsv2-b14", device_map="auto")
out = classifier(image, candidate_labels=candidate_labels)
print(out)
# [{'score': 0.997, 'label': 'a photo of a cat'}, {'score': 0.002, 'label': 'a photo of a dog'}, {'score': 0.001, 'label': 'a photo of a car'}]
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModel, AutoProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-b14"
model = AutoModel.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = candidate_labels[most_likely_idx]
print(f"Most likely label: '{most_likely_label}' with probability: {probs[0][most_likely_idx].item():.3f}")
# Most likely label: 'a photo of a cat' with probability: 0.997
```

</hfoption>
<hfoption id="get_image_features and get_text_features">

Use [`~Tipsv2Model.get_image_features`] and [`~Tipsv2Model.get_text_features`] to encode images and texts separately, which is useful for retrieval workflows where embeddings are computed independently (e.g., pre-indexing a large image database). The returned embeddings are *not* normalized, so apply L2 normalization before computing similarity.

```python
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-b14"
model = AutoModel.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

image_inputs = processor(images=image, return_tensors="pt").to(model.device)
text_inputs = processor(text=candidate_labels, return_tensors="pt").to(model.device)

with torch.no_grad():
    image_outputs = model.get_image_features(**image_inputs)
    text_outputs = model.get_text_features(**text_inputs)

image_embeds = F.normalize(image_outputs.pooler_output, dim=-1)
text_embeds = F.normalize(text_outputs.pooler_output, dim=-1)

probs = (image_embeds @ text_embeds.T / model.temperature).softmax(dim=-1)
most_likely_idx = probs.argmax(dim=-1).item()
most_likely_label = candidate_labels[most_likely_idx]
print(f"Most likely label: '{most_likely_label}' with probability: {probs[0][most_likely_idx].item():.3f}")
# Most likely label: 'a photo of a cat' with probability: 0.997
```

<hfoption id="AutoBackbone for vision feature maps">

Use [`AutoBackbone`] to load the vision backbone directly and get spatial feature maps, without the text model. 

```python
import torch
from transformers import AutoBackbone, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-b14"
backbone = AutoBackbone.from_pretrained(model_id, out_indices=[-1], device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(backbone.device)

with torch.no_grad():
    outputs = backbone(**inputs)

# feature_maps is a tuple of tensors, one per requested stage
patch_features = outputs.feature_maps[-1] # (batch_size, hidden_size, height, width) tensor
```

</hfoption>

</hfoption>
<hfoption id="Tipsv2VisionModel">

Use [`Tipsv2VisionModel`] if you only need access to the vision features. In particular, this allows you to access the two class tokens from Tipsv2 as shown below. Note that Tipsv2 repurposed the register token from [`Dinov2WithRegistersModel`] as a secondary class token. The two tokens differ in how they were trained:

- Class token 1: Supervised by web alt-text captions
- Class token 2: Supervised by PaliGemma synthetic captions

```python
import torch
from transformers import AutoConfig, Tipsv2VisionModel, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-b14"
config = AutoConfig.from_pretrained(model_id)
model = Tipsv2VisionModel.from_pretrained(model_id, config=config.vision_config, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

sequence = outputs.last_hidden_state # (batch_size, 1 + config.num_register_tokens + num_patches, hidden_size)
cls_token_1 = sequence[:, 0] # (batch_size, hidden_size)
cls_token_2 = sequence[:, 1 : 1 + model.config.num_register_tokens] # (batch_size, hidden_size)
```

</hfoption>
</hfoptions>

## Notes

- [`Tipsv2Model`] returns normalized `image_embeds` and `text_embeds`. `logits_per_image` and `logits_per_text` are convenience outputs computed as cosine similarity divided by the temperature.
- Use [`~Tipsv2Model.get_image_features`] and [`~Tipsv2Model.get_text_features`] to retrieve image and text embeddings individually.
- Use [`Tipsv2VisionBackbone`] if you need access to feature maps from all layers.
- Use [`Tipsv2VisionModel`] if you need access to the two vision class tokens.


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

## Tipsv2VisionBackbone

[[autodoc]] Tipsv2VisionBackbone
    - forward
