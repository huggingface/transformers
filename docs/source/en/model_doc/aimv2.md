<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-11-21 and added to Hugging Face Transformers on 2025-07-08.*

# AIMv2

## Overview

The AIMv2 model was proposed in [Multimodal Autoregressive Pre-training of Large Vision Encoders](https://huggingface.co/papers/2411.14402) by Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, Alaaeldin El-Nouby.

The abstract from the paper is the following:

*We introduce a novel method for pre-training of large-scale vision encoders. Building on recent advancements in autoregressive pre-training of vision models, we extend this framework to a multimodal setting, i.e., images and text. In this paper, we present AIMV2, a family of generalist vision encoders characterized by a straightforward pre-training process, scalability, and remarkable performance across a range of downstream tasks. This is achieved by pairing the vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. Our encoders excel not only in multimodal evaluations but also in vision benchmarks such as localization, grounding, and classification. Notably, our AIMV2-3B encoder achieves 89.5% accuracy on ImageNet-1k with a frozen trunk. Furthermore, AIMV2 consistently outperforms state-of-the-art contrastive models (e.g., CLIP, SigLIP) in multimodal image understanding across diverse settings.*


This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali).
The original code can be found [here](https://github.com/apple/ml-aim).

## Usage Example

Here is an example of Image Feature Extraction using specific checkpoints on resized images and native resolution images:

```python
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-native")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-native")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

Here is an example of a checkpoint performing zero-shot classification:

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["Picture of a dog.", "Picture of a cat.", "Picture of a horse."]

processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit")

inputs = processor(
    images=image,
    text=text,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
```

## Aimv2Config

[[autodoc]] Aimv2Config

## Aimv2TextConfig

[[autodoc]] Aimv2TextConfig

## Aimv2VisionConfig

[[autodoc]] Aimv2VisionConfig

## Aimv2Model

[[autodoc]] Aimv2Model
    - forward

## Aimv2VisionModel

[[autodoc]] Aimv2VisionModel
    - forward

## Aimv2TextModel

[[autodoc]] Aimv2TextModel
    - forward
