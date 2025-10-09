<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-11-12 and added to Hugging Face Transformers on 2023-01-04 and contributed by [jongjyh](https://huggingface.co/jongjyh).*

# AltCLIP

[AltCLIP](https://huggingface.co/papers/2211.06679v2) alters the text encoder in CLIP by replacing it with a pretrained multilingual text encoder XLM-R. This modification enables the model to achieve state-of-the-art performance on tasks such as ImageNet-CN, Flicker30k-CN, and COCO-CN, while maintaining performance close to CLIP on other tasks. The approach involves a two-stage training schema with teacher learning and contrastive learning to align language and image representations, extending CLIP's capabilities to multilingual understanding.

This model was contributed by [jongjyh](https://huggingface.co/jongjyh).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-image-classification", model="kakaobrain/align-base", dtype="auto")
candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=candidate_labels)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AltCLIPModel, AutoProcessor

model = AltCLIPModel.from_pretrained("BAAI/AltCLIP", dtype="auto")
processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

labels = ["a photo of a cat", "a photo of a dog"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

</hfoption>
</hfoptions>

## AltCLIPConfig

[[autodoc]] AltCLIPConfig
    - from_text_vision_configs

## AltCLIPTextConfig

[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig

[[autodoc]] AltCLIPVisionConfig

## AltCLIPProcessor

[[autodoc]] AltCLIPProcessor

## AltCLIPModel

[[autodoc]] AltCLIPModel
    - forward
    - get_text_features
    - get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel
    - forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel
    - forward

