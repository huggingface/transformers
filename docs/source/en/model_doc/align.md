<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-02-11 and added to Hugging Face Transformers on 2023-03-01 and contributed by [adirik](https://huggingface.co/adirik).*

# ALIGN

[ALIGN](https://huggingface.co/papers/2102.05918) is a multi-modal vision and language model utilizing a dual-encoder architecture with EfficientNet for vision and BERT for text. It employs contrastive learning to align visual and text representations using a noisy dataset of over one billion image-alt text pairs. Despite the noise, the scale of the dataset enables state-of-the-art performance in image classification and image-text retrieval tasks, surpassing more complex models.

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
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
model = AutoModelForZeroShotImageClassification.from_pretrained("kakaobrain/align-base", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = requests.get(url, stream=True)
inputs = Image.open(image.raw).convert("RGB")

image_inputs = processor(images=inputs, return_tensors="pt").to(model.device)
with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)

candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
text_inputs = processor(text=candidate_labels, padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)

image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

logits = (image_embeds @ text_embeds.T) * 100.0
probs  = logits.softmax(dim=-1).cpu().squeeze()

for label, score in zip(candidate_labels, probs):
    print(f"{label:20s} → {score.item():.4f}")
```

</hfoption>
</hfoptions>

## Usage tips

- ALIGN projects text and visual features into latent space. The dot product between projected image and text features becomes the similarity score. The example below shows how to calculate image-text similarity scores with [`AlignProcessor`] and [`AlignModel`].

## AlignConfig

[[autodoc]] AlignConfig

## AlignTextConfig

[[autodoc]] AlignTextConfig

## AlignVisionConfig

[[autodoc]] AlignVisionConfig

## AlignProcessor

[[autodoc]] AlignProcessor

## AlignModel

[[autodoc]] AlignModel
    - forward
    - get_text_features
    - get_image_features

## AlignTextModel

[[autodoc]] AlignTextModel
    - forward

## AlignVisionModel

[[autodoc]] AlignVisionModel
    - forward

