<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-02-26 and added to Hugging Face Transformers on 2021-05-12 and contributed by [valhalla](https://huggingface.co/valhalla).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CLIP

[CLIP](https://huggingface.co/papers/2103.00020) is a neural network trained on 400 million (image, text) pairs from the internet. It learns to predict which caption corresponds to which image, enabling zero-shot transfer to various computer vision tasks. Benchmarked on over 30 datasets, CLIP demonstrates competitive performance without task-specific training, matching ResNet-50's accuracy on ImageNet zero-shot without using its training examples.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-base-patch32", dtype="auto")
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

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32", dtype="auto")

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

- Use [`CLIPImageProcessor`] to resize and normalize images for the model.

## CLIPConfig

[[autodoc]] CLIPConfig

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPImageProcessorFast

[[autodoc]] CLIPImageProcessorFast
    - preprocess

## CLIPProcessor

[[autodoc]] CLIPProcessor

## CLIPModel

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection
    - forward

## CLIPVisionModel

[[autodoc]] CLIPVisionModel
    - forward

## CLIPForImageClassification

[[autodoc]] CLIPForImageClassification
    - forward

