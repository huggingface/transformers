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
<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="Transformers" src="https://img.shields.io/badge/Transformers-6B5B95?style=flat&logo=transformers&logoColor=white">
  </div>
</div>

# ALIGN

[ALIGN](https://huggingface.co/papers/2102.05918) is pretrained on a noisy 1.8 billion alt‑text and image pair dataset to show that scale can make up for the noise. It uses a dual‑encoder architecture, [EfficientNet](./efficientnet) for images and [BERT](./bert) for text, and a contrastive loss to align similar image–text embeddings together while pushing different embeddings apart. Once trained, ALIGN can encode any image and candidate captions into a shared vector space for zero‑shot retrieval or classification without requiring extra labels. This scale‑first approach reduces dataset curation costs and powers state‑of‑the‑art image–text retrieval and zero‑shot ImageNet classification.

You can find all the original ALIGN checkpoints under the [Kakao Brain](https://huggingface.co/kakaobrain?search_models=align) organization.

> [!TIP]
> Click on the ALIGN models in the right sidebar for more examples of how to apply ALIGN to different vision and text related tasks.

The example below demonstrates zero-shot image classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">  

<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="zero-shot-image-classification",
    model="kakaobrain/align-base",
    device=0,
    dtype=torch.bfloat16
)

candidate_labels = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a person"
]

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
model = AutoModelForZeroShotImageClassification.from_pretrained("kakaobrain/align-base").to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = requests.get(url, stream=True)
inputs = Image.open(image.raw).convert("RGB")

image_inputs = processor(images=inputs, return_tensors="pt").to("cuda")
with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)

candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
text_inputs = processor(text=candidate_labels, padding=True, return_tensors="pt").to("cuda")
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

## Notes

- ALIGN projects the text and visual features into latent space and the dot product between the projected image and text features is used as the similarity score. The example below demonstrates how to calculate the image-text similarity score with [`AlignProcessor`] and [`AlignModel`].

  ```py
  # Example of using ALIGN for image-text similarity
  from transformers import AlignProcessor, AlignModel
  import torch
  from PIL import Image
  import requests
  from io import BytesIO
  
  # Load processor and model
  processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
  model = AlignModel.from_pretrained("kakaobrain/align-base")
  
  # Download image from URL
  url = "https://huggingface.co/roschmid/dog-races/resolve/main/images/Golden_Retriever.jpg"
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))  # Convert the downloaded bytes to a PIL Image
  
  texts = ["a photo of a cat", "a photo of a dog"]
  
  # Process image and text inputs
  inputs = processor(images=image, text=texts, return_tensors="pt")
  
  # Get the embeddings
  with torch.no_grad():
      outputs = model(**inputs)
  
  image_embeds = outputs.image_embeds
  text_embeds = outputs.text_embeds
  
  # Normalize embeddings for cosine similarity
  image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
  text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
  
  # Calculate similarity scores
  similarity_scores = torch.matmul(text_embeds, image_embeds.T)
  
  # Print raw scores
  print("Similarity scores:", similarity_scores)
  
  # Convert to probabilities
  probs = torch.nn.functional.softmax(similarity_scores, dim=0)
  print("Probabilities:", probs)
  
  # Get the most similar text
  most_similar_idx = similarity_scores.argmax().item()
  print(f"Most similar text: '{texts[most_similar_idx]}'")
  ```

## Resources
- Refer to the [Kakao Brain’s Open Source ViT, ALIGN, and the New COYO Text-Image Dataset](https://huggingface.co/blog/vit-align) blog post for more details.

## AlignConfig

[[autodoc]] AlignConfig
    - from_text_vision_configs

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
