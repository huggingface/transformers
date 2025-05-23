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

You can find all the original ALIGN checkpoints under the [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) collection.

> [!TIP]
> Click on the ALIGN models in the right sidebar for more examples of how to apply ALIGN to different vision and text related tasks.

The example below demonstrates how to retrieve zero-shot image labels with [Pipeline] or the [AutoModel] class.

<hfoptions id="usage">  
<hfoption id="Pipeline">

```py
from transformers import pipeline
from PIL import Image
import requests

# Initialize the zero-shot image-classification pipeline with ALIGN
pipe = pipeline(
    task="zero-shot-image-classification",
    model="kakaobrain/align-base"
)

# Fetch and open the image from a URL
# you can provide any image you want, this is just for example usecase
url = "https://huggingface.co/roschmid/dog-races/resolve/main/images/Golden_Retriever.jpg"
response = requests.get(url, stream=True)
image = Image.open(response.raw)

# Define candidate captions or labels
candidate_labels = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a person"
]

# Run zero-shot classification
outputs = pipe(image, candidate_labels=candidate_labels)
print(outputs)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AlignProcessor, AlignModel
from PIL import Image
import requests
import torch

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load processor and model
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model     = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
model.eval()

# 2. Fetch and open image
url      = "https://huggingface.co/roschmid/dog-races/resolve/main/images/Golden_Retriever.jpg"
response = requests.get(url, stream=True)
image    = Image.open(response.raw).convert("RGB")

# 3. Prepare inputs
#   a) image embeddings
image_inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)

#   b) text embeddings
candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
text_inputs      = processor(text=candidate_labels, padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)

# 4. Normalize embeddings
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

# 5. Compute logits and probabilities
logits = (image_embeds @ text_embeds.T) * 100.0
probs  = logits.softmax(dim=-1).cpu().squeeze()

# 6. Display results
for label, score in zip(candidate_labels, probs):
    print(f"{label:20s} → {score.item():.4f}")
```

</hfoption>
<hfoption id="transformers-cli">

```py
# this command downloads the kakaobrain/align-base for offline use
transformers-cli download kakaobrain/align-base
```

</hfoption>
</hfoptions>

### Quantization

Quantizing `align-base` to 8-bit or 4-bit does not reduce memory usage—in fact, it can increase it. This is because the model is relatively small, and components like the vision encoder aren’t optimized for quantized execution. Quantization benefits are typically seen in larger models (e.g., LLMs) where memory and compute reductions are more significant. Use full precision unless you're experimenting or working under extreme constraints.

---

### Attention Mask Visualization

ALIGN is a **dual‑encoder** (separate image+text) model with **no** autoregressive decoding head, so it does **not** support AttentionMaskVisualizer.  
That utility only works on models with a `.generate()` step that emits token‑level attentions (e.g. GPT‑style or encoder‑decoder models).

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

## AlignConfig

[[autodoc]] AlignConfig - from_text_vision_configs

## AlignTextConfig

[[autodoc]] AlignTextConfig

## AlignVisionConfig

[[autodoc]] AlignVisionConfig

## AlignProcessor

[[autodoc]] AlignProcessor

## AlignModel

[[autodoc]] AlignModel - forward - get_text_features - get_image_features

## AlignTextModel

[[autodoc]] AlignTextModel - forward

## AlignVisionModel

[[autodoc]] AlignVisionModel - forward
