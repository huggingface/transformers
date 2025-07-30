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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# AltCLIP

[AltCLIP](https://huggingface.co/papers/2211.06679) replaces the [CLIP](./clip) text encoder with a multilingual XLM-R encoder and aligns image and text representations with teacher learning and contrastive learning.

You can find all the original AltCLIP checkpoints under the [AltClip](https://huggingface.co/collections/BAAI/alt-clip-diffusion-66987a97de8525205f1221bf) collection.

> [!TIP]
> Click on the AltCLIP models in the right sidebar for more examples of how to apply AltCLIP to different tasks.

The examples below demonstrates how to calculate similarity scores between an image and one or more captions with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor

model = AltCLIPModel.from_pretrained("BAAI/AltCLIP", dtype=torch.bfloat16)
processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

labels = ["a photo of a cat", "a photo of a dog"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```python
# !pip install torchao
import torch
import requests
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor, TorchAoConfig

model = AltCLIPModel.from_pretrained(
    "BAAI/AltCLIP",
    quantization_config=TorchAoConfig("int4_weight_only", group_size=128),
    dtype=torch.bfloat16,
)

processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

labels = ["a photo of a cat", "a photo of a dog"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

## Notes

- AltCLIP uses bidirectional attention instead of causal attention and it uses the `[CLS]` token in XLM-R to represent a text embedding.
- Use [`CLIPImageProcessor`] to resize (or rescale) and normalize images for the model.
- [`AltCLIPProcessor`] combines [`CLIPImageProcessor`] and [`XLMRobertaTokenizer`] into a single instance to encode text and prepare images.

## AltCLIPConfig
[[autodoc]] AltCLIPConfig

## AltCLIPTextConfig
[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig
[[autodoc]] AltCLIPVisionConfig

## AltCLIPModel
[[autodoc]] AltCLIPModel

## AltCLIPTextModel
[[autodoc]] AltCLIPTextModel

## AltCLIPVisionModel
[[autodoc]] AltCLIPVisionModel

## AltCLIPProcessor
[[autodoc]] AltCLIPProcessor
