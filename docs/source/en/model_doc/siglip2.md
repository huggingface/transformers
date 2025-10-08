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
*This model was released on 2025-02-20 and added to Hugging Face Transformers on 2025-02-21 and contributed by [qubvel-hf](https://huggingface.co/qubvel-hf).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SigLIP2

[SigLIP2](https://huggingface.co/papers/2502.14786) introduces a family of multilingual vision-language encoders that enhance the original SigLIP with decoder-based pretraining, self-supervised losses, and online data curation. These improvements result in superior performance in zero-shot classification, image-text retrieval, and transfer learning for visual representations. SigLIP2 also excels in localization and dense prediction tasks. Available in FixRes and NaFlex variants, the models support multiple resolutions and maintain native aspect ratios. Trained on a diverse dataset with de-biasing techniques, SigLIP2 offers better multilingual understanding and fairness. Model checkpoints are provided in four sizes: ViT-B/86M, L/303M, So400m/400M, and g/1B.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

pipeline = pipeline(task="zero-shot-image-classification", model="google/siglip2-base-patch16-224", dtype="auto")
pipeline(image, candidate_labels=candidate_labels)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype="auto")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

</hfoption>
</hfoptions>

## Siglip2Config

[[autodoc]] Siglip2Config

## Siglip2TextConfig

[[autodoc]] Siglip2TextConfig

## Siglip2VisionConfig

[[autodoc]] Siglip2VisionConfig

## Siglip2ImageProcessor

[[autodoc]] Siglip2ImageProcessor
    - preprocess

## Siglip2ImageProcessorFast

[[autodoc]] Siglip2ImageProcessorFast
    - preprocess

## Siglip2Processor

[[autodoc]] Siglip2Processor

## Siglip2Model

[[autodoc]] Siglip2Model
    - forward
    - get_text_features
    - get_image_features

## Siglip2TextModel

[[autodoc]] Siglip2TextModel
    - forward

## Siglip2VisionModel

[[autodoc]] Siglip2VisionModel
    - forward

## Siglip2ForImageClassification

[[autodoc]] Siglip2ForImageClassification
    - forward

