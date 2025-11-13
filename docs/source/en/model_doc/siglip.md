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
*This model was released on 2023-03-27 and added to Hugging Face Transformers on 2024-01-08 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SigLIP

[SigLIP](https://huggingface.co/papers/2303.15343) introduces a pairwise sigmoid loss for image-text pre-training, which, unlike traditional contrastive learning with softmax, evaluates only individual image-text pairs without requiring global normalization. This loss enables both scaling to extremely large batch sizes and strong performance at smaller batch sizes. Using just four TPUv4 chips, the authors train a Base CLIP model with a 4k batch size and a Large LiT model with a 20k batch size, achieving 84.5% ImageNet zero-shot accuracy in two days. Experiments reveal that extremely large batch sizes offer diminishing returns, with 32k being sufficient, and the approach allows systematic study of the effects of example and negative-to-positive ratios.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

pipeline = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224", dtype="auto")
pipeline(image, candidate_labels=candidate_labels)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip-base-patch16-224", dtype="auto")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

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

## Usage options

- Training supports DDP and FSDP on single-node multi-GPU setups. The model doesn't use `torch.distributed` utilities, which may limit batch size scalability.
- Use `padding="max_length"` when using standalone [`SiglipTokenizer`] or [`SiglipProcessor`]. This matches how the model was trained.
- Pass the prompt template `"This is a photo of {label}."` to the processor to get the same results as the [`Pipeline`].

## SiglipConfig

[[autodoc]] SiglipConfig

## SiglipTextConfig

[[autodoc]] SiglipTextConfig

## SiglipVisionConfig

[[autodoc]] SiglipVisionConfig

## SiglipTokenizer

[[autodoc]] SiglipTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SiglipImageProcessor

[[autodoc]] SiglipImageProcessor
    - preprocess

## SiglipImageProcessorFast

[[autodoc]] SiglipImageProcessorFast
    - preprocess

## SiglipProcessor

[[autodoc]] SiglipProcessor

## SiglipModel

[[autodoc]] SiglipModel
    - forward
    - get_text_features
    - get_image_features

## SiglipTextModel

[[autodoc]] SiglipTextModel
    - forward

## SiglipVisionModel

[[autodoc]] SiglipVisionModel
    - forward

## SiglipForImageClassification

[[autodoc]] SiglipForImageClassification
    - forward

