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
*This model was released on 2021-11-15 and added to Hugging Face Transformers on 2021-11-30.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# VisionTextDualEncoder

VisionTextDualEncoderModel ccombines a pretrained vision encoder (such as ViT, BEiT, or DeiT) with a pretrained text encoder (such as RoBERTa or BERT) to form a dual-encoder architecture. Each encoder’s output is mapped into a shared latent space via randomly initialized projection layers, which require fine-tuning. The model supports CLIP-style contrastive training to align image and text embeddings for tasks like zero-shot image classification and retrieval. Building on the LiT framework, it can also leverage frozen pretrained encoders to achieve stronger zero-shot performance across vision tasks.

<hfoptions id="usage">
<hfoption id="VisionTextDualEncoderModel">

```py
import torch
import requests
from PIL import Image
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
model = VisionTextDualEncoderModel.from_vision_text_pretrained("google/vit-base-patch16-224", "google-bert/bert-base-uncased")

urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
text_labels = ["a photo of a cat", "a photo of a dog"]
inputs = processor(
    text=text_labels, images=images, return_tensors="pt", padding=True
)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

for i, image_probs in enumerate(probs):
    print(f"\nImage {i+1}:")
    for j, (label, prob) in enumerate(zip(text_labels, image_probs)):
        print(f"  {label}: {prob:.4f}")
```

</hfoption>
</hfoptions>

## VisionTextDualEncoderConfig

[[autodoc]] VisionTextDualEncoderConfig

## VisionTextDualEncoderProcessor

[[autodoc]] VisionTextDualEncoderProcessor

## VisionTextDualEncoderModel

[[autodoc]] VisionTextDualEncoderModel
    - forward

