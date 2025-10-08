<!--Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-02-22 and added to Hugging Face Transformers on 2022-06-28 and contributed by [xvjiarui](https://huggingface.co/xvjiarui).*

# GroupViT

[GroupViT](https://huggingface.co/papers/2202.11094) is a hierarchical Grouping Vision Transformer that learns to group image regions into progressively larger segments using only text supervision. By training jointly with a text encoder on a large-scale image-text dataset via contrastive losses, GroupViT can perform zero-shot semantic segmentation without pixel-level annotations. It achieves 52.3% mIoU on PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context, competitive with state-of-the-art transfer-learning methods that require more supervision.

<hfoptions id="usage">
<hfoption id="GroupViTModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, GroupViTModel

model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc", dtype="auto")
processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
text_labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(
    text=text_labels, images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
for i, (label, prob) in enumerate(zip(text_labels, probs[0])):
    print(f"{label}: {prob:.4f}")
```

</hfoption>
</hfoptions>

## GroupViTConfig

[[autodoc]] GroupViTConfig

## GroupViTTextConfig

[[autodoc]] GroupViTTextConfig

## GroupViTVisionConfig

[[autodoc]] GroupViTVisionConfig

## GroupViTModel

[[autodoc]] GroupViTModel
    - forward
    - get_text_features
    - get_image_features

## GroupViTTextModel

[[autodoc]] GroupViTTextModel
    - forward

## GroupViTVisionModel

[[autodoc]] GroupViTVisionModel
    - forward

