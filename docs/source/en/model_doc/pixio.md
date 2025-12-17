<!--Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-16.*
*This model is to be announced*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Pixio

[Pixio]() is a vision foundation model that uses [ViT](./vit) as a feature extractor for multiple downstream tasks like depth estimation, semantic segmentation, feed-forward 3D reconstruction, robotics, and image classification. It is built on the Masked Autoencoder (MAE) pre-training framework, with four minimal yet critical updates: 1) deeper decoder, 2) larger masking granularity, 3) more class tokens, and 4) web-scale curated training data.

You can find all the original Pixio checkpoints under the [Pixio]() collection.

The example below demonstrates how to obtain an image embedding with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("facebook/pixio-vith16")
model = AutoModel.from_pretrained("facebook/pixio-vith16")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
features_norm = outputs.last_hidden_state # class tokens + patch tokens after last LayerNorm
features = outputs.hidden_states[-1] # class tokens + patch tokens before last LayerNorm
```

## Notes

- The example below shows how to split the output tensor into:
  - a set of global embeddings for the whole image, commonly referred to as `CLS` token,
    useful for classification and retrieval. 
    You can either average them (recommended) or concatenate them along the channel dimension.
  - a set of local embeddings, one for each `16x16` patch of the input image,
    useful for dense tasks, such as depth estimation and semantic segmentation.

  ```py
  from transformers import AutoImageProcessor, AutoModel
  from PIL import Image
  import requests

  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  image = Image.open(requests.get(url, stream=True).raw)
  print(image.height, image.width)  # [480, 640]

  processor = AutoImageProcessor.from_pretrained('facebook/pixio-vith16')
  model = AutoModel.from_pretrained('facebook/pixio-vith16')
  patch_size = model.config.patch_size

  inputs = processor(images=image, return_tensors="pt")
  print(inputs.pixel_values.shape)  # [1, 3, 256, 256]
  batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
  num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
  num_patches_flat = num_patches_height * num_patches_width

  outputs = model(**inputs)
  last_hidden_states = outputs.last_hidden_state
  print(last_hidden_states.shape)  # [1, 8 + 256, 1280]
  assert last_hidden_states.shape == (batch_size, model.config.n_cls_tokens + num_patches_flat, model.config.hidden_size)

  cls_tokens = last_hidden_states[:, :model.config.n_cls_tokens, :]
  patch_features = last_hidden_states[:, model.config.n_cls_tokens:, :].unflatten(1, (num_patches_height, num_patches_width))
  ```

- Use [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to speedup inference.

  ```py
  import torch
  from transformers import AutoImageProcessor, AutoModel
  from PIL import Image
  import requests

  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  image = Image.open(requests.get(url, stream=True).raw)

  processor = AutoImageProcessor.from_pretrained('facebook/pixio-vith16')
  model = AutoModel.from_pretrained('facebook/pixio-vith16')

  compiled_model = torch.compile(model)

  inputs = processor(images=image, return_tensors="pt")
  outputs = compiled_model(**inputs)
  last_hidden_states = outputs.last_hidden_state
  ```

## PixioConfig

[[autodoc]] PixioConfig

## PixioModel

[[autodoc]] PixioModel
    - forward

## PixioBackbone

[[autodoc]] PixioBackbone
    - forward
