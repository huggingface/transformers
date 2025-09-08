<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2025-08-13 and added to Hugging Face Transformers on 2025-08-14.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# DINOv3

[DINOv3](https://huggingface.co/papers/2508.10104) is a family of versatile vision foundation models that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models.

You can find all the original DINOv3 checkpoints under the [DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) collection.

> [!TIP]
> Click on the DINOv3 models in the right sidebar for more examples of how to apply DINOv3 to different vision tasks.

The example below demonstrates how to obtain an image embedding with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.bfloat16,
)

pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from torchao.quantization import Int4WeightOnlyConfig
from transformers.image_utils import load_image


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitsplus-pretrain-lvd1689m")

quant_type = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_type)

model = AutoModel.from_pretrained(
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
```

## Notes

- The example below shows how to split the output tensor into:
  - one embedding for the whole image, commonly referred to as a `CLS` token,
    useful for classification and retrieval
  - register tokens - learnable embeddings that act as dedicated “memory slots” for global information,
    they reduce high-norm artifacts in patch tokens, yielding cleaner attention maps and better
    performance on dense prediction tasks.
  - a set of local embeddings, one for each `16x16` patch of the input image,
    useful for dense tasks, such as semantic segmentation

  ```py
  import torch
  from transformers import AutoImageProcessor, AutoModel
  from transformers.image_utils import load_image

  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = load_image(url)
  print("Image size:", image.height, image.width)  # [480, 640]

  processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  patch_size = model.config.patch_size
  print("Patch size:", patch_size) # 16
  print("Num register tokens:", model.config.num_register_tokens) # 4

  inputs = processor(images=image, return_tensors="pt")
  print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

  batch_size, _, img_height, img_width = inputs.pixel_values.shape
  num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
  num_patches_flat = num_patches_height * num_patches_width

  with torch.inference_mode():
    outputs = model(**inputs)

  last_hidden_states = outputs.last_hidden_state
  print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
  assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

  cls_token = last_hidden_states[:, 0, :]
  patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
  patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
  ```

## DINOv3ViTConfig

[[autodoc]] DINOv3ViTConfig

## DINOv3ConvNextConfig

[[autodoc]] DINOv3ConvNextConfig

## DINOv3ViTModel

[[autodoc]] DINOv3ViTModel
    - forward

## DINOv3ConvNextModel

[[autodoc]] DINOv3ConvNextModel
    - forward

## DINOv3ViTImageProcessorFast

[[autodoc]] DINOv3ViTImageProcessorFast
    - preprocess
