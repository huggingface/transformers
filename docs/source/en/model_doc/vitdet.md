<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2022-03-30 and added to Hugging Face Transformers on 2023-08-29 and contributed by [nielsr](https://huggingface.co/nielsr).*

# ViTDet

[ViTDet](https://huggingface.co/papers/2203.16527) explores the use of a plain Vision Transformer (ViT) as a backbone for object detection, eliminating the need for a hierarchical design. Minimal modifications are required for fine-tuning, and the model achieves competitive results using a simple feature pyramid and window attention. Pre-trained as Masked Autoencoders (MAE), ViTDet reaches up to 61.3 AP_box on the COCO dataset with only ImageNet-1K pre-training.

<hfoptions id="usage">
<hfoption id="VitDetModel">

```py
import torch
from transformers import VitDetConfig, VitDetModel

config = VitDetConfig()
model = VitDetModel(config)

pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values)

last_hidden_states = outputs.last_hidden_state
```

</hfoption>
</hfoptions>

## Usage tips

- Only the backbone is available.

## VitDetConfig

[[autodoc]] VitDetConfig

## VitDetModel

[[autodoc]] VitDetModel
    - forward

