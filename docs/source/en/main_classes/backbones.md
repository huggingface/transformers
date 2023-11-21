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

# Backbones

Backbones are models used for feature extraction for computer vision tasks. One can use a model as backbone in two ways:

* initializing `AutoBackbone` class with a pretrained model,
* initializing a supported backbone configuration and passing it to the model architecture. 

## Using AutoBackbone 

You can use `AutoBackbone` class to initialize a model as a backbone and get the feature maps for any stage. You can define `out_indices` to indicate the index of the layers which you would like to get the feature maps from. You can also use `out_features` if you know the name of the layers. You can use them interchangeably. If you are using both `out_indices` and `out_features`, ensure they are consistent. Not passing any of the feature map arguments will make the backbone yield the feature maps of the last layer.

```py
>>> from transformers import AutoImageProcessor, AutoBackbone
>>> import torch
>>> from PIL import Image
>>> import requests

>>> processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(0,))
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> feature_maps = outputs.feature_maps
>>> list(feature_maps[-1].shape)
[1, 96, 56, 56]
```

## Initializing Backbone Configuration

You can initialize architectures with model backbones by passing a model configuration to `backbone_config` to the model neck configuration. For example, below you can see how to initialize the [MaskFormer](model_doc/maskformer.md) model with [ResNet](model_doc/resnet.md) backbone.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-50")
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```
You can also initialize a backbone with random weights to initialize the model neck with it. 

```py
backbone_config = ResNetConfig()
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

`timm` models are also supported in transformers through `TimmBackbone` and `TimmBackboneConfig`.

```python
from transformers import TimmBackboneConfig, TimmBackbone

backbone_config = TimmBackboneConfig("resnet50")
model = TimmBackbone(config=backbone_config)
```
