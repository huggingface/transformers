<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FastViT

## Overview

The FastViT model was proposed in [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189) by Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel and Anurag Ranjan.
FastViT is a hybrid Transformer with some several modifications, such as replacing denses with a factored version, 
replace self-attention to large kernel convolutions, with the objective of reducing latency.
The authors claims that FastViT is 3.5× faster than CMT, a recent state-of-the-art hybrid transformer architecture, 
4.9× faster than EfficientNet, and 1.9× faster than ConvNeXt on a mobile device for the same accuracy on the ImageNet dataset.   

The abstract from the paper is the following:

*The recent amalgamation of transformer and convolutional designs has led to steady improvements in accuracy and efficiency of the models. 
In this work, we introduce FastViT, a hybrid vision transformer architecture that obtains the state-of-the-art latency-accuracy trade-off. 
To this end, we introduce a novel token mixing operator, RepMixer, a building block of FastViT, that uses structural reparameterization 
to lower the memory access cost by removing skip-connections in the network. We further apply train-time overparametrization and 
large kernel convolutions to boost accuracy and empirically show that these choices have minimal effect on latency. 
We show that – our model is 3.5× faster than CMT, a recent state-of-the-art hybrid transformer architecture, 
4.9× faster than EfficientNet, and 1.9× faster than ConvNeXt on a mobile device for the same accuracy on the ImageNet dataset. 
At similar latency, our model obtains 4.2% better Top-1 accuracy on ImageNet than MobileOne. 
Our model consistently outperforms competing architectures across several tasks – image classification, detection, segmentation and 3D mesh regression 
with significant improvement in latency on both a mobile device and a desktop GPU. Furthermore, our model is highly robust to out-of-distribution samples 
and corruptions, improving over competing robust models.*

Tips:

- One can use the [`AutoImageProcessor`] class to prepare images for the model.
- When variable `inference` in [`FastViTConfig`] is set to True, batchnorms and residual connections are removed to speed up inference.


There are 3 ways to instantiate a FastViT model (depending on what you prefer):

Option 1: Instantiate FastViT with pre-trained weights for entire model
```py
>>> from transformers import FastViTForImageClassification

>>> model = FastViTForImageClassification.from_pretrained("JorgeAV/fastvit_t8")
```

Option 2: Instantiate FastViT with randomly initialized weights
```py
>>> from transformers import FastViTConfig, FastViTForImageClassification

>>> config = FastViTConfig()
>>> model = FastViTForImageClassification(config)
```

Option 3: Instantiate FastViT only for inference (faster) with randomly initialized weights
```py
>>> from transformers import FastViTConfig, FastViTForImageClassification

>>> config = FastViTConfig(inference=True)
>>> model = FastViTForImageClassification(config)
```

Complete example showcasing the use of FastViT model with pre-trained weights:

```py
>>> from transformers import AutoImageProcessor, FastViTModel
>>> from PIL import Image
>>> import requests

>>> ImageProcessor = AutoImageProcessor.from_pretrained("JorgeAV/fastvit_t8")
>>> model = FastViTModel.from_pretrained("JorgeAV/fastvit_t8")

>>> # load image
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)


>>> pixel_values = ImageProcessor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(pixel_values)

```
This model was contributed by [JorgeAV](https://huggingface.co/JorgeAV).
The original code can be found [here](https://github.com/apple/ml-fastvit).


## FastViTConfig

[[autodoc]] FastViTConfig

## FastViTModel

[[autodoc]] FastViTModel
    - forward

## FastViTForImageClassification

[[autodoc]] FastViTForImageClassification
    - forward
