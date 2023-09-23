<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AugViT

## Overview

TFAugViT model is the tensorflow implementation of the 
[AugViT: Augmented Shortcuts for Vision Transformers](https://arxiv.org/pdf/2106.15941v1.pdf) by Yehui Tang, Kai Han, Chang Xu, An Xiao, Yiping Deng, Chao Xu and Yunhe Wang, 
and first released in [this repository](https://github.com/kingcong/augvit).


## Model description

Aug-ViT inserts additional paths with learnable parameters in parallel on the original shortcuts for alleviating the feature collapse. The block-circulant projection is used to implement augmented shortcut, which brings negligible increase of computational cost.

## Intended uses & limitations

This model can be used for image classification tasks and easily be fine-tuned to suite your purpose of use.

### How to use

Here is how to use this model to classify an image into one of the 1,000 ImageNet classes:

```python
from transformers import TFAugViTForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = TFAugViTForImageClassification.from_pretrained("tensorgirl/TFaugvit")

outputs = model({'pixel_values':image})


# model predicts one of the 1000 ImageNet classes
predicted_class_idx = outputs.argmax(-1)
```

## Training data

The TFAugViT model is trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k), a dataset consisting of 1 million images and 1,000 classes. 

## Training procedure

Due to the use of einops library you cannot use the model,fit() directly on this model, you will have to either write a custom training loop by passing the inputs as shown above or you can wrap the model in a functional model of keras and specify the batch_size beforehand.
If you want to train the model on some other data then either resize the images to 224x224 or change the model config image_size to suit your requirements.


### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: Adam
- batch_size: 32
- training_precision: float32
- 
## Evaluation results

| Model            | ImageNet top-1 accuracy | # params  | Resolution |
|------------------|-------------------------|-----------|------------|
| Aug-ViT-S    | 81                   | 22.2 M     | 224x224 |
| Aug-ViT-B     | 82.4                    | 86.5 M     | 224x224|
| Aug-ViT-B (Upsampled)  | 84.2                | 86.5 M | 384x384|



### Framework versions

- Transformers 4.33.2
- TensorFlow 2.13.0
- Tokenizers 0.13.3

### BibTeX entry and citation info

```bibtex
@inproceedings{aug-vit tf,
title = {AugViT: Augmented Shortcuts for Vision Transformers},
author = {Yehui Tang, Kai Han, Chang Xu, An Xiao, Yiping Deng, Chao Xu and Yunhe Wang},
year = {2021},
URL = {https://arxiv.org/abs/2106.15941}
}
```