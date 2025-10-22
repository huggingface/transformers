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
*This model was released on 2024-04-05 and added to Hugging Face Transformers on 2025-10-21.* 

# LW-DETR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The LW-Detr model was proposed
in [LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection](https://huggingface.co/papers/2407.17140) by
Qiang Chen, Xiangbo Su, Xinyu Zhang, Jian Wang, Jiahui Chen, Yunpeng Shen, Chuchu Han, Ziliang Chen, Weixiang Xu,
Fanrong Li, Shan Zhang, Kun Yao, Errui Ding, Gang Zhang and Jingdong Wang.

LW-DETR (Light-weight Detection Transformer) is a real-time object detector designed to outperform existing YOLO-based
models. Its architecture is a simple composition of a Vision Transformer (ViT) encoder, a projector, and a shallow DETR
decoder. The model's effectiveness comes from integrating advanced techniques such as pretraining on large datasets, an
improved IoU-aware loss function, and an efficient ViT encoder that uses interleaved window and global attentions to
reduce computational complexity. The authors also introduce a window-major feature map organization to improve the
efficiency of attention computations.

The abstract from the paper is the following:

*In this paper, we present a light-weight detection transformer, LW-DETR, which outperforms YOLOs for real-time object
detection. The architecture is a simple stack of a ViT encoder, a projector, and a shallow DETR decoder. Our approach
leverages recent advanced techniques, such as training-effective techniques, e.g., improved loss and pretraining, and
interleaved window and global attentions for reducing the ViT encoder complexity. We improve the ViT encoder by
aggregating multi-level feature maps, and the intermediate and final feature maps in the ViT encoder, forming richer
feature maps, and introduce window-major feature map organization for improving the efficiency of interleaved attention
computation. Experimental results demonstrate that the proposed approach is superior over existing real-time detectors,
e.g., YOLO and its variants, on COCO and other benchmark datasets.*

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/Atten4Vis/LW-DETR).

## Usage tips

This second version of RT-DETR improves how the decoder finds objects in an image.

- **Simple Architecture** – The model consists of a ViT encoder, a projector, and a shallow (3-layer) DETR decoder,
  making it straightforward to implement.
- **Efficient Inference** – To reduce the quadratic complexity of global self-attention in the ViT encoder, some global
  attention layers are replaced with window self-attention. Further speed-up is achieved through a window-major feature
  map organization that reduces costly memory permutation operations.
- **Effective Training** – The model benefits significantly from pretraining on the Objects365 dataset. It also uses
  an IoU-aware classification loss and adopts the Group DETR training scheme with multiple weight-sharing decoders to
  accelerate training.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import LwDetrForObjectDetection, LwDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = LwDetrImageProcessor.from_pretrained("stevenbucaille/lwdetr_small_60e_coco")
>>> model = LwDetrForObjectDetection.from_pretrained("stevenbucaille/lwdetr_small_60e_coco")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.97[341.14, 25.11, 639.98, 372.89]
cat: 0.96[12.78, 56.35, 317.67, 471.34]
remote: 0.95[39.96, 73.12, 175.65, 117.44]
sofa: 0.86[-0.11, 2.97, 639.89, 473.62]
sofa: 0.82[-0.12, 1.78, 639.87, 473.52]
remote: 0.79[333.65, 76.38, 370.69, 187.48]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LW Detr.

<PipelineTag pipeline="object-detection"/>

- Scripts for finetuning [`LwDetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## LwDetrConfig

[[autodoc]] LwDetrConfig

## LwDetrViTConfig

[[autodoc]] LwDetrViTConfig

## LwDetrModel

[[autodoc]] LwDetrModel
    - forward

## LwDetrForObjectDetection

[[autodoc]] LwDetrForObjectDetection
    - forward

## LwDetrViTBackbone

[[autodoc]] LwDetrViTBackbone
    - forward
