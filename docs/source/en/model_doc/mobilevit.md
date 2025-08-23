<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with  the License. You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on  an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the  specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be  rendered properly in your Markdown viewer.

-->
*This model was released on 2021-10-05 and added to Hugging Face Transformers on 2022-06-29.*



# MobileViT


<div style="float: right;">
    <div class="flex flex-wrap space-x-2">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[MobileViT](https://huggingface.co/papers/2110.02178) is a lightweight vision transformer for mobile devices that merges CNNs's efficiency and inductive biases with transformers global context modeling. It treats transformers as convolutions, enabling global information processing without the heavy computational cost of standard ViTs.


<div class="flex justify-center">
   <img src = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/MobileViT.png">
</div>


You can find all the original MobileViT checkpoints under the [Apple](https://huggingface.co/apple/models?search=mobilevit) organization.


> [!TIP]
> - This model was contributed by [matthijs](https://huggingface.co/Matthijs).
>
> Click on the MobileViT models in the right sidebar for more examples of how to apply MobileViT to different vision tasks.


The example below demonstrates how to do [Image Classification] with [`Pipeline`] and the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python

import torch
from transformers import pipeline

classifier = pipeline(
   task="image-classification",
   model="apple/mobilevit-small",
   dtype=torch.float16, device=0,
)

preds = classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print(f"Prediction: {preds}\n")
```

</hfoption>

<hfoption id="AutoModel">

```python

import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, MobileViTForImageClassification

image_processor = AutoImageProcessor.from_pretrained(
   "apple/mobilevit-small",
   use_fast=True,
)
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small", device_map="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is:{predicted_class_label}")
```

</hfoption>
</hfoptions>


## Notes

- Does **not** operate on sequential data, it's purely designed for image tasks.
- Feature maps are used directly instead of token embeddings.
- Use [`MobileViTImageProcessor`] to preprocess images.
- If using custom preprocessing, ensure that images are in **BGR** format (not RGB), as expected by the pretrained weights.
- The classification models are pretrained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k).
- The segmentation models use a [DeepLabV3](https://huggingface.co/papers/1706.05587) head and are pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).



## MobileViTConfig

[[autodoc]] MobileViTConfig

## MobileViTFeatureExtractor

[[autodoc]] MobileViTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## MobileViTImageProcessor

[[autodoc]] MobileViTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileViTImageProcessorFast

[[autodoc]] MobileViTImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## MobileViTModel

[[autodoc]] MobileViTModel
    - forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification
    - forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation
    - forward
