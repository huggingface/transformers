<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# Convolutional Vision Transformer (CvT)

Convolutional Vision Transformer (CvT) is a model that combines the strengths of convolutional neural networks (CNNs) and Vision transformers for the computer vision tasks. It introduces convolutional layers into the vision transformer architecture, allowing it to capture local patterns in images while maintaining the global context provided by self-attention mechanisms.

You can find all the CvT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=cvt) organization.

> [!TIP]
> This model was contributed by [anujunj](https://huggingface.co/anugunj).
> 
> Click on the CvT models in the right sidebar for more examples of how to apply CvT to different computer vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/cvt-13",
    dtype=torch.float16,
    device=0 
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/cvt-13",
    dtype=torch.float16,
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to("cuda")

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

</hfoption>
</hfoptions>

## Resources

Refer to this set of ViT [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) for examples of inference and fine-tuning on custom datasets. Replace [`ViTFeatureExtractor`] and [`ViTForImageClassification`] in these notebooks with [`AutoImageProcessor`] and [`CvtForImageClassification`].

## CvtConfig

[[autodoc]] CvtConfig

<frameworkcontent>
<pt>

## CvtModel

[[autodoc]] CvtModel
    - forward

## CvtForImageClassification

[[autodoc]] CvtForImageClassification
    - forward

</pt>
<tf>

## TFCvtModel

[[autodoc]] TFCvtModel
    - call

## TFCvtForImageClassification

[[autodoc]] TFCvtForImageClassification
    - call

</tf>
</frameworkcontent>
