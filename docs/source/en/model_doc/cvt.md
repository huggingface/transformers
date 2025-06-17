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

# TODO: add information about CvT in simple language like Vit
Convolutional Vision Transformer (CvT) is a model that combines the strengths of convolutional neural networks (CNNs) and Vision transformers for the computer vision tasks. It introduces convolutional layers into the vision transformer architecture, allowing it to capture local patterns in images while maintaining the global context provided by self-attention mechanisms.
You can find all the CvT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=cvt) organization.


> [!TIP]
> Click on the CvT models in the right sidebar (if available) or check model hub pages for more examples of how to apply CvT to different computer vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

# Replace "microsoft/cvt-13" with the specific CvT model you want to use
pipe = pipeline(
    task="image-classification",
    model="microsoft/cvt-13",
    torch_dtype=torch.float16,
    device=0 
)
output = pipe(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print(output)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Replace "microsoft/cvt-13" with the specific CvT model you want to use
model_name = "microsoft/cvt-13"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Using float16 for faster inference
    device_map="auto" # Automatically maps model to available device (GPU/CPU)
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

</hfoption>
</hfoptions>

## Notes

- The CvT models are regular Vision Transformers, but trained with convolutions. They outperform the original model (ViT) when fine-tuned on ImageNet-1K and CIFAR-100.
- The original ViT demo notebooks, such as those found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer), can often be adapted for CvT. You would typically replace ViT-specific classes like `ViTFeatureExtractor` with `AutoImageProcessor` and `ViTForImageClassification` with `CvtForImageClassification` or `AutoModelForImageClassification` using a CvT checkpoint.
- CvT checkpoints available on the Hugging Face Hub are often pre-trained on large-scale datasets like ImageNet-22k and may also be fine-tuned on datasets like ImageNet-1k.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CvT.

<PipelineTag pipeline="image-classification"/>

- [`CvtForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

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
