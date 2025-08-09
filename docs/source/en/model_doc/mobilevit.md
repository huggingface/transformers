<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with  the License. You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on  an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the  specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be  rendered properly in your Markdown viewer.

-->



# MobileViT


<div style="float: right;">
    <div class="flex flex-wrap space-x-2">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">

</div>

[MobileViT](https://huggingface.co/papers/2110.02178) is a lightweight vision transformer for mobile devices that merges CNNs's efficiency and inductive biases with transformers global context modeling. It treats transformers as convolutions, enabling global information processing without the heavy computational cost of standard ViTs.

![enter image description here](https://user-images.githubusercontent.com/67839539/136470152-2573529e-1a24-4494-821d-70eb4647a51d.png)

<div class="flex justify-center">
   <img>
</div>


You can find all the original MobileViT checkpoints under the [Apple](https://huggingface.co/apple/models?search=mobilevit) organization.


> [!TIP]
> - This model was contributed by [matthijs](https://huggingface.co/Matthijs) and the TensorFlow version was contributed by [sayakpaul](https://huggingface.co/sayakpaul).
>
> Click on the MobileViT models in the right sidebar for more examples of how to apply MobileViT to different vision tasks.


The example below demonstrates how to do [Image Classification] with [`Pipeline`] and the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python

from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")

#initialize an image classification pipeline
classifier = pipeline("image-classification", model = "apple/mobilevit-small")


#run inference on each image in dataset:
for i in dataset["test"]:
	preds = classifier(i["image"])
	print(f"Prediction: {preds}\n")
```

</hfoption>

<hfoption id="AutoModel">

```python

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, MobileViTForImageClassification


dataset = load_dataset("huggingface/cats-image")
img = dataset["test"]["image"][0]

# Load processor and model
processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained(
    								"apple/mobilevit-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Preprocess
inputs = processor(images = img, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

#Forward Pass
with torch.no_grad():
    outputs = model(**inputs)

#Get Predicted label
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])

```

</hfoption>
</hfoptions>




## Notes

- Does **not** operate on sequential data, it's purely designed for image tasks.
- Feature maps are used directly instead of token embeddings.
- Use [`MobileViTImageProcessor`](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to preprocess images.
- If using custom preprocessing, ensure that images are in **BGR** format (not RGB), as expected by the pretrained weights.
- The **classification models** are pretrained on [**ImageNet-1k**](https://huggingface.co/datasets/imagenet-1k) (ILSVRC 2012).
- The **segmentation models** use a [**DeepLabV3**](https://huggingface.co/papers/1706.05587) head and are pretrained on [**PASCAL VOC**](http://host.robots.ox.ac.uk/pascal/VOC/).
- TensorFlow versions are compatible with **TensorFlow Lite**, making them ideal for edge/mobile deployment.




  
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

<frameworkcontent>
<pt>

## MobileViTModel

[[autodoc]] MobileViTModel
    - forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification
    - forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation
    - forward

</pt>
<tf>

## TFMobileViTModel

[[autodoc]] TFMobileViTModel
    - call

## TFMobileViTForImageClassification

[[autodoc]] TFMobileViTForImageClassification
    - call

## TFMobileViTForSemanticSegmentation

[[autodoc]] TFMobileViTForSemanticSegmentation
    - call

</tf>
</frameworkcontent>
