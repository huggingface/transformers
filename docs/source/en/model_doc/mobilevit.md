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

[MobileViT](https://huggingface.co/papers/2110.02178) proposed by Sachin Mehta and Mohammad Rastegari aims to combine best of both worlds, the Convolutional Neural Networks ([CNNs]( https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/introduction)) and VisionTransformers ([ViTs](https://huggingface.co/docs/transformers/en/model_doc/vit)), to make a model which is both powerful and lightweight in the sense that it can be used for mobile vision tasks.

So, the CNNs are very good at learning the local patterns such as edges, textures, etc. While the transformers are great at seeing the larger picture, trying to understand the relation of different parts of an image no matter how far they are. MobileViT block combines both these ideas, the core global representation is learnt  with transformers as convolution. It leverages the same building block as convolutions but replacing the local processing with deeper global processing. Since the Transformer mimics the convolution, we still get the image-specific knowledge that the CNNs are good at.

**What makes MobileViT lightweight?** 
Most of the models that uses Transformers for vision would first divide the images into several patches which are further flattened and converted into vectors. This causes in the loss of structural properties of an image, which isn't the case for CNNs. Now, this causes the Transformer models to go bigger and deeper to learn visual representations. 
But MobileViT uses both convolutions and transformers in a way that the resultant block has convolution-like properties while simultaneously allowing for global interactions. This allows us to design a more shallow and narrow models, which are light-weight.

![enter image description here](https://user-images.githubusercontent.com/67839539/136470152-2573529e-1a24-4494-821d-70eb4647a51d.png)


You can find all the original MobileViT checkpoints under the [Apple](https://huggingface.co/apple/models?search=mobilevit) organization.


> [!TIP]
> - This model was contributed by [Matthijs](https://huggingface.co/Matthijs).
> - Use [`MobileViTImageProcessor`](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to preprocess images.
> - If using custom preprocessing, ensure that images are in **BGR** format (not RGB), as expected by the pretrained weights.
> - The **classification models** are pretrained on [**ImageNet-1k**](https://huggingface.co/datasets/imagenet-1k) (ILSVRC 2012).
> - The **segmentation models** use a [**DeepLabV3**](https://huggingface.co/papers/1706.05587) head and are pretrained on [**PASCAL VOC**](http://host.robots.ox.ac.uk/pascal/VOC/).
> - TensorFlow versions are compatible with **TensorFlow Lite**, making them ideal for edge/mobile deployment.


Click on the MobileViT models in the right sidebar for more examples of how to apply Model name to different vision tasks.

The example below demonstrates how to  [Convert Mobile ViT image classification checkpoint to a TensorFlow Lite Model] using the  [`TFMobileViTForImageClassification`] class:

<hfoptions id = "usage">
<hfoption id="AutoModel">

```py

from transformers import TFMobileViTForImageClassification
import tensorflow as tf

  
model_ckpt = "apple/mobilevit-xx-small"
model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

  

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
								tf.lite.OpsSet.TFLITE_BUILTINS,
								tf.lite.OpsSet.SELECT_TF_OPS,
								]

tflite_model = converter.convert()
tflite_filename = model_ckpt.split("/")[-1] + ".tflite"

with  open(tflite_filename, "wb") as f:
f.write(tflite_model)

```

</hfoption>
</hfoptions>




## Notes

- **MobileViT** was designed to be performant and efficient on **mobile devices**.
- It combines the **inductive biases of CNNs** with the **global context modelling of Transformers**.
- Does **not** operate on sequential data, it's purely designed for image tasks.
- Feature maps are used directly instead of token embeddings.




## Resources
- Primary Source: [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer] (https://arxiv.org/pdf/2110.02178).
- Apple machine learning research: [PyTorch Implementation](https://github.com/apple/ml-cvnets).
- You can follow [this keras tutorial](https://keras.io/examples/vision/mobilevit) for a lightweight introduction.
- See also: [Image classification task guide](../tasks/image_classification).

  
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