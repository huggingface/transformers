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

# SigLIP

## Overview

The SigLIP model was proposed in [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) by Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. SigLIP proposes to replace the loss function used in [CLIP](clip) by a simple pairwise sigmoid loss. This results in better performance in terms of zero-shot classification accuracy on ImageNet.

The abstract from the paper is the following:

*We propose a simple pairwise Sigmoid loss for Language-Image Pre-training (SigLIP). Unlike standard contrastive learning with softmax normalization, the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. The sigmoid loss simultaneously allows further scaling up the batch size, while also performing better at smaller batch sizes. Combined with Locked-image Tuning, with only four TPUv4 chips, we train a SigLiT model that achieves 84.5% ImageNet zero-shot accuracy in two days. The disentanglement of the batch size from the loss further allows us to study the impact of examples vs pairs and negative to positive ratio. Finally, we push the batch size to the extreme, up to one million, and find that the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient.*

## Usage tips

- Usage of SigLIP is similar to [CLIP](clip). The main difference is the training loss, which does not require a global view of all the pairwise similarities of images and texts within a batch. One needs to apply the sigmoid activation function to the logits, rather than the softmax.
- Training is not yet supported. If you want to fine-tune SigLIP or train from scratch, refer to the loss function from [OpenCLIP](https://github.com/mlfoundations/open_clip/blob/73ad04ae7fb93ede1c02dc9040a828634cb1edf1/src/open_clip/loss.py#L307), which leverages various `torch.distributed` utilities.
- When using the standalone [`SiglipTokenizer`] or [`SiglipProcessor`], make sure to pass `padding="max_length"` as that's how the model was trained.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/siglip_table.jpeg"
alt="drawing" width="600"/>

<small> SigLIP evaluation results compared to CLIP. Taken from the <a href="https://arxiv.org/abs/2303.15343">original paper</a>.</small>

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/google-research/big_vision/tree/main).

## Usage example

There are 2 main ways to use SigLIP: either using the pipeline API, which abstracts away all the complexity for you, or by using the `SiglipModel` class yourself.

### Pipeline API

The pipeline allows to use the model in a few lines of code:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> # load pipe
>>> image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")

>>> # load image
>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # inference
>>> outputs = image_classifier(image, candidate_labels=["2 cats", "a plane", "a remote"])
>>> outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
>>> print(outputs)
[{'score': 0.1979, 'label': '2 cats'}, {'score': 0.0, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
```

### Using the model yourself

If you want to do the pre- and postprocessing yourself, here's how to do that:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
>>> # important: we pass `padding=max_length` since the model was trained with this
>>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
>>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
31.9% that image 0 is 'a photo of 2 cats'
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SigLIP.

- [Zero-shot image classification task guide](../tasks/zero_shot_image_classification_md)
- Demo notebooks for SigLIP can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SigLIP). 🌎

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## SiglipConfig

[[autodoc]] SiglipConfig
    - from_text_vision_configs

## SiglipTextConfig

[[autodoc]] SiglipTextConfig

## SiglipVisionConfig

[[autodoc]] SiglipVisionConfig

## SiglipTokenizer

[[autodoc]] SiglipTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SiglipImageProcessor

[[autodoc]] SiglipImageProcessor
    - preprocess

## SiglipProcessor

[[autodoc]] SiglipProcessor

## SiglipModel

[[autodoc]] SiglipModel
    - forward
    - get_text_features
    - get_image_features

## SiglipTextModel

[[autodoc]] SiglipTextModel
    - forward

## SiglipVisionModel

[[autodoc]] SiglipVisionModel
    - forward


## SiglipForImageClassification

[[autodoc]] SiglipForImageClassification
    - forward