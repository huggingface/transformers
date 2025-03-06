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

# SigLIP2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The SigLIP2 model was proposed in [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://huggingface.co/papers/2502.14786) by Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin,
Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen,
Andreas Steiner and Xiaohua Zhai.

The model comes in two variants

 1) FixRes - model works with fixed resolution images (backward compatible with SigLIP v1)
 2) NaFlex - model works with variable image aspect ratios and resolutions (SigLIP2 in `transformers`)

The abstract from the paper is the following:

*We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success
of the original SigLIP. In this second iteration, we extend the original image-text training objective with
several prior, independently developed techniques into a unified recipe—this includes decoder-based
pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With
these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, 
including zero-shot classification (best SigLIP 2 ViT-g/16 achieves 85.0% ImageNet zero-shot
accuracy), image-text retrieval, and transfer performance when extracting visual representations for
Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements 
on localization and dense prediction tasks. We also train variants which support multiple resolutions 
and preserve the input’s native aspect ratio. Finally, we train on a more diverse data-mixture that
includes de-biasing techniques, leading to much better multilingual understanding and improved fair-
ness. To provide users with the ability to trade-off inference cost with performance, we release model
checkpoints at four sizes (ViT-B/86M, L/303M, So400m/400M, and g/1B).*

## Usage tips

- Usage of SigLIP2 is similar to [SigLIP](siglip) and [CLIP](clip). The main difference from CLIP is the training loss, which does not require a global view of all the pairwise similarities of images and texts within a batch. One needs to apply the sigmoid activation function to the logits, rather than the softmax.
- Training is supported but does not use `torch.distributed` utilities which may limit the scalability of batch size. However, DDP and FDSP works on single-node multi-gpu setup.
- When using the standalone [`GemmaTokenizerFast`] make sure to pass `padding="max_length"` and `max_length=64` as that's how the model was trained.
- Model was trained with *lowercased* text, make sure you make the same preprocessing for your text labels.
- To get the same results as the pipeline, a prompt template of "this is a photo of {label}" should be used.
- The NaFlex variant supports processing images at higher resolutions by adjusting the `max_num_patches` parameter in the `Processor`. The default value is `max_num_patches=256`. Increasing `max_num_patches` to 1024 (4x) will approximately double processed image height and width, while preserving the aspect ratio.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/siglip2_metrics_table.png"
alt="drawing" width="600"/>

This model was contributed by [qubvel](https://huggingface.co/qubvel-hf).
The original code can be found [here](https://github.com/google-research/big_vision/tree/main).

## Usage example

There are 2 main ways to use SigLIP2: either using the pipeline API, which abstracts away all the complexity for you, or by using the `Siglip2Model` class yourself.

### FixRes variant

**Pipeline API**

The pipeline allows to use the model in a few lines of code:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> # load pipe
>>> image_classifier = pipeline(
...     task="zero-shot-image-classification",
...     model="google/siglip2-base-patch16-224",
... )

>>> # load image
>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # inference
>>> candidate_labels = ["2 cats", "a plane", "a remote"]
>>> outputs = image_classifier(image, candidate_labels=candidate_labels)
>>> outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
>>> print(outputs)
[{'score': 0.1499, 'label': '2 cats'}, {'score': 0.0008, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
```

**Using the model yourself**

If you want to do the pre- and postprocessing yourself, here's how to do that:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> candidate_labels = ["2 cats", "2 dogs"]
# follows the pipeline prompt template to get same results
>>> texts = [f"This is a photo of {label}." for label in candidate_labels]

# IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
>>> inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
>>> print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
15.0% that image 0 is '2 cats'
```

### NaFlex variant

NaFlex combines ideas from FlexiViT, i.e. supporting multiple, predefined sequence lengths 
with a single ViT model, and NaViT, namely processing images at their native aspect ratio.
This enables processing different types of images at appropriate resolution, e.g. using a
larger resolution to process document images, while at the same time minimizing the impact 
of aspect ratio distortion on certain inference tasks, e.g. on OCR.

Given a patch size and target sequence length, NaFlex preprocesses the data by first resizing 
the input image such that the height and width after resizing are multiples of the patch size,
while 
    
    1. keeping the aspect ratio distortion as small as possible
    2. producing a sequence length of at most the desired target sequence length (`max_num_patches`)
    
The resulting distortion in width and height is at most `(patch_size - 1) / width` and
`(patch_size - 1) / height`, respectively, which tends to be small for common resolutions and aspect ratios. 
After resizing, the image is split into a sequence of patches, and a mask with padding information is added.

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex")
>>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> candidate_labels = ["2 cats", "2 dogs"]
# follows the pipeline prompt template to get same results
>>> texts = [f"This is a photo of {label}." for label in candidate_labels]

# default value for `max_num_patches` is 256, but you can increase resulted image resolution providing
# higher values e.g. `max_num_patches=512`
>>> inputs = processor(text=texts, images=image, max_num_patches=256, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
>>> print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
21.1% that image 0 is '2 cats'
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SigLIP2.

- [Zero-shot image classification task guide](../tasks/zero_shot_image_classification)
- Demo notebook for SigLIP2 can be found [here](https://github.com/qubvel/transformers-notebooks/tree/master/notebooks/SigLIP2_inference.ipynb). 🌎

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.


## Combining SigLIP2 and Flash Attention 2

First, make sure to install the latest version of Flash Attention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of flash-attn repository. Make also sure to load your model in half-precision (e.g. `torch.float16``)

To load and run a model using Flash Attention 2, refer to the snippet below:

```python
>>> import torch
>>> import requests
>>> from PIL import Image
>>> from transformers import AutoProcessor, AutoModel
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModel.from_pretrained(
...     "google/siglip2-so400m-patch14-384",
...     attn_implementation="flash_attention_2",
...     torch_dtype=torch.float16,
...     device_map=device,
... )
>>> processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> candidate_labels = ["2 cats", "2 dogs"]
# follows the pipeline prompt template to get same results
>>> texts = [f'This is a photo of {label}.' for label in candidate_labels]
# important: we pass `padding=max_length` since the model was trained with this
>>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     with torch.autocast(device):
...         outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
>>> print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
19.8% that image 0 is '2 cats'
```

## Siglip2Config

[[autodoc]] Siglip2Config

## Siglip2TextConfig

[[autodoc]] Siglip2TextConfig

## Siglip2VisionConfig

[[autodoc]] Siglip2VisionConfig

## Siglip2ImageProcessor

[[autodoc]] Siglip2ImageProcessor
    - preprocess

## Siglip2ImageProcessorFast

[[autodoc]] Siglip2ImageProcessorFast
    - preprocess

## Siglip2Processor

[[autodoc]] Siglip2Processor

## Siglip2Model

[[autodoc]] Siglip2Model
    - forward
    - get_text_features
    - get_image_features

## Siglip2TextModel

[[autodoc]] Siglip2TextModel
    - forward

## Siglip2VisionModel

[[autodoc]] Siglip2VisionModel
    - forward

## Siglip2ForImageClassification

[[autodoc]] Siglip2ForImageClassification
    - forward
