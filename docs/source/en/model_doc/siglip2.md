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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
            <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SigLIP2

## Overview

[SigLIP2](https://huggingface.co/papers/2502.14786) is a family of multilingual vision-language encoders that builds on the [SigLIP](./siglip) training recipe. It includes decoder-based pretraining, self-distillation, and masked prediction to improve dense prediction tasks (segmentation, depth estimation, etc.). This model is available in two variants:

- NaFlex supports different resolutions and maintains the native image aspect ratio
- FixRes supports fixed resolutions and is backwards compatible with [SigLIP](./siglip)


You can find all the original SigLIP2 checkpoints under the [SigLIP2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) collection.

> [!TIP]
> Click on the SigLIP2 models in the right sidebar for more examples of how to apply SigLIP2 to different image and text tasks.

The example below demonstrates zero-shot classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

pipeline = pipeline(task="zero-shot-image-classification", model="google/siglip2-base-patch16-224", device=0, dtype=torch.bfloat16)
pipeline(image, candidate_labels=candidate_labels)
```

</hfoption>
<hfoption id="AutoModel (FixRes)">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

# follows the pipeline prompt template to get same results
texts = [f'This is a photo of {label}.' for label in candidate_labels]

# IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

</hfoption>
<hfoption id="AutoModel (NaFlex)">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]

# default value for `max_num_patches` is 256, but you can increase resulted image resolution providing higher values e.g. `max_num_patches=512`
inputs = processor(text=texts, images=image, padding="max_length", max_num_patches=256, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModel.from_pretrained("google/siglip2-large-patch16-512", quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

# follows the pipeline prompt template to get same results
texts = [f'This is a photo of {label}.' for label in candidate_labels]

# IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

## Notes

- Training is supported for DDP and FSDP on single-node multi-GPU setups. However, it does not use [torch.distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) utilities which may limit the scalability of batch size.
- When using the standalone [`GemmaTokenizerFast`] make sure to pass `padding="max_length"` and `max_length=64` as that's how the model was trained.
- Model was trained with *lowercased* text, so make sure your text labels are preprocessed the same way.
- To get the same results as the [`Pipeline`], a prompt template of `"This is a photo of {label}."` should be passed to the processor.
- The NaFlex variant processes different types of images at the appropriate resolution (using a larger resolution to process document images for example), while also minimizing the impact of aspect ratio distortion for certain inference tasks like OCR.

   NaFlex resizes the input image so the height and width are multiples of the patch size after resizing. It keeps the aspect ratio distortion as low as possible and produces a sequence length of at most the desired target sequence length (`max_num_patches`). After resizing, the image is split into a sequence of patches and a mask with padding information is added.
- Toggle the `attn_implementation` parameter to either `"sdpa"` or `"flash_attention_2"` to use a more memory-efficient attention.
    ```py
    # pip install -U flash-attn --no-build-isolation

    from transformers import SiglipModel

    model = SiglipModel.from_pretrained(
        "google/siglip2-so400m-patch14-384",
        attn_implementation="flash_attention_2",
        dtype=torch.float16,
        device_map=device,
    )
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
