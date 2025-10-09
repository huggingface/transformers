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
*This model was released on 2023-06-21 and added to Hugging Face Transformers on 2023-08-18 and contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# IDEFICS

[IDEFICS](https://huggingface.co/papers/2306.16527) trains a large multimodal model using the OBELICS dataset, which consists of 141 million web pages, 353 million images, and 115 billion text tokens extracted from Common Crawl. The dataset includes comprehensive filtering rules and is released openly. Training an 80 billion parameter vision and language model on OBELICS yields competitive results on multimodal benchmarks.

<hfoptions id="usage">
<hfoption id="IdeficsForVisionText2Text">

```py
import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text

model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b", dtype="auto")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

dogs_image_url_1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"
dogs_image_url_2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image2.jpeg"

prompts = [
    [
        "User:",
        dogs_image_url_1,
        "Describe this image.\nAssistant: An image of two dogs.\n",
        "User:",
        dogs_image_url_2,
        "Describe this image.\nAssistant:",
    ]
]
inputs = processor(prompts, return_tensors="pt")
generate_ids = model.generate(**inputs, max_new_tokens=6)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```

</hfoption>
</hfoptions>

## IdeficsConfig

[[autodoc]] IdeficsConfig

## IdeficsModel

[[autodoc]] IdeficsModel
    - forward

## IdeficsForVisionText2Text

[[autodoc]] IdeficsForVisionText2Text
    - forward

## IdeficsImageProcessor

[[autodoc]] IdeficsImageProcessor
    - preprocess

## IdeficsProcessor

[[autodoc]] IdeficsProcessor
    - __call__

