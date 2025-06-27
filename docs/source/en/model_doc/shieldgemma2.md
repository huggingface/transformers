
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

# ShieldGemma 2

## Overview

The ShieldGemma 2 model was proposed in a [technical report](https://huggingface.co/papers/2504.01081) by Google. ShieldGemma 2, built on [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3), is a 4 billion (4B) parameter model that checks the safety of both synthetic and natural images against key categories to help you build robust datasets and models. With this addition to the Gemma family of models, researchers and developers can now easily minimize the risk of harmful content in their models across key areas of harm as defined below:

-   No Sexually Explicit content: The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).
-   No Dangerous Content: The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).
-   No Violence/Gore content: The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death).

We recommend using ShieldGemma 2 as an input filter to vision language models, or as an output filter of image generation systems. To train a robust image safety model, we curated training datasets of natural and synthetic images and instruction-tuned Gemma 3 to demonstrate strong performance.

This model was contributed by [Ryan Mullins](https://huggingface.co/RyanMullins).

## Usage Example

- ShieldGemma 2 provides a Processor that accepts a list of `images` and an optional list of `policies` as input, and constructs a batch of prompts as the product of these two lists using the provided chat template.
- You can extend ShieldGemma's built-in in policies with the `custom_policies` argument to the Processor. Using the same key as one of the built-in policies will overwrite that policy with your custom defintion.
- ShieldGemma 2 does not support the image cropping capabilities used by Gemma 3.

### Classification against Built-in Policies

```python
from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)
```

### Classification against Custom Policies

```python
from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

custom_policies = {
    "key_a": "descrition_a",
    "key_b": "descrition_b",
}

inputs = processor(
    images=[image],
    custom_policies=custom_policies,
    policies=["dangerous", "key_a", "key_b"],
    return_tensors="pt",
).to(model.device)

output = model(**inputs)
print(output.probabilities)
```


## ShieldGemma2Processor

[[autodoc]] ShieldGemma2Processor

## ShieldGemma2Config

[[autodoc]] ShieldGemma2Config

## ShieldGemma2ForImageClassification

[[autodoc]] ShieldGemma2ForImageClassification
    - forward
