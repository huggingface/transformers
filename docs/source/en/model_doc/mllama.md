<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# mllama

## Overview

The mllama model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*


This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## Usage Tips;

- For text-only generation use `MllamaForCausalGeneration` and for image+text cases use `MllamaForConditionalGeneration`.
- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images in a batch for input to the model.
- The text passed to the processor should have the <|image|> tokens where the images should be inserted
- The processor has its own apply_chat_template method to convert chat messages to text that can then be passed as text to the processor.

## Usage Example;

```python

import requests
from PIL import Image

import torch
from transformers import MllamaProcessor, MllamaForConditionalGeneration, StaticCache


url = "https://www.ilankelman.org/stopsigns/australia.jpg"
stop_image = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
snowman_image = Image.open(requests.get(url, stream=True).raw)

processor = MllamaProcessor.from_pretrained("s0409/model-3")
model = MllamaForConditionalGeneration.from_pretrained("s0409/model-3", torch_dtype=torch.bfloat16, device_map="auto")

texts = ["<|image|><|begin_of_text|>The image shows"]
inputs = processor(text=texts, images=[[stop_image]], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)

output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
generated_text = processor.batch_decode(output.sequences, skip_special_tokens=False)

```


## MllamaConfig

[[autodoc]] MllamaConfig

## MllamaProcessor

[[autodoc]] MllamaProcessor

## MllamaForConditionalGeneration

[[autodoc]] MllamaForConditionalGeneration
    - forward
