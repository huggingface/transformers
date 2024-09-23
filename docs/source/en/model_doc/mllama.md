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

# Mllama

## Overview

The mllama model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*


This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## Usage Tips

- For text-only generation use `MllamaForCausalLM` to avoid loading vision tower.
- For text-only or image+text cases use `MllamaForConditionalGeneration`.
- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images across samples and to a maximum number of tiles within each image.
- The text passed to the processor should have the `"<|image|>"` tokens where the images should be inserted.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as text to the processor.

## Usage Example

```python

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<|image|>If I had to write a haiku for this one"
url = "https://llava-vl.github.io/static/images/view.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=25)
print(processor.decode(output[0], skip_special_tokens=True))
```


## MllamaConfig

[[autodoc]] MllamaConfig

## MllamaProcessor

[[autodoc]] MllamaProcessor


## MllamaImageProcessor

[[autodoc]] MllamaImageProcessor

## MllamaForConditionalGeneration

[[autodoc]] MllamaForConditionalGeneration
    - forward

## MllamaForCausalLM

[[autodoc]] MllamaForCausalLM
    - forward

## MllamaTextModel

[[autodoc]] MllamaTextModel
    - forward

## MllamaForCausalLM

[[autodoc]] MllamaForCausalLM
    - forward

## MllamaVisionModel

[[autodoc]] MllamaVisionModel
    - forward
