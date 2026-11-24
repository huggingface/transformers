<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-04-22 and contributed to Hugging Face Transformers on 2026-11-24.*

# Phi-3-vision

## Overview

Phi-3-vision is the multimodal member of the Phi-3 family by Microsoft, described in the
[Phi-3 Technical Report](https://huggingface.co/papers/2404.14219). It covers
[microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) and
[microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct). The
[TIGER-Lab/VLM2Vec-Full](https://huggingface.co/TIGER-Lab/VLM2Vec-Full) embedding model loads with the same classes.

## Architecture

Phi-3-vision combines a CLIP ViT-L/14 vision encoder at 336x336 pixels with a [Phi-3](./phi3) language model. An image
is resized to fit at most `num_crops` crops of 336x336, padded with white and split into crops, and a downscaled global
image is added. The features of the second to last vision layer are merged 2x2 into one token, a learned newline token
ends every row, and a learned separator sits between the crops and the global image. A two-layer MLP projects the
tokens into the language model.

## Usage

Put one `<|image|>` in the prompt per image, in the order of `images`, where the model cards use `<|image_1|>`, `<|image_2|>`, ...

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

model_id = "microsoft/Phi-3.5-vision-instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

image = load_image("https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg")
prompt = "<|user|>\n<|image|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=64, eos_token_id=processor.tokenizer.eos_token_id)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Phi3VConfig

[[autodoc]] Phi3VConfig

## Phi3VImageProcessor

[[autodoc]] Phi3VImageProcessor
    - preprocess

## Phi3VProcessor

[[autodoc]] Phi3VProcessor

## Phi3VModel

[[autodoc]] Phi3VModel
    - forward

## Phi3VForConditionalGeneration

[[autodoc]] Phi3VForConditionalGeneration
    - forward
