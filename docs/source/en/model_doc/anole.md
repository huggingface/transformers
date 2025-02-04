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

# Anole

## Overview

The Anole model was proposed in [ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation](https://arxiv.org/abs/2407.06135v1) by Ethan Chern, Jiadi Su, Yan Ma and Pengfei Liu.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*Previous open-source large multimodal models (LMMs) have faced several limitations: (1) they often lack
native integration, requiring adapters to align visual representations with pre-trained large language models
(LLMs); (2) many are restricted to single-modal generation; (3) while some support multimodal generation,
they rely on separate diffusion models for visual modeling and generation. To mitigate these limitations,
we present ANOLE, an open, autoregressive, native large multimodal model for interleaved image-text
generation. We build ANOLE from Meta AI’s Chameleon, adopting an innovative fine-tuning strategy that
is both data-efficient and parameter-efficient. ANOLE demonstrates high-quality, coherent multimodal
generation capabilities. We have open-sourced our model, training framework, and instruction tuning data.*


## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to set `processor.tokenizer.padding_side = "left"` before generating.

- When generating images, we advice users to load the model in `bfloat16` for better results. Simply make sure to set `torch_dtype=torch.bfloat16` when loading the model.

> [!NOTE]
> Note that Anole has a bias for "empty" or background patches, so it is recommended to use sampling when generating images (i.e. setting `do_sample=True` during generation) to reduce the likelihood of generating a blank image.


This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](https://github.com/GAIR-NLP/anole/tree/main).


## Usage example

### Single image inference

Chameleon is a gated model so make sure to have access and login to Hugging Face Hub using a token.
Here's how to load the model and perform inference in half-precision (`torch.bfloat16`):

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# prepare image and text prompt
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Multi image inference

Chameleon can perform inference with multiple images as input, where images either belong to the same prompt or different prompts (in batched inference). Here is how you can do it:

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# Get three different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batched prompt, where the first one is a multi-image prompt and the second is not
prompts = [
    "What do these images have in common?<image><image>",
    "<image>What is shown in this image?"
]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

### Text to image generation

Chameleon can also generate images. However, the official model checkpoint currently only supports text generation. We need to use finetuned versions such as [Anole](https://arxiv.org/abs/2407.06135) to do image generation. Here is how you can do it:

```python
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Prepare a prompt
prompt = "Generate an image of a snowman."

# Preprocess the prompt
inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

# Generate discrete image tokens
generate_ids = model.generate(
    **inputs,
    multimodal_generation_mode="image-only",
    # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
    max_new_tokens=1026,
    # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
    do_sample=True,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

# Decode the generated image tokens
pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
images = processor.postprocess_pixel_values(pixel_values)

# Save the image
images[0].save("snowman.png")
```

### Text-image to image generation

We can also interleave text and images in the prompt to generate images. Here is how you can do it:

```python
import requests

import torch
from PIL import Image
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from transformers.image_transforms import to_pil_image

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Get image of a snowman
url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a prompt
prompt = "Generate a variation of this image.<image>"

# Preprocess the prompt
inputs = processor(
    images=[image_snowman],
    text=prompt,
    padding=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

# Generate discrete image tokens
generate_ids = model.generate(
    **inputs,
    multimodal_generation_mode="image-only",
    # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
    do_sample=True,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

# The generated image tokens are wrapped by the `image_start_token` and `image_end_token` tokens. We need to remove them before decoding the image tokens.
image_token_ids = response_ids[:, 1:-1]

# Decode the generated image tokens
pixel_values = model.decode_image_tokens(image_token_ids)
pixel_values = processor.postprocess_pixel_values(pixel_values)

# Save the image
image = to_pil_image(pixel_values[0].detach().cpu())
image.save("snowman.png")
```

### Interleaved text-image generation

We can also generate interleaved text and images in the output. Here is how you can do it:

```python
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Prepare a prompt
prompt = "Can you draw a snowman and explain how to build one?"

# Preprocess the prompt
inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

# Generate interleaved text and discrete image tokens
generate_ids = model.generate(
    **inputs,
    multimodal_generation_mode="interleaved-text-image",
    # Note: We will need a larger `max_new_tokens` value since we are generating both text and image tokens.
    max_new_tokens=4096,
    # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
    do_sample=True,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
```

## AnoleConfig

[[autodoc]] AnoleConfig

## AnoleVQVAEConfig

[[autodoc]] AnoleVQVAEConfig

## AnoleVQVAE

[[autodoc]] AnoleVQVAE
    - forward

## AnoleModel

[[autodoc]] AnoleModel
    - forward

## AnoleForConditionalGeneration

[[autodoc]] AnoleForConditionalGeneration
    - forward
