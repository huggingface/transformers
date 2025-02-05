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
generation. We build ANOLE from Meta AI’s Anole, adopting an innovative fine-tuning strategy that
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

Anole is a gated model so make sure to have access and login to Hugging Face Hub using a token.
Here's how to load the model and perform inference in half-precision (`torch.bfloat16`):

```python
from transformers import AnoleProcessor, AnoleForConditionalGeneration
import torch
from PIL import Image
import requests

processor = AnoleProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = AnoleForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# prepare image and text prompt
image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, suppress_tokens=model.vocabulary_mapping.image_tokens, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Multi image inference

Anole can perform inference with multiple images as input, where images either belong to the same prompt or different prompts (in batched inference). Here is how you can do it:

```python
from transformers import AnoleProcessor, AnoleForConditionalGeneration
import torch
from PIL import Image
import requests

processor = AnoleProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = AnoleForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Get three different images
image_stop = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)
image_cats = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
image_snowman = Image.open(requests.get("https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True).raw)

# Prepare a batched prompt, where the first one is a multi-image prompt and the second is not
prompts = [
    "What do these images have in common?<image><image>",
    "<image>What is shown in this image?"
]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(
    images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt"
).to(device="cuda:0", dtype=torch.bfloat16)

# Generate
generate_ids = model.generate(**inputs, suppress_tokens=model.vocabulary_mapping.image_tokens, max_new_tokens=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

### Text to image generation

Anole can also generate images. Here is how you can do it:

```python
import torch
from transformers import AnoleProcessor, AnoleForConditionalGeneration

processor = AnoleProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = AnoleForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

inputs = processor("Generate an image of a snowman.", return_tensors="pt").to(model.device)


# NOTE: We need to set `max_new_tokens` to at least 1026 since the model generates the `image_start_token` marker
# then 1024 image tokens, and finally the `image_end_token` marker.
generation_max_length = 1500
input_sequence_length = inputs.input_ids.shape[1]
visual_tokens = model.vocabulary_mapping.image_tokens


# prepare a constraint to suppress all tokens expcet for image tokens
def prefix_allowed_tokens_fn_image_only(batch_id, input_ids):
    boi_token_id = torch.tensor([processor.tokenizer.boi_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
    max_length = generation_max_length
    image_seq_length = 1024
    begin_index = input_sequence_length

    # If one image generation is completed, force end-of-image token
    if boi_token_id in input_ids:
        position = torch.nonzero(input_ids == boi_token_id, as_tuple=True)[0][-1]
        offset = input_ids.shape[0] - position
        if offset == image_seq_length + 1:
            return eoi_token_id

    # If we just started generating an new image, then force begin-of-image token
    if (input_ids.shape[0] - begin_index) % 1026 == 0:
        # Check if we can start generating image (i.e. enough space to fit one image of 1024 tokens)
        # otherwise generate EOS and stop 
        if max_length - input_ids.shape[0] < image_seq_length + 1:
            return eos_token_id
        return boi_token_id

    # Otherwise just generate any image token id
    return visual_tokens


# Generate discrete image tokens
generate_ids = model.generate(
    **inputs,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_image_only,
    max_new_tokens=generation_max_length,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

# Decode the generated image token. First crop boi/eos/eos tokens because decoder accepts only image tokens
# of shape `(bs, 1024)`
image_token_ids = response_ids[:, 1:-2]

pixel_values = model.decode_image_tokens(image_token_ids)
pixel_values = processor.postprocess(pixel_values.float(), return_tensors="pil")["pixel_values"]

# Save the image
pixel_values[0].save("snowman.png")
```

### Text-image to image generation

We can also interleave text and images in the prompt to generate images. Here is how you can do it:

```python

# Get image of a snowman
image_snowman = Image.open(requests.get("https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True).raw)

# Preprocess the prompt
inputs = processor(
    images=[image_snowman],
    text="Generate a variation of this image.<image>",
    padding=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

# Generate discrete image tokens
generate_ids = model.generate(
    **inputs,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_image_only_,
    max_new_tokens=generation_max_length,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

# Decode the generated image token. First crop boi/eos/eos tokens because decoder accepts only image tokens
# of shape `(bs, 1024)`
image_token_ids = response_ids[:, 1:-2]

pixel_values = model.decode_image_tokens(image_token_ids)
pixel_values = processor.postprocess(pixel_values.float(), return_tensors="pil")["pixel_values"]

# Save the image
pixel_values[0].save("snowman.png")
```

### Interleaved text-image generation

We can also generate interleaved text and images in the output. Here is how you can do it:

```python
import torch
from transformers import AnoleProcessor, AnoleForConditionalGeneration

processor = AnoleProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = AnoleForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Preprocess the prompt
inputs = processor("Can you draw a snowman and explain how to build one?", return_tensors="pt").to(model.device)


# NOTE: We need to set `max_new_tokens` to at least 1026 since the model generates the `image_start_token` marker
# then 1024 image tokens, and finally the `image_end_token` marker.
generation_max_length = 1500
input_sequence_length = inputs.input_ids.shape[1]
visual_tokens = model.vocabulary_mapping.image_tokens
text_tokens = [token_id for token_id in range(model.vocab_size) if token_id not in visual_tokens + image_special_tokens]

# Prepare a constraint to suppress all tokens expcet for image tokens
def prefix_allowed_tokens_fn_interleaved(batch_id, input_ids):
    boi_token_id = torch.tensor([processor.tokenizer.boi_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    max_length = generation_max_length
    image_seq_length = 1024

    # Check if we are in image generation mode
    if boi_token_id in input_ids: 
        position = torch.nonzero(input_ids == boi_token_id, as_tuple=True)[0][-1]
        offset = input_ids.shape[0] - position
        if offset < image_seq_length + 1:
            return visual_tokens
        elif offset == image_seq_length + 1:
            return eoi_token_id

    # Check if we can start generating image (i.e. have enough space to fit one image of 1024 tokens)
    if max_length - input_ids.shape[0] < image_seq_length + 1:
        return text_tokens
    
    # Else allow text tokens and possible to start generating an image
    return text_tokens + [boi_token_id]


# Generate interleaved text and discrete image tokens
generate_ids = model.generate(
    **inputs,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_interleaved,
    max_new_tokens=generation_max_length,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,  
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
batch_indices, token_indices = torch.where(response_ids == processor.tokenizer.boi_token_id)
batch_image_token_ids = []
for i, (batch_id, token_id) in enumerate(zip(batch_indices, token_indices)):
    image_token_ids = response_ids[batch_id + 1, token_id: token_id + 1025]
    pixel_values = model.decode_image_tokens(image_token_ids)
    pixel_values = processor.postprocess(pixel_values.float(), return_tensors="pil")["pixel_values"]
    pixel_values[0].save(f"fig_{i}.png")

text = processor.batch_decode(response_ids, skip_special_token=True)
print(text)
```

## AnoleProcessor

[[autodoc]] AnoleProcessor

## AnoleImageProcessor

[[autodoc]] AnoleImageProcessor

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
