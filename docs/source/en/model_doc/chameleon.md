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

# Chameleon

## Overview

The Chameleon model was proposed in [Chameleon: Mixed-Modal Early-Fusion Foundation Models
](https://arxiv.org/abs/2405.09818v1) by META AI Chameleon Team. Chameleon is a Vision-Language Model that use vector quantization to tokenize images which enables the model to generate multimodal output. The model takes images and texts as input, including an interleaved format, and generates textual response. Image generation module is not released yet. 


The abstract from the paper is the following:

*We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training
approach from inception, an alignment recipe, and an architectural parameterization tailored for the
early-fusion, token-based, mixed-modal setting. The models are evaluated on a comprehensive range
of tasks, including visual question answering, image captioning, text generation, image generation, and
long-form mixed modal generation. Chameleon demonstrates broad and general capabilities, including
state-of-the-art performance in image captioning tasks, outperforms Llama-2 in text-only tasks while
being competitive with models such as Mixtral 8x7B and Gemini-Pro, and performs non-trivial image
generation, all in a single model. It also matches or exceeds the performance of much larger models,
including Gemini Pro and GPT-4V, according to human judgments on a new long-form mixed-modal
generation evaluation, where either the prompt or outputs contain mixed sequences of both images and
text. Chameleon marks a significant step forward in unified modeling of full multimodal documents*


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/chameleon_arch.png"
alt="drawing" width="600"/>

<small> Chameleon incorporates a vector quantizer module to transform images into discrete tokens. That also enables image generation using an auto-regressive transformer. Taken from the <a href="https://arxiv.org/abs/2405.09818v1">original paper.</a> </small>

This model was contributed by [joaogante](https://huggingface.co/joaogante) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/facebookresearch/chameleon).


## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to set `processor.tokenizer.padding_side = "left"` before generating.

- When generating images, we advice users to load the model in `bfloat16` for better results. Simply make sure to set `torch_dtype=torch.bfloat16` when loading the model.

- Note that Chameleon was tuned for safety alignment. If the model is refusing to answer, consider asking a more concrete question, instead of an open question.

- Chameleon generates in chat format which means that the generated text will always be the "assistant's turn". You can enable a text completion generation by passing `return_for_text_completion=True` when calling the processor.

> [!NOTE]
> Chameleon implementation in Transformers uses a special image token to indicate where to merge image embeddings. For special image token we didn't add a new one but used one of the reserved tokens: `<reserved08707>`. You have to add `<image>` to your prompt in the place where the image should be embedded for correct generation.

> [!NOTE]
> The official model checkpoint currently only supports text generation. To generate images and interleaved text-image responses, you can use finetuned versions such as [Anole](https://arxiv.org/abs/2407.06135). Note however that Anole has a bias for "empty" or background patches, so it is recommended to use sampling when generating images (i.e. setting `do_sample=True` during generation) to reduce the likelihood of generating a blank image.

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

inputs = processor(prompt, image, return_tensors="pt").to(model.device)

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
inputs = processor(
    text=prompts,
    images=[image_stop, image_cats, image_snowman],
    padding=True,
    return_tensors="pt",
).to(device="cuda", dtype=torch.bfloat16)

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
    prompt,
    images=[image_snowman],
    padding=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

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

From here, you can split the response tokens into text and image token segments, decode them separately as shown in the previous examples, and finally render the resulting text and images together. You can also use [MMSG](https://github.com/leloykun/mmsg) to do this more easily.

## Model optimization

### Quantization using Bitsandbytes

The model can be loaded in 8 or 4 bits, greatly reducing the memory requirements while maintaining the performance of the original model. First make sure to install bitsandbytes, `pip install bitsandbytes` and make sure to have access to a CUDA compatible GPU device. Simply change the snippet above with:

```python
from transformers import ChameleonForConditionalGeneration, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", quantization_config=quantization_config, device_map="cuda")
```

### Use Flash-Attention 2 and SDPA to further speed-up generation

The models supports both, Flash-Attention 2 and PyTorch's [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) which can be enables for optimization. SDPA is the default options when you load the model, If you want to switch for Flash Attention 2, first make sure to install flash-attn. Refer to the [original repository](https://github.com/Dao-AILab/flash-attention) regarding that package installation. Simply change the snippet above with:

```python
from transformers import ChameleonForConditionalGeneration

model_id = "facebook/chameleon-7b"
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
).to(0)
```

## ChameleonConfig

[[autodoc]] ChameleonConfig

## ChameleonVQVAEConfig

[[autodoc]] ChameleonVQVAEConfig

## ChameleonProcessor

[[autodoc]] ChameleonProcessor

## ChameleonImageProcessor

[[autodoc]] ChameleonImageProcessor
    - preprocess

## ChameleonVQVAE

[[autodoc]] ChameleonVQVAE
    - forward

## ChameleonModel

[[autodoc]] ChameleonModel
    - forward

## ChameleonForConditionalGeneration

[[autodoc]] ChameleonForConditionalGeneration
    - forward
