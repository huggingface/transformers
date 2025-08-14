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

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Chameleon model was proposed in [Chameleon: Mixed-Modal Early-Fusion Foundation Models
](https://huggingface.co/papers/2405.09818) by META AI Chameleon Team. Chameleon is a Vision-Language Model that use vector quantization to tokenize images which enables the model to generate multimodal output. The model takes images and texts as input, including an interleaved format, and generates textual response. Image generation module is not released yet.


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

<small> Chameleon incorporates a vector quantizer module to transform images into discrete tokens. That also enables image generation using an auto-regressive transformer. Taken from the <a href="https://huggingface.co/papers/2405.09818">original paper.</a> </small>

This model was contributed by [joaogante](https://huggingface.co/joaogante) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/facebookresearch/chameleon).


## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to set `processor.tokenizer.padding_side = "left"` before generating.

- Note that Chameleon was tuned for safety alignment. If the model is refusing to answer, consider asking a more concrete question, instead of an open question.

- Chameleon generates in chat format which means that the generated text will always be the "assistant's turn". You can enable a text completion generation by passing `return_for_text_completion=True` when calling the processor.

> [!NOTE]
> Chameleon implementation in Transformers uses a special image token to indicate where to merge image embeddings. For special image token we didn't add a new one but used one of the reserved tokens: `<reserved08707>`. You have to add `<image>` to your prompt in the place where the image should be embedded for correct generation.

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
model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", dtype=torch.bfloat16, device_map="cuda")

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

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", dtype=torch.bfloat16, device_map="cuda")

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

## Model optimization

### Quantization using Bitsandbytes

The model can be loaded in 8 or 4 bits, greatly reducing the memory requirements while maintaining the performance of the original model. First make sure to install bitsandbytes, `pip install bitsandbytes` and to have access to a GPU/accelerator that is supported by the library.

<Tip>

bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend).

We value your feedback to help identify bugs before the full release! Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

</Tip>

Simply change the snippet above with:

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
    dtype=torch.bfloat16,
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

## ChameleonImageProcessorFast

[[autodoc]] ChameleonImageProcessorFast
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
