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
*This model was released on 2024-09-17 and added to Hugging Face Transformers on 2024-09-14.*


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Pixtral

[Pixtral](https://huggingface.co/papers/2410.07073) is a multimodal model trained to understand natural images and documents. It accepts images in their natural resolution and aspect ratio without resizing or padding due to it's 2D RoPE embeddings. In addition, Pixtral has a long 128K token context window for processing a large number of images. Pixtral couples a 400M vision encoder with a 12B Mistral Nemo decoder.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/pixtral_architecture.webp"
alt="drawing" width="600"/>

<small> Pixtral architecture. Taken from the <a href="https://mistral.ai/news/pixtral-12b/">blog post.</a> </small>

You can find all the original Pixtral checkpoints under the [Mistral AI](https://huggingface.co/mistralai/models?search=pixtral) organization.

> [!TIP]
> This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ).
> Click on the Pixtral models in the right sidebar for more examples of how to apply Pixtral to different vision and language tasks.

<hfoptions id="usage">

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"}, 
        {"type": "image", "url": url_dog}, 
        {"type": "text", "content": "live here?"}, 
        {"type": "image", "url" : url_mountain}
      ]
    }
]

inputs = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors"pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the model to 4-bits.

```python
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

model_id = "mistral-community/pixtral-12b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

dog_url = "https://picsum.photos/id/237/200/300"
mountain_url = "https://picsum.photos/seed/picsum/200/300"
dog_image = Image.open(requests.get(dog_url, stream=True).raw)
mountain_image = Image.open(requests.get(mountain_url, stream=True).raw)

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "text": "Can this animal"},
        {"type": "image"},
        {"type": "text", "text": "live here?"},
        {"type": "image"}
      ]
    }
]

prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[dog_image, mountain_image], return_tensors="pt")

inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

generate_ids = model.generate(**inputs, max_new_tokens=100)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output)
```

## Notes

- Pixtral uses [`PixtralVisionModel`] as the vision encoder and [`MistralForCausalLM`]  for its language decoder.
- The model internally replaces `[IMG]` token placeholders with image embeddings.

    ```py
    "<s>[INST][IMG]\nWhat are the things I should be cautious about when I visit this place?[/INST]"
    ```

    The `[IMG]` tokens are replaced with a number of `[IMG]` tokens that depend on the height and width of each image. Each row of the image is separated by a `[IMG_BREAK]` token and each image is separated by a `[IMG_END]` token. Use the [`~Processor.apply_chat_template`] method to handle these tokens for you.

## PixtralVisionConfig

[[autodoc]] PixtralVisionConfig

## MistralCommonTokenizer

[[autodoc]] MistralCommonTokenizer

## PixtralVisionModel

[[autodoc]] PixtralVisionModel
    - forward

## PixtralImageProcessor

[[autodoc]] PixtralImageProcessor
    - preprocess

## PixtralImageProcessorFast

[[autodoc]] PixtralImageProcessorFast
    - preprocess

## PixtralProcessor

[[autodoc]] PixtralProcessor
