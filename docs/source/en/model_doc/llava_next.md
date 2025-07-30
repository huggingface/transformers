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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# LLaVA-NeXT

[LLaVA‑NeXT](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/) improves on [Llava](./llava) by increasing the input image resolution by 4x more pixels and supporting 3 aspect ratios (up to 672x672, 336x1344, 1344x336) to better grasp visual details. It is also trained on an improved visual instruction tuning dataset covering more scenarios and applications to improve OCR and common sense reasoning.

You can find all the original LLaVA‑NeXT checkpoints under the [LLaVA-NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf) collection.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the LLaVA‑NeXT models in the right sidebar for more examples of how to apply Llava-NeXT to different multimodal tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
import torch  
from transformers import pipeline  

pipeline = pipeline(  
    task="image-text-to-text",  
    model="llava-hf/llava-v1.6-mistral-7b-hf",  
    device=0,  
    dtype=torch.bfloat16  
)  
messages = [  
    {  
        "role": "user",  
        "content": [  
            {  
                "type": "image",  
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",  
            },  
            { "type": "text", "text": "Describe this image."},  
        ]  
    }  
]  
pipeline(text=messages, max_new_tokens=20, return_full_text=False)
```

</hfoption>

<hfoption id="AutoModel">

```python
import torch  
import requests  
from PIL import Image  
from transformers import AutoProcessor, LlavaNextForConditionalGeneration  

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")  
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16).to("cuda")  

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"  
image = Image.open(requests.get(url, stream=True).raw)  

conversation = [  
    {  
        "role": "user",  
        "content": [  
            {"type": "image"},  
            {"type": "text", "text": "What is shown in this image?"},  
        ],  
    },  
]  
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)  
inputs = processor(image, prompt, return_tensors="pt").to("cuda")  
output = model.generate(**inputs, max_new_tokens=100)  
print(processor.decode(output[0], skip_special_tokens=True))  
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
import torch  
import requests  
from PIL import Image  
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig  

quant_config = BitsAndBytesConfig(  
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4"  
)  

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")  
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quant_config, device_map="auto")  

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"  
image = Image.open(requests.get(url, stream=True).raw)  

conversation = [  
    {  
        "role": "user",  
        "content": [  
            {"type": "image"},  
            {"type": "text", "text": "What does this chart show?"},  
        ],  
    },  
]  
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)  
inputs = processor(image, prompt, return_tensors="pt").to("cuda")  

with torch.inference_mode():  
    output = model.generate(**inputs, max_new_tokens=100)  
print(processor.decode(output[0], skip_special_tokens=True))  
```


## Notes

* Different checkpoints (Mistral, Vicuna, etc.) require a specific prompt format depending on the underlying LLM. Always use [`~ProcessorMixin.apply_chat_template`] to ensure correct formatting. Refer to the [Templates](../chat_templating) guide for more details.

* Set `padding_side="left"` during batched generation for more accurate results.

```py
processor.tokenizer.padding_side = "left"
```

* LLaVA-NeXT uses different numbers of patches for images and pads the inputs inside the modeling code except when padding is done during processing. The default setting is *left-padding* if the model is in `eval()` mode, otherwise it is *right-padding*.

* LLaVA models after v4.46 raises warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}`, and `processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. It is strongly recommended to add these attributes to the processor if you own the model checkpoint or open a PR if it isn't.

  Adding these attributes means LLaVA will try to infer the number of image tokens required per image and expand the text with the same number of `<image>` token placeholders. There are usually ~500 tokens per image, so make sure the text is not truncated because it will cause a failure when merging the embeddings. The attributes can be found in `model.config.vision_config.patch_size` or `model.config.vision_feature_select_strategy`.

  The `num_additional_image_tokens` should be `1` if the vision backbone adds a `CLS` token or `0` if nothing extra is added.

* The example below demonstrates inference with multiple input images.

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests, torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16
).to("cuda")

# Load multiple images
url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_comparison.png"

image1 = Image.open(requests.get(url1, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

conversation = [
    {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare these two images and describe the differences."}]}
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor([image1, image2], prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```


## LlavaNextConfig

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor

[[autodoc]] LlavaNextImageProcessor
    - preprocess

## LlavaNextImageProcessorFast

[[autodoc]] LlavaNextImageProcessorFast
    - preprocess

## LlavaNextProcessor

[[autodoc]] LlavaNextProcessor

## LlavaNextModel

[[autodoc]] LlavaNextModel

## LlavaNextForConditionalGeneration

[[autodoc]] LlavaNextForConditionalGeneration
    - forward
