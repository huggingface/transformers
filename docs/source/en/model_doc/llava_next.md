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
    <img alt="Multimodal" src="https://img.shields.io/badge/Multimodal-vision--language-blue">
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
from transformers import pipeline
from PIL import Image
import requests

pipe = pipeline("image-to-text", model="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda")
image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png", stream=True).raw)

result = pipe(image, prompt="What does this chart show?")
print(result[0]["generated_text"])
```

</hfoption>

<hfoption id="AutoModel">

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests, torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16
).to("cuda")

image = Image.open(requests.get(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png", stream=True).raw)

conversation = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does this chart show?"}]}
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>

<hfoption id="transformers-cli">
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForImageTextToText.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=quant_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
```

Use the AttentionMaskVisualizer to explore which tokens the model attends to:

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

viz = AttentionMaskVisualizer("llava-hf/llava-v1.6-mistral-7b-hf")
viz("<image> What is shown in this image?")
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"/>
</div>

## Notes

* Different checkpoints (Mistral, Vicuna, etc.) require a specific prompt format depending on the underlying LLM. Always use [`~ProcessorMixin.apply_chat_template`] to ensure correct formatting. Refer to the [Templates](../chat_templating) guide for more details.
- The example below demonstrates inference with multiple input images.

   ```py
   add code snippet here


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
