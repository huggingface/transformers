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
*This model was released on 2024-10-09 and added to Hugging Face Transformers on 2024-09-14 and contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ).*

# Pixtral

[Pixtral](https://huggingface.co/papers/2410.07073) is a multimodal language model capable of understanding both natural images and documents while maintaining strong text-only performance. It features a vision encoder trained from scratch that processes images at their natural resolution and aspect ratio, allowing flexible token usage and handling multiple images within a 128K-token context window. The model outperforms similarly sized open models like Llama-3.2 11B and Qwen-2-VL 7B, as well as much larger models such as Llama-3.2 90B, while being significantly smaller. Additionally, Pixtral-12B is released under an Apache 2.0 license alongside the open-source MM-MT-Bench benchmark for standardized evaluation of vision-language models.

<hfoptions id="usage">
<hfoption id="LlavaForConditionalGeneration">

```py
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("mistral-community/pixtral-12b", dtype="auto")
processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can you name the animal"}, 
        {"type": "image", "url": url_dog}, 
        {"type": "text", "content": "that lives here?"}, 
        {"type": "image", "url" : url_mountain}
      ]
    }
]

inputs = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors"pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```

</hfoption>
</hfoptions>

## Usage tips

- Pixtral uses [`PixtralVisionModel`] as the vision encoder and [`MistralForCausalLM`] for its language decoder.
- The model internally replaces `[IMG]` token placeholders with image embeddings. The number of `[IMG]` tokens depends on each image's height and width.
- Each image row separates by a `[IMG_BREAK]` token. Each image separates by an `[IMG_END]` token. Use [`Processor.apply_chat_template`] to manage these tokens automatically.

## PixtralVisionConfig

[[autodoc]] PixtralVisionConfig

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
