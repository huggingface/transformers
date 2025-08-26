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

You can find all the original Pixtral checkpoints under the [Mistral AI](https://huggingface.co/mistralai/models?search=pixtral) organization.

> [!TIP]
> This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ).
> Click on the Pixtral models in the right sidebar for more examples of how to apply Pixtral to different vision and language tasks.

<hfoptions id="usage">

<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"}, 
        {"type": "image", "url": "https://picsum.photos/id/237/200/300"}, 
        {"type": "text", "content": "live here?"}, 
        {"type": "image", "url": "https://picsum.photos/seed/picsum/200/300"}
      ]
    }
]

inputs = processor.apply_chat_template(
    chat,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

</hfoptions>

## Notes

- Pixtral is a multimodal model that follows the [Llava](llava) architecture, using a [`PixtralVisionModel`] and a [`MistralForCausalLM`] decoder.
- The model internally replaces `[IMG]` token placeholders with image embeddings. To correctly format the prompt with text and images, it is highly recommended to use the `apply_chat_template` method of the processor, which handles the complex formatting automatically.

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
