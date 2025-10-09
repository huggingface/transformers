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
*This model was released on 2023-08-24 and added to Hugging Face Transformers on 2024-08-26 and contributed by [simonJJJ](https://huggingface.co/simonJJJ).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# Qwen2-VL

[Qwen2-VL](https://huggingface.co/papers/2409.12191) is a multimodal model family that improves on its predecessor by introducing the Naive Dynamic Resolution mechanism, allowing images of varying resolutions to be processed into different numbers of visual tokens for more efficient and accurate visual representation. It incorporates Multimodal Rotary Position Embedding (M-RoPE) to effectively fuse positional information across text, images, and videos, and uses a unified framework for both image and video processing. The series explores scaling laws for large vision-language models, with versions at 2B, 8B, and 72B parameters, achieving highly competitive performance on multimodal benchmarks. The Qwen2-VL-72B model, in particular, reaches results comparable to GPT-4o and Claude3.5-Sonnet, outperforming other generalist models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
pipeline = pipeline(task="image-text-to-text", model="Qwen/Qwen2-VL-7B-Instruct", dtype="auto")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe the weather in this image."},
        ]
    }
]
pipeline(text=messages,max_new_tokens=20, return_full_text=False)

```

</hfoption>
<hfoption id="Qwen2VLForConditionalGeneration">

```py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type":"text",
                "text":"Describe the weather in this image."
            }
        ]
    }

]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
</hfoptions>

## Qwen2VLConfig

[[autodoc]] Qwen2VLConfig

## Qwen2VLImageProcessor

[[autodoc]] Qwen2VLImageProcessor
    - preprocess

## Qwen2VLImageProcessorFast

[[autodoc]] Qwen2VLImageProcessorFast
    - preprocess

## Qwen2VLProcessor

[[autodoc]] Qwen2VLProcessor

## Qwen2VLModel

[[autodoc]] Qwen2VLModel
    - forward

## Qwen2VLForConditionalGeneration

[[autodoc]] Qwen2VLForConditionalGeneration
    - forward

## Qwen2VLTextConfig

[[autodoc]] Qwen2VLTextConfig

## Qwen2VLTextModel

[[autodoc]] Qwen2VLTextModel
    - forward

## Qwen2VLVideoProcessor

[[autodoc]] Qwen2VLVideoProcessor
