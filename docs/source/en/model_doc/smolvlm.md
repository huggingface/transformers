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
*This model was released on 2025-04-07 and added to Hugging Face Transformers on 2025-02-20 and contributed by [orrzohar](https://huggingface.co/orrzohar).*

# SmolVLM

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[SmolVLM2](https://huggingface.co/papers/2504.05299) is a series of compact vision-language models designed for efficient on-device inference, emphasizing low GPU memory usage without sacrificing performance. The models optimize architecture, tokenization, and training data to reduce computational overhead, enabling substantial gains on image and video tasks. SmolVLM-256M, the smallest model, uses under 1GB of GPU memory yet outperforms a model 300 times its size, while the largest 2.2B-parameter model rivals state-of-the-art VLMs using twice the memory. These models demonstrate that careful design and data curation can deliver high multimodal performance with minimal resource requirements.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", dtype="auto")

conversation = [
    {
        "role": "user",
        "content":[
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe the weather in this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)
```

</hfoption>
</hfoptions>

## SmolVLMConfig

[[autodoc]] SmolVLMConfig

## SmolVLMVisionConfig

[[autodoc]] SmolVLMVisionConfig

## Idefics3VisionTransformer

[[autodoc]] SmolVLMVisionTransformer

## SmolVLMModel

[[autodoc]] SmolVLMModel
    - forward

## SmolVLMForConditionalGeneration

[[autodoc]] SmolVLMForConditionalGeneration
    - forward

## SmolVLMImageProcessor

[[autodoc]] SmolVLMImageProcessor
    - preprocess

## SmolVLMImageProcessorFast

[[autodoc]] SmolVLMImageProcessorFast
    - preprocess

## SmolVLMProcessor

[[autodoc]] SmolVLMProcessor
    - __call__

## SmolVLMVideoProcessor
[[autodoc]] SmolVLMVideoProcessor
