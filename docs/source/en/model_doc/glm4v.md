<!--Copyright 2025 The ZhipuAI Inc. and The HuggingFace Inc. team. All rights reserved.

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
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# GLM-4.1V

## Overview

**GLM-4.1V-9B-Thinking** is a bilingual vision-language model optimized for reasoning, built on GLM-4-9B. It introduces
a "thinking paradigm" with reinforcement learning, achieving state-of-the-art results among 10B-class models and
rivaling 72B-scale models. It supports 64k context, 4K resolution, and arbitrary aspect ratios, with an open-source base
model for further research. You can check our paper [here](https://huggingface.co/papers/2507.01006). and below is a abstract.

*We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to advance general-purpose multimodal understanding
and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework.
We first develop a capable vision foundation model with significant potential through large-scale pre-training, which
arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum
Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a
diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding,
GUI-based agents, and long document understanding. We open-source GLM-4.1V-9B-Thinking, which achieves state-of-the-art
performance among models of comparable size. In a comprehensive evaluation across 28 public benchmarks, our model
outperforms Qwen2.5-VL-7B on nearly all tasks and achieves comparable or even superior performance on 18 benchmarks
relative to the significantly larger Qwen2.5-VL-72B. Notably, GLM-4.1V-9B-Thinking also demonstrates competitive or
superior performance compared to closed-source models such as GPT-4o on challenging tasks including long document
understanding and STEM reasoning, further underscoring its strong capabilities. Code, models and more information
are released at https://github.com/THUDM/GLM-4.1V-Thinking.*

## Usage

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
pipe = pipeline(
    task="image-text-to-text",
    model="THUDM/GLM-4.1V-9B-Thinking",
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
pipe(text=messages,max_new_tokens=20, return_full_text=False)
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import Glm4vForConditionalGeneration, AutoProcessor

model = Glm4vForConditionalGeneration.from_pretrained(
    "THUDM/GLM-4.1V-9B-Thinking",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
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
                "text":"Describe this image."
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
).to("cuda")

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

Using GLM-4.1V with video input is similar to using it with image input.
The model can process video data and generate text based on the content of the video.

```python
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="THUDM/GLM-4.1V-9B-Thinking",
    dtype=torch.bfloat16,
    device_map="cuda:0"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
            },
            {
                "type": "text",
                "text": "discribe this video",
            },
        ],
    }
]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(output_text)
```

## Glm4vConfig

[[autodoc]] Glm4vConfig

## Glm4vTextConfig

[[autodoc]] Glm4vTextConfig

## Glm4vImageProcessor

[[autodoc]] Glm4vImageProcessor
    - preprocess

## Glm4vVideoProcessor

[[autodoc]] Glm4vVideoProcessor
    - preprocess

## Glm4vImageProcessorFast

[[autodoc]] Glm4vImageProcessorFast
    - preprocess

## Glm4vProcessor

[[autodoc]] Glm4vProcessor

## Glm4vTextModel

[[autodoc]] Glm4vTextModel
    - forward

## Glm4vModel

[[autodoc]] Glm4vModel
    - forward

## Glm4vForConditionalGeneration

[[autodoc]] Glm4vForConditionalGeneration
    - forward
