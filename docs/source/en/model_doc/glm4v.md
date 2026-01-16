<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-01 and added to Hugging Face Transformers on 2025-06-25.*

# GLM-V

## Overview

The GLM-V model was proposed in [GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://huggingface.co/papers/2507.01006v6).

The abstract from the paper is the following:

> *We present GLM-4.1V-Thinking, GLM-4.5V, and GLM-4.6V, a family of vision-language models (VLMs) designed to advance
general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of
the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential
through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose
Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to
comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video
understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a
comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks
among open-source models of similar size, and demonstrates competitive or even superior results compared to
closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the
smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on
29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. We further introduce the GLM-4.6V series,
open-source multimodal models with native tool use and a 128K context window. A brief overview is available at this
https URL. Code, models and more information are released at https://github.com/zai-org/GLM-V*

## Support Model

This Model type support these model of zai-org:

+ [GLM-4.1V-9B-Base](https://huggingface.co/zai-org/GLM-4.1V-9B-Base)
+ [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
+ [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash)
+ [AutoGLM-Phone-9B](https://huggingface.co/zai-org/AutoGLM-Phone-9B)
+ [AutoGLM-Phone-9B-Multilingual](https://huggingface.co/zai-org/AutoGLM-Phone-9B-Multilingual)
+ [Glyph](https://huggingface.co/zai-org/Glyph)
+ [WebVIA-Agent](https://huggingface.co/zai-org/WebVIA-Agent)
+ [UI2Code_N](https://huggingface.co/zai-org/UI2Code_N)

This model was contributed by [Raushan Turganbay](https://huggingface.co/RaushanTurganbay)
and [Yuxuan Zhang](https://huggingface.co/ZHANGYUXUAN-zR).

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
            {"type": "text", "text": "Describe this image."},
        ]
    }
]
pipe(text=messages, max_new_tokens=20, return_full_text=False)
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
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type": "text",
                "text": "Describe this image."
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
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
from accelerate import Accelerator
import torch

device = Accelerator().device

processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="THUDM/GLM-4.1V-9B-Thinking",
    dtype=torch.bfloat16,
    device_map=device
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
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output_text)
```

## Glm4vConfig

[[autodoc]] Glm4vConfig

## Glm4vVisionConfig

[[autodoc]] Glm4vVisionConfig

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
    - __call__

## Glm4vVisionModel

[[autodoc]] Glm4vVisionModel
- forward

## Glm4vTextModel

[[autodoc]] Glm4vTextModel
- forward

## Glm4vModel

[[autodoc]] Glm4vModel
- forward

## Glm4vForConditionalGeneration

[[autodoc]] Glm4vForConditionalGeneration
- forward
