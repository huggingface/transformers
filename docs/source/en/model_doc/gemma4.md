<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-01.*


# Gemma4

## Overview

Gemma 4 is a multimodal model with pretrained and instruction-tuned variants, available in E2B, E4B, 31B and 26B-A4B (MoE) parameter sizes. Gemma 4 models provide the following capabilities:
- Reasoning: All models in the family are designed as highly capable reasoners, with configurable thinking modes.
- Extended Multimodalities: Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B and E4B models).
- Increased Context Window: Small models feature a 128K context window, while the other models support 256K.
- Enhanced Coding & Agentic Capabilities: Achieves notable improvements in coding benchmarks alongside built-in function-calling support, powering highly capable autonomous agents.
- Native System Prompt Support: Gemma 4 introduces built-in support for the system role, enabling more structured and controllable conversations.

You can find all the original Gemma 4 checkpoints under the [Gemma 4](https://huggingface.co/collections/google/gemma-4) release.

### Gemma4 Vision Model

The key difference from previous Gemma releases for vision is the new design to process **images of different sizes** using a **fixed-budget number of tokens**. Unlike many models that squash every image into a fixed square (like 224×224), Gemma 4 keeps the image's natural aspect ratio while making it the right size. There are a couple constraints to follow:
- The total number of pixels must fit within a patch budget
- Both height and width must be divisible by **48** (= patch size 16 × pooling kernel 3)

> [!IMPORTANT]
> Gemma 4 does **not** apply the standard ImageNet mean/std normalization that many other vision models use. The model's own patch embedding layer handles the final scaling internally (shifting values to the [-1, 1] range).

The number of "soft tokens" (aka vision tokens) an image processor can produce is configurable. The supported options are outlined below and the default is **280 soft tokens** per image.


| Soft Tokens | Patches (before pooling) | Approx. Image Area |
|:-----------:|:------------------------:|:-------------------:|
| 70          | 630                      | ~161K pixels        |
| 140         | 1,260                    | ~323K pixels        |
| **280**     | **2,520**                | **~645K pixels**    |
| 560         | 5,040                    | ~1.3M pixels        |
| 1,120       | 10,080                   | ~2.6M pixels        |


To encode positional information for each patch in the image, Gemma 4 uses a learned 2D position embedding table. The position table stores up to 10,240 positions per axis, which allows the model to handle very large images. Each position is a learned vector of the same dimensions as the patch embedding. The 2D RoPE which Gemma 4 uses independently rotate half the attention head dimensions for the x-axis and the other half for the y-axis. This allows the model to understand spatial relationships like "above," "below," "left of," and "right of."



## Usage examples

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-4-E2B-it",
    dtype=torch.bfloat16
)
pipeline(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<|image|>\n\nWhat is shown in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
```

### Function calling

```py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast",
                },
            },
            "required": ["location", "format", "num_days"],
        },
    },
}

messages = [
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in San Francisco, CA (using F)?",
    },
]

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

text = processor.apply_chat_template(
    messages,
    tools=[WEATHER_TOOL],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

### Audio (E2B and E4B Only)

```py
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please transcribe the following audio:"},
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
        ],
    }
]

model = AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it",
    padding_side="left"
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

## Gemma4AudioConfig

[[autodoc]] Gemma4AudioConfig

## Gemma4VisionConfig

[[autodoc]] Gemma4VisionConfig

## Gemma4TextConfig

[[autodoc]] Gemma4TextConfig

## Gemma4Config

[[autodoc]] Gemma4Config

## Gemma4AudioFeatureExtractor

[[autodoc]] Gemma4AudioFeatureExtractor
    - __call__

## Gemma4ImageProcessorPil

[[autodoc]] Gemma4ImageProcessorPil
    - preprocess

## Gemma4ImageProcessor

[[autodoc]] Gemma4ImageProcessor
    - preprocess

## Gemma4VideoProcessor

[[autodoc]] Gemma4VideoProcessor
    - preprocess

## Gemma4Processor

[[autodoc]] Gemma4Processor
    - __call__

## Gemma4PreTrainedModel

[[autodoc]] Gemma4PreTrainedModel
    - forward

## Gemma4AudioModel

[[autodoc]] Gemma4AudioModel
    - forward

## Gemma4VisionModel

[[autodoc]] Gemma4VisionModel
    - forward

## Gemma4TextModel

[[autodoc]] Gemma4TextModel
    - forward

## Gemma4ForCausalLM

[[autodoc]] Gemma4ForCausalLM

## Gemma4Model

[[autodoc]] Gemma4Model
    - forward

## Gemma4ForConditionalGeneration

[[autodoc]] Gemma4ForConditionalGeneration
    - forward
