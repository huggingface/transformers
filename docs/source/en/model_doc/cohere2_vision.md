# Command A Vision

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
</div>

## Overview

Command A Vision is a state-of-the-art multimodal model designed to seamlessly integrate visual and textual information for a wide range of applications. By combining advanced computer vision techniques with natural language processing capabilities, Command A Vision enables users to analyze, understand, and generate insights from both visual and textual data.

The model excels at tasks including image captioning, visual question answering, document understanding, and chart understanding. This makes it a versatile tool for AI practitioners. Its ability to process complex visual and textual inputs makes it useful in settings where text-only representations are imprecise or unavailable, like real-world image understanding and graphics-heavy document processing.

Command A Vision is built upon a robust architecture that leverages the latest advancements in VLMs. It's highly performant and efficient, even when dealing with large-scale datasets. The model's flexibility makes it suitable for a wide range of use cases, from content moderation and image search to medical imaging analysis and robotics.

## Usage tips

The model and image processor can be loaded as follows:

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "CohereLabs/command-a-vision-07-2025"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", dtype=torch.float16
)

# Format message with the Command-A-Vision chat template
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
            },
            {"type": "text", "text": "what is in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    padding=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)

print(
    processor.tokenizer.decode(
        gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
)
```

</hfoption>
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(model="CohereLabs/command-a-vision-07-2025", task="image-text-to-text", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo=",
            },
            {"type": "text", "text": "Where was this taken ?"},
        ],
    },
]

outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)

print(outputs)
```
</hfoption>
</hfoptions>

## Cohere2VisionConfig

[[autodoc]] Cohere2VisionConfig

## Cohere2VisionForConditionalGeneration

[[autodoc]] Cohere2VisionForConditionalGeneration
    - forward

## Cohere2VisionModel

[[autodoc]] Cohere2VisionModel
    - forward

## Cohere2VisionImageProcessorFast

[[autodoc]] Cohere2VisionImageProcessorFast
    - preprocess

## Cohere2VisionProcessor

[[autodoc]] Cohere2VisionProcessor
