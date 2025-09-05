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

# Ovis2

## Overview

The [Ovis2](https://github.com/AIDC-AI/Ovis) is an updated version of the [Ovis](https://huggingface.co/papers/2405.20797) model developed by the AIDC-AI team at Alibaba International Digital Commerce Group. 

Ovis2 is the latest advancement in multi-modal large language models (MLLMs), succeeding Ovis1.6. It retains the architectural design of the Ovis series, which focuses on aligning visual and textual embeddings, and introduces major improvements in data curation and training methods.

<img src="https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/XB-vgzDL6FshrSNGyZvzc.png"  width="600">

<small> Ovis2 architecture.</small>

This model was contributed by [thisisiron](https://huggingface.co/thisisiron).

## Usage example

```python

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers.image_utils import load_images, load_video
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, infer_device

device = f"{infer_device()}:0"

model = AutoModelForVision2Seq.from_pretrained(
    "thisisiron/Ovis2-2B-hf",
    dtype=torch.bfloat16,
).eval().to(device)
processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image."},
        ],
    },
]
url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
image = Image.open(requests.get(url, stream=True).raw)
messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(messages)

inputs = processor(
    images=[image],
    text=messages,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_text)
```

## Ovis2Config

[[autodoc]] Ovis2Config

## Ovis2VisionConfig

[[autodoc]] Ovis2VisionConfig

## Ovis2Model

[[autodoc]] Ovis2Model

## Ovis2ForConditionalGeneration

[[autodoc]] Ovis2ForConditionalGeneration
    - forward

## Ovis2ImageProcessor

[[autodoc]] Ovis2ImageProcessor

## Ovis2ImageProcessorFast

[[autodoc]] Ovis2ImageProcessorFast

## Ovis2Processor

[[autodoc]] Ovis2Processor
