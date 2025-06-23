# Ovis2

## Overview

The [Ovis2](https://github.com/AIDC-AI/Ovis) is an updated version of the [Ovis](https://arxiv.org/abs/2405.20797) model developed by the AIDC-AI team at Alibaba International Digital Commerce Group. 

The abstract from this update is the following:

*It brings major improvements, including better performance for small models, stronger reasoning ability, advanced video and multi-image processing, wider multilingual OCR support, and improved handling of high-resolution images.*


```python

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers.image_utils import load_images, load_video
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "thisisiron/Ovis2-2B-hf",
    torch_dtype=torch.bfloat16,
).eval().to("cuda:0")
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
inputs = inputs.to("cuda:0")
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
