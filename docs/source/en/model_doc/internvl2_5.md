<!--Copyright 2025 The OpenGVLab and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# InternVL 2.5

## Overview

The [InternVL 2.5](https://github.com/OpenGVLab/InternVL) model was proposed to [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](https://arxiv.org/abs/2412.05271) from OpenGVLab. 

The abstract from the paper is the following:

*We introduce InternVL2.5, an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0, maintaining its core model architecture while introducing significant enhancements in training and testing strategies as well as data quality.<br>
In this work, we delve into the relationship between model scaling and performance, systematically exploring the performance trends in vision encoders, language models, dataset sizes, and test-time configurations. Through extensive evaluations on a wide range of benchmarks, InternVL2.5 exhibits competitive performance, rivaling leading commercial models such as GPT-4o and Claude-3.5-Sonnet.*

InternVL2.5 family is built upon the following designs:

**Progressive Scaling Strategy**: We propose a progressive scaling strategy to efficiently align the vision encoder with LLMs. This strategy adopts a staged training approach, starting with smaller, resource-efficient LLMs and progressively scaling up to larger LLMs. This approach stems from our observation that even when the ViT and LLM are jointly trained using NTP loss, the resulting visual features are generalizable representations that can be easily understood by other LLMs. Specifically, the InternViT is trained alongside a smaller LLM (e.g., 20B), focusing on optimizing fundamental visual capabilities and cross-modal alignment. This phase avoids the high computational costs associated with training directly with a large LLM. Using a shared-weight mechanism, the trained InternViT can be seamlessly transferred to a larger LLM (e.g., 72B) without requiring retraining. Consequently, when training a larger model, much less data is required and the computation cost is significantly reduced.

**Improved Training Strategy**: To enhance the model’s adaptability to real-world scenarios and overall performance, we introduce two key techniques: Random JPEG Compression and Loss Reweighting. For Random JPEG Compression, random JPEG compression with quality levels between 75 and 100 is applied to simulate the degradation commonly found in internet-sourced images. For Loss Reweighting, we express the widely applied strategies (i.e., token averaging and sample averaging) in a unified format and propose square averaging to balance the gradients biases towards long or short responses.

**Well-structed Data Organization**: During model development, we observed that even a small fraction of anomalous samples can lead to aberrant model behavior during inference. To address this issue, we propose a filtering pipeline consisting of LLM-Based Quality Scoring and Rule-Based Filtering, which significantly reduced the occurrence of anomalous behaviors, particularly repetitive generation, with notable improvements in CoT reasoning tasks. Additionally, we implement a data-packing strategy to enhance GPU utilization and improve training efficiency.

<small> 
This model was contributed by [thisisiron](https://huggingface.co/thisisiron). The original code can be found [here](https://github.com/OpenGVLab/InternVL).
</small>

## Usage example

### Converting the Original Model to Hugging Face Format
Run the following code to convert the original model into the Hugging Face format.

```
python convert_internvl2_5_to_hf.py --model_name InternVL2_5-1B
```

### Inference

Here's an example code for inference.

```python

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers.image_utils import load_images, load_video
from transformers import InternVL2_5ForConditionalGeneration, InternVL2_5Processor

# Load the model
model = InternVL2_5ForConditionalGeneration.from_pretrained("thisisiron/InternVL2_5-1B", device_map="auto")
processor = InternVL2_5Processor.from_pretrained("thisisiron/InternVL2_5-1B")

# Image
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)

```


## InternVL2_5Config

[[autodoc]] InternVL2_5Config

## InternVL2_5ImageProcessor

[[autodoc]] InternVL2_5ImageProcessor
    - preprocess

## InternVL2_5Processor

[[autodoc]] InternVL2_5Processor

## InternVL2_5ForConditionalGeneration

[[autodoc]] InternVL2_5ForConditionalGeneration
    - forward
