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

# DeepseekVL

## Overview

<!-- TODO: update this para -->

The DeepseekVL Model was originally proposed in [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525) by DeepSeek AI team. DeepseekVL is a vision-language model that can generate both image and text output, it can also take both images and text as input.

The abstract from the original paper is the following:

*We present DeepSeek-VL, an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. Our approach is structured around three key dimensions:
We strive to ensure our data is diverse, scalable, and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content, aiming for a comprehensive representation of practical contexts. Further, we create a use case taxonomy from real user scenarios and construct an instruction tuning dataset accordingly. The fine-tuning with this dataset substantially improves the model's user experience in practical applications. Considering efficiency and the demands of most real-world scenarios, DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024), while maintaining a relatively low computational overhead. This design choice ensures the model's ability to capture critical semantic and detailed information across various visual tasks. We posit that a proficient Vision-Language Model should, foremost, possess strong language abilities. To ensure the preservation of LLM capabilities during pretraining, we investigate an effective VL pretraining strategy by integrating LLM training from the beginning and carefully managing the competitive dynamics observed between vision and language modalities.
The DeepSeek-VL family (both 1.3B and 7B models) showcases superior user experiences as a vision-language chatbot in real-world applications, achieving state-of-the-art or competitive performance across a wide range of visual-language benchmarks at the same model size while maintaining robust performance on language-centric benchmarks. We have made both 1.3B and 7B models publicly accessible to foster innovations based on this foundation model.*

This model was contributed by [Armaghan](https://huggingface.co/geetu040).
The original code can be found [here](https://github.com/deepseek-ai/DeepSeek-VL).

## Usage Example

<!-- TODO: update example -->

Here is the example of visual understanding with a single image.

> [!NOTE]
> Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

```python
import torch  
from PIL import Image  
import requests  

from transformers import DeepseekVLForConditionalGeneration, DeepseekVLProcessor  

model_id = "yaswanthgali/DeepseekVL-Pro-1B-HF"
# Prepare Input for generation.
image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw)
messages = [
    {
        "role": "user",
        "content": [
            {'type':'image', 'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
            {'type':"text", "text":"What do you see in this image?."}
        ]
    },
]

# Set generation mode to `text` to perform text generation.
processor = DeepseekVLProcessor.from_pretrained(model_id)
model = DeepseekVLForConditionalGeneration.from_pretrained(model_id,     
        torch_dtype=torch.bfloat16,
        device_map="auto")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    generation_mode="text",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=40,generation_mode='text',do_sample=True)
text = processor.decode(output[0], skip_special_tokens=True)
print(text)
```

## DeepseekVLConfig

[[autodoc]] DeepseekVLConfig

## DeepseekVLProcessor

[[autodoc]] DeepseekVLProcessor

## DeepseekVLImageProcessor

[[autodoc]] DeepseekVLImageProcessor

## DeepseekVLImageProcessorFast

[[autodoc]] DeepseekVLImageProcessorFast

## DeepseekVLModel

[[autodoc]] DeepseekVLModel
    - forward

## DeepseekVLForConditionalGeneration

[[autodoc]] DeepseekVLForConditionalGeneration
    - forward
