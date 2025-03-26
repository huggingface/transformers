<!--Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepseekVL

## Overview

The Deepseek-VL model was introduced in [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525) by the DeepSeek AI team. It is a vision-language model (VLM) designed to process both text and images for generating contextually relevant responses. The model leverages [LLaMA](./llama) as its text encoder, while [SigLip](./siglip) is used for encoding low-resolution images. In some variants, [SAM (Segment Anything Model)](./sam) is incorporated to handle high-resolution image encoding, enhancing the model’s ability to process fine-grained visual details.

The abstract from the original paper is the following:

*We present DeepSeek-VL, an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. Our approach is structured around three key dimensions:
We strive to ensure our data is diverse, scalable, and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content, aiming for a comprehensive representation of practical contexts. Further, we create a use case taxonomy from real user scenarios and construct an instruction tuning dataset accordingly. The fine-tuning with this dataset substantially improves the model's user experience in practical applications. Considering efficiency and the demands of most real-world scenarios, DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024), while maintaining a relatively low computational overhead. This design choice ensures the model's ability to capture critical semantic and detailed information across various visual tasks. We posit that a proficient Vision-Language Model should, foremost, possess strong language abilities. To ensure the preservation of LLM capabilities during pretraining, we investigate an effective VL pretraining strategy by integrating LLM training from the beginning and carefully managing the competitive dynamics observed between vision and language modalities.
The DeepSeek-VL family (both 1.3B and 7B models) showcases superior user experiences as a vision-language chatbot in real-world applications, achieving state-of-the-art or competitive performance across a wide range of visual-language benchmarks at the same model size while maintaining robust performance on language-centric benchmarks. We have made both 1.3B and 7B models publicly accessible to foster innovations based on this foundation model.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deepseek_vl_outputs.png"
alt="DeepseekVL Outputs" width="600"/>

<small> DeepseekVL Outputs. Taken from the <a href="https://github.com/deepseek-ai/DeepSeek-VL" target="_blank">official code</a>. </small>

This model was contributed by [Armaghan](https://huggingface.co/geetu040).
The original code can be found [here](https://github.com/deepseek-ai/DeepSeek-VL).

## Usage Example

### Single image inference

Here is the example of visual understanding with a single image.

```python
>>> import torch
>>> from transformers import DeepseekVLForConditionalGeneration, DeepseekVLProcessor

>>> # model_id = "deepseek-ai/deepseek-vl-7b-chat-hf"
>>> model_id = "deepseek-ai/deepseek-vl-1.3b-chat-hf"

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {'type':'image', 'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
...             {'type':"text", "text":"What do you see in this image?."}
...         ]
...     },
... ]

>>> processor = DeepseekVLProcessor.from_pretrained(model_id)
>>> model = DeepseekVLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

>>> inputs = processor.apply_chat_template(
...     messages,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=40, do_sample=True)
>>> text = processor.decode(output[0], skip_special_tokens=True)
```

As can be seen, the instruction-tuned model requires a [chat template](../chat_templating) to be applied to make sure the inputs are prepared in the right format.

### Multi image inference

DeepseekVL can perform inference with multiple images as input, where images can belong to the same prompt or different prompts in batched inference, where the model processes many conversations in parallel. Here is how you can do it:

```python
>>> import torch
>>> from transformers import DeepseekVLForConditionalGeneration, DeepseekVLProcessor

>>> # model_id = "deepseek-ai/deepseek-vl-7b-chat-hf"
>>> model_id = "deepseek-ai/deepseek-vl-1.3b-chat-hf"

>>> image_urls = [
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     "https://www.ilankelman.org/stopsigns/australia.jpg",
...     "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
... ]

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "text", "text": "What’s the difference between"},
...                 {"type": "image", "url": image_urls[0]},
...                 {"type": "text", "text": " and "},
...                 {"type": "image", "url": image_urls[1]}
...             ]
...         }
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": image_urls[2]},
...                 {"type": "text", "text": "What do you see in this image?"}
...             ]
...         }
...     ]
... ]

>>> processor = DeepseekVLProcessor.from_pretrained(model_id)
>>> model = DeepseekVLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

>>> inputs = processor.apply_chat_template(
...     messages,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=40, do_sample=True)
>>> text = processor.decode(output[0], skip_special_tokens=True)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DepthPro:

- Research Paper: [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525)
- Official Implementation: [deepseek-ai/DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

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
