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
*This model was released on 2025-01-22 and added to Hugging Face Transformers on 2025-10-13.*

# VideoLLaMA3

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
</div>

## Overview

The [VideoLLaMA3](https://huggingface.co/papers/2501.13106) model is a major update to [VideoLLaMA2](https://huggingface.co/papers/2406.07476) from Alibaba DAMO Academy.

The abstract from the paper is as following:

*In this paper, we propose VideoLLaMA 3, a more advanced multimodal foundation model for image and video understanding. The core design philosophy of VideoLLaMA3 is vision-centric. The meaning of “vision-centric” is two-fold: the vision-centric training paradigm and vision-centric framework design. The key insight of our vision-centric training paradigm is that high-quality image-text data is crucial for both image and video understanding. Instead of preparing massive video-text datasets, we focus on constructing large-scale, high-quality image-text datasets. VideoLLaMA3 has four training stages: 1) Vision Encoder Adaptation, which enables the vision encoder to accept images of variable resolutions
as input; 2) Vision-Language Alignment, which jointly tunes the vision encoder, projector, and LLM with large-scale image-text data covering multiple types (including scene images, documents, and charts) as well as text-only data. 3) Multi-task Fine-tuning, which incorporates image-text SFT data for downstream tasks and video-text data to establish a foundation for video understanding. 4) Video-centric Fine-tuning, which further improves the model’s capability in video understanding. As for the framework design, to better capture fine-grained details in images, the pretrained vision encoder is adapted to encode images of varying sizes into vision tokens with corresponding numbers, rather than a fixed number of tokens. For video inputs, we reduce the number of vision tokens according to their similarity so that the representation of videos will be more precise and compact. Benefiting from vision-centric designs, VideoLLaMA3 achieves compelling performances in both image and video understanding benchmarks.*

<img src="https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/pipeline.jpg"
alt="drawing" width="600"/>

<small> VideoLLaMA3 architecture. Taken from the <a href="https://huggingface.co/papers/2501.13106">technical report.</a> </small>

This model was contributed by [lkhl](https://huggingface.co/lkhl).

## Usage example

### Single Media inference

The model can accept both images and videos as input. Here's an example code for inference.

```python
import torch
from transformers import VideoLlama3ForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = VideoLlama3ForConditionalGeneration.from_pretrained("lkhl/VideoLLaMA3-2B-Image-HF", device_map="auto")
processor = AutoProcessor.from_pretrained("lkhl/VideoLLaMA3-2B-Image-HF")


conversation = [
    {
        "role":"user",
        "content":[
            {"type": "image", "image": "https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/sora.png"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)



# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/cat_and_chicken.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as images, videos, and text. Here is an example.

```python
# Image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/sora.png"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Video
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/cat_and_chicken.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

# Text
conversation3 = [
    {
        "role": "user",
        "content": "What color is a banana?"
    }
]


conversations = [conversation1, conversation2, conversation3]
# Preparation for batch inference
inputs = processor.apply_chat_template(
    conversations,
    fps=1,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    padding_side="left",
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# Batch Inference
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using Flash Attention-2, simply add `attn_implementation="flash_attention_2"` when loading the model as follows:

```python
from transformers import VideoLlama3ForConditionalGeneration

model = VideoLlama3ForConditionalGeneration.from_pretrained(
    "lkhl/VideoLLaMA3-2B-Image-HF", 
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```

## VideoLlama3Config

[[autodoc]] VideoLlama3Config

## VideoLlama3VisionConfig

[[autodoc]] VideoLlama3VisionConfig

## VideoLlama3ImageProcessor

[[autodoc]] VideoLlama3ImageProcessor
    - preprocess

## VideoLlama3VideoProcessor

[[autodoc]] VideoLlama3VideoProcessor
    - preprocess

## VideoLlama3ImageProcessorFast

[[autodoc]] VideoLlama3ImageProcessorFast
    - preprocess

## VideoLlama3Processor

[[autodoc]] VideoLlama3Processor
    - __call__

## VideoLlama3Model

[[autodoc]] VideoLlama3Model
    - forward

## VideoLlama3VisionModel

[[autodoc]] VideoLlama3VisionModel
    - forward

## VideoLlama3ForConditionalGeneration

[[autodoc]] VideoLlama3ForConditionalGeneration
    - forward
