<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

*This paper was published on Hugging Face Papers on 2026-05-25.*

# LLaVA-OneVision-2

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

LLaVA-OneVision-2 was introduced in [LLaVA-OneVision-2: Towards Next-Generation Perceptual Intelligence](https://huggingface.co/papers/2605.25979) by Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, Yifei Shen, Yuanhan Zhang, Kaichen Zhang, Wenkang Zhang, Zheng Cheng, Nansen Zhang, Chunsheng Wu, Chunjiang Ge, Zimin Ran, Dehua Song, Chunyuan Li, Shikun Feng, Ming Hu, Zhangquan Chen, Junbo Niu, Bo Li, Ziyong Feng, Ziwei Liu, Zongyuan Ge, and Jiankang Deng.

LLaVA-OneVision-2 is an open vision-language model for image understanding, long-video understanding, and spatial reasoning. It combines a native-resolution OneVision-Encoder, a lightweight vision-language connector, and an autoregressive text decoder. Images, sampled video frames, and codec-derived visual canvases share the same visual encoder. Codec-stream tokenization uses compression signals to select temporal and spatial evidence, allowing video input to concentrate its visual-token budget on relevant content.

The release includes an **8B Instruct** checkpoint, together with training code, data resources, and recipes. The paper studies video question answering, temporal grounding, spatial grounding, tracking, and manipulation-trace reasoning, and introduces JumpScore to evaluate fine-grained grounding in repeated motion. See the [paper](https://arxiv.org/abs/2605.25979) for task-specific protocols and evaluation results.

### Abstract

The paper's abstract can be summarized as follows:

*LLaVA-OneVision-2 explores a common representation for images, sampled video frames, and codec-derived visual evidence. Its native-resolution encoder uses spatiotemporal positions, while codec-stream tokenization allocates visual tokens according to compression signals and motion residuals. This makes it possible to retain relevant events over longer videos within a constrained token budget. The training recipe combines open image supervision, re-captioned videos, and spatial data. The work also introduces JumpScore for evaluating repeated-motion temporal grounding and studies image understanding, long-video reasoning, spatial grounding, and tracking. Models, datasets, and training resources are released to support reproducible multimodal research.*

<img src="https://arxiv.org/html/2605.25979v1/arch.png" alt="LLaVA-OneVision-2 architecture: codec videos, sampled frames, and images pass through the shared OneVision-Encoder, and the visual embeddings and text tokens are processed by an autoregressive decoder." width="1000"/>

<small>LLaVA-OneVision-2 architecture, reproduced without changes from Figure 2 of <a href="https://arxiv.org/html/2605.25979v1#S2.F2">An et al. (2026)</a>, under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Illustrated resolutions, frame counts, and compression ratios are examples rather than fixed processor settings.</small>

### Formatting prompts with chat templates

Use the processor's `apply_chat_template` method to format conversations for the checkpoint. Each message has a `role` and a list of `content` items. Image and video items are placed alongside text items in the order they appear in the conversation.

The following example formats a prompt without loading an image or running the model:

```python
from transformers import AutoProcessor

model_id = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(prompt)
```

Setting `tokenize=False` produces the formatted text prompt. The usage examples below show how to supply the actual images or videos and generate a response.

The original implementation, training resources, and model releases are available in the [official repository](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2).

## Usage examples

### Load the model and processor

Load the model once, then reuse `model` and `processor` in the examples below. The native implementation can load the public checkpoint without `trust_remote_code=True`. Use left padding for batched generation.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
```

### Text-only inference

For a text-only conversation, omit image and video content items:

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What integer comes after 2?"}],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Multiple image inference

Place multiple image items in the same conversation. The processor expands each image placeholder and preserves the order of its visual features:

```python
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg",
]
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_urls[0]},
            {"type": "image", "url": image_urls[1]},
            {"type": "text", "text": "Describe the differences between these two images."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Batched image inference

Pass a list of conversations to process independent requests in one batch. Use `padding=True` together with left padding:

```python
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg",
]
conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_urls[0]},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_urls[1]},
                {"type": "text", "text": "What objects and surroundings are visible in this image?"},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    processor_kwargs={"padding": True},
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Video inference with sampled frames

Download an example video, format a video placeholder, and pass the encoded file to the processor. `num_frames` controls uniform sampling, while `size` sets the per-frame pixel-area bounds. Original frame indices are retained for visual positions, and timestamps are inserted into the prompt.

```python
from huggingface_hub import hf_hub_download

video_path = hf_hub_download(
    repo_id="raushan-testing-hf/videos-test",
    filename="sample_demo_1.mp4",
    repo_type="dataset",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=prompt,
    videos=[video_path],
    videos_kwargs={
        "num_frames": 16,
        "size": {"shortest_edge": 3136, "longest_edge": 200704},
    },
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Video inference with codec canvases

The optional codec backend requires `codec-video-prep` and ffmpeg on `PATH`. Provide an encoded video file rather than decoded frames. Codec preprocessing builds visual canvases and retains each patch's source position; the model then applies the checkpoint's temporal attention windows.

The following example uses an explicit canvas budget and grouping settings:

```python
from huggingface_hub import hf_hub_download

video_path = hf_hub_download(
    repo_id="raushan-testing-hf/videos-test",
    filename="sample_demo_1.mp4",
    repo_type="dataset",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=prompt,
    videos=[video_path],
    videos_kwargs={
        "video_backend": "codec",
        "size": {"shortest_edge": 3136, "longest_edge": 50176},
        "codec_config": {
            "target_canvas": 8,
            "group_size": 8,
            "images_per_group": 4,
            "min_group_frames": 8,
            "max_group_frames": 8,
        },
    },
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

`target_canvas` controls the requested canvas budget; shorter videos can produce fewer usable canvases. `size["longest_edge"]` sets the canvas pixel budget. If no size is supplied, the codec budget is 150000 pixels. The settings in this example were used to check inference on a local encoded video.

<img src="https://raw.githubusercontent.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/5837d88ac60bd922a38464938f9a24ebfc2a2e2f/asset/method_codec_selection_light.svg" alt="Illustration comparing dense uniform frame sampling with codec-aligned selection of dense I-frame patches and sparse P-frame patches." width="900"/>

<small>Codec patch selection from the <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/5837d88ac60bd922a38464938f9a24ebfc2a2e2f">official project</a>. The displayed token budget and temporal coverage are illustrative rather than fixed processor settings.</small>

### Processing notes

- The image processor accepts dynamic image sizes. Use `images_kwargs` or `videos_kwargs` to set pixel budgets for the corresponding modality.
- Native loading honors `video_preprocessor_config.json`. The original remote processor constructs its frame processor from image defaults, so use explicit sampling and pixel budgets when comparing implementations.
- The default image backend uses torchvision; [`LlavaOnevision2ImageProcessorPil`] is available for PIL preprocessing.
- Dynamic visual grid boundaries currently prevent full-graph compilation. The examples above use the validated SDPA attention backend.


## LlavaOnevision2Config

[[autodoc]] LlavaOnevision2Config

## LlavaOnevision2VisionConfig

[[autodoc]] LlavaOnevision2VisionConfig

## LlavaOnevision2ImageProcessor

[[autodoc]] LlavaOnevision2ImageProcessor

## LlavaOnevision2ImageProcessorPil

[[autodoc]] LlavaOnevision2ImageProcessorPil

## LlavaOnevision2VideoProcessor

[[autodoc]] LlavaOnevision2VideoProcessor

## LlavaOnevision2Processor

[[autodoc]] LlavaOnevision2Processor

## LlavaOnevision2VisionModel

[[autodoc]] LlavaOnevision2VisionModel
    - forward

## LlavaOnevision2Model

[[autodoc]] LlavaOnevision2Model
    - forward
    - get_image_features

## LlavaOnevision2ForConditionalGeneration

[[autodoc]] LlavaOnevision2ForConditionalGeneration
    - forward
    - get_image_features
