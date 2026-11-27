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
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

LLaVA-OneVision-2 was introduced in [LLaVA-OneVision-2: Towards Next-Generation Perceptual Intelligence](https://huggingface.co/papers/2605.25979) by Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, Yifei Shen, Yuanhan Zhang, Kaichen Zhang, Wenkang Zhang, Zheng Cheng, Nansen Zhang, Chunsheng Wu, Chunjiang Ge, Zimin Ran, Dehua Song, Chunyuan Li, Shikun Feng, Ming Hu, Zhangquan Chen, Junbo Niu, Bo Li, Ziyong Feng, Ziwei Liu, Zongyuan Ge, and Jiankang Deng.

LLaVA-OneVision-2 is an open vision-language model for image understanding, long-video understanding, and spatial reasoning. It combines a native-resolution OneVision-Encoder, a lightweight vision-language connector, and an autoregressive text decoder. Images, sampled video frames, and codec-derived visual canvases share the same visual encoder. Codec-stream tokenization uses compression signals to select temporal and spatial evidence, allowing video input to concentrate its visual-token budget on relevant content.

The release includes training code, data resources, and recipes alongside the [8B Instruct checkpoint](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct). The paper studies video question answering, temporal grounding, spatial grounding, tracking, and manipulation-trace reasoning, and introduces JumpScore to evaluate fine-grained grounding in repeated motion. See the [paper](https://arxiv.org/abs/2605.25979) for task-specific protocols and evaluation results.

### Abstract

The paper's abstract can be summarized as follows:

*LLaVA-OneVision-2 explores a common representation for images, sampled video frames, and codec-derived visual evidence. Its native-resolution encoder uses spatiotemporal positions, while codec-stream tokenization allocates visual tokens according to compression signals and motion residuals. This makes it possible to retain relevant events over longer videos within a constrained token budget. The training recipe combines open image supervision, re-captioned videos, and spatial data. The work also introduces JumpScore for evaluating repeated-motion temporal grounding and studies image understanding, long-video reasoning, spatial grounding, and tracking. Models, datasets, and training resources are released to support reproducible multimodal research.*

<img src="https://arxiv.org/html/2605.25979v1/arch.png" alt="LLaVA-OneVision-2 architecture: codec videos, sampled frames, and images pass through the shared OneVision-Encoder, and the visual embeddings and text tokens are processed by an autoregressive decoder." width="1000"/>

<small>LLaVA-OneVision-2 architecture, reproduced without changes from Figure 2 of <a href="https://arxiv.org/html/2605.25979v1#S2.F2">An et al. (2026)</a>, under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Illustrated resolutions, frame counts, and compression ratios are examples rather than fixed processor settings.</small>

The original implementation, training resources, and model releases are available in the [official repository](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2).

## Usage

Use a Transformers build with native LLaVA-OneVision-2 support. The examples use `accelerate` for automatic device placement.

### Multiple-image quickstart

Pass image and text items in a conversation. `apply_chat_template()` loads the images, formats the prompt, and prepares model inputs.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {
                "type": "image",
                "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg",
            },
            {"type": "text", "text": "Describe the differences between these two images."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(generated_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

For text-only inference, include only text items. To batch independent conversations, set
`processor.tokenizer.padding_side = "left"`, pass a list of conversations, and set
`processor_kwargs={"padding": True}` in `apply_chat_template()`.

Set `attn_implementation="flash_attention_2"` when loading the model to use FlashAttention 2 on a compatible GPU with `flash-attn` installed.

### Video inference with sampled frames

Reuse the model and processor from the quickstart. The frame-sampling backend is the default. `num_frames` controls uniform sampling, and `size` sets the pixel-area bounds per frame. The processor keeps the source frame positions and adds timestamps to the prompt.

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
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={
        "videos_kwargs": {
            "num_frames": 16,
            "size": {"shortest_edge": 3136, "longest_edge": 200704},
        }
    },
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(generated_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

### Video inference with codec canvases

For codec preprocessing, install `codec-video-prep` and put `ffmpeg` on `PATH`. Pass the local path of an encoded video file. The backend builds visual canvases and retains each patch's source position for spatiotemporal encoding.

Reuse `video_path` and `messages` from the sampled-frame example. The codec backend uses `target_canvas` to set the requested canvas count and `size["longest_edge"]` to set the canvas pixel budget.

```python
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
```

Pass these inputs to `model.generate()` and decode the generated tokens as in the sampled-frame example. Short clips can produce fewer usable canvases than requested.

<img src="https://raw.githubusercontent.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/5837d88ac60bd922a38464938f9a24ebfc2a2e2f/asset/method_codec_selection_light.svg" alt="Illustration comparing dense uniform frame sampling with codec-aligned selection of dense I-frame patches and sparse P-frame patches." width="900"/>

<small>Codec patch selection from the <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/5837d88ac60bd922a38464938f9a24ebfc2a2e2f">official project</a>. The displayed token budget and temporal coverage are illustrative rather than fixed processor settings.</small>

### Processing notes

- The image processor accepts dynamic image sizes. Use `images_kwargs` or `videos_kwargs` to set pixel budgets for the corresponding modality.
- Native loading honors `video_preprocessor_config.json`. The original remote processor constructs its frame processor from image defaults, so use explicit sampling and pixel budgets when comparing implementations.
- The default image backend uses torchvision; [`LlavaOnevision2ImageProcessorPil`] is available for PIL preprocessing.
- Dynamic visual grid boundaries currently prevent full-graph compilation.


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
