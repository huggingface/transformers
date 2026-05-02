<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-15 and added to Hugging Face Transformers on 2026-05-02.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Molmo2

[Molmo2](https://huggingface.co/papers/2601.10611) is a family of open-weight vision-language models by AllenAI that are state-of-the-art among open-source models, with exceptional capabilities in point-driven grounding for single image, multi-image, and video tasks. The architecture combines a Vision Transformer (ViT) for image processing with an adapter layer connecting vision and text modalities, and a text decoder based on transformer architecture with rotary position embeddings.

The abstract from the paper is the following:

*Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).*

You can find all the original Molmo2 checkpoints under the [Molmo2](https://huggingface.co/collections/allenai/molmo2-67d6b5b0e138c5d621de1e5d) collection.

## Usage example

### Image-text-to-text generation

Here's how to use Molmo2 for image-text-to-text generation:

```python
from transformers import Molmo2ForConditionalGeneration, Molmo2Processor
import torch

processor = Molmo2Processor.from_pretrained("allenai/Molmo2-8B")
model = Molmo2ForConditionalGeneration.from_pretrained(
    "allenai/Molmo2-8B",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)
print(generated_text[0])
```

## Molmo2Config

[[autodoc]] Molmo2Config

## Molmo2VitConfig

[[autodoc]] Molmo2VitConfig

## Molmo2AdapterConfig

[[autodoc]] Molmo2AdapterConfig

## Molmo2TextConfig

[[autodoc]] Molmo2TextConfig

## Molmo2Processor

[[autodoc]] Molmo2Processor
    - __call__

## Molmo2ImageProcessor

[[autodoc]] Molmo2ImageProcessor
    - __call__
    - preprocess

## Molmo2VideoProcessor

[[autodoc]] Molmo2VideoProcessor
    - __call__

## Molmo2Model

[[autodoc]] Molmo2Model
    - forward

## Molmo2TextModel

[[autodoc]] Molmo2TextModel
    - forward

## Molmo2VisionBackbone

[[autodoc]] Molmo2VisionBackbone
    - forward

## Molmo2VisionModel

[[autodoc]] Molmo2VisionModel
    - forward

## Molmo2ForConditionalGeneration

[[autodoc]] Molmo2ForConditionalGeneration
    - forward
