<!--Copyright 2026 Perceptron, Inc. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-13.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Isaac

## Overview

Isaac is Perceptron's vision-language model (VLM) that pairs a SigLIP2 vision encoder with a Qwen3 decoder-only stack. The
Transformers implementation supports text-only and image-conditioned generation, including prompts with multiple interleaved
images. Isaac uses variable-resolution image preprocessing and can optionally reduce spatial tokens with pixel shuffle to keep
long multimodal prompts manageable. For more information, refer to the [technical report](https://github.com/perceptron-ai-inc/perceptron/blob/main/papers/isaac_01.pdf).

Isaac checkpoints are distributed under Perceptron's Non-Production license; please review the license that ships with the
weights before using them in commercial settings.

## Usage tips

- Batched inputs can mix text-only and multimodal samples. For direct processor/model batching, pass images as a nested
  list such as `[[], [image_a], [image_b, image_c]]`.
- `image_grid_thw[batch_idx, image_slot] == (0, 0, 0)` marks a padded empty slot. Real image slots have
  `(T=1, H>0, W>0)`.
- If truncation is enabled, the processor keeps the rightmost part of the multimodal prompt and updates the slot-local
  `image_metadata[..., 0]` and `image_metadata[..., 1]` values automatically.

## Usage example

Isaac uses explicit image placeholders in the rendered prompt. Every occurrence of `processor.image_token` (usually `<image>`) must have a matching image in the `images` argument.

```py
import torch
from PIL import Image
from transformers import AutoProcessor, IsaacForConditionalGeneration

model_id = "PerceptronAI/Isaac-0.1"
processor = AutoProcessor.from_pretrained(model_id)
model = IsaacForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Compare the two figures and explain what changed."},
            {"type": "image", "path": "first_image.png"},
            {"type": "image", "path": "second_image.png"},
            ],
    },
]

prompt = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False,)

generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Post-processing grounded outputs

Isaac can generate grounded points and boxes in tagged text spans. Use `post_process_generation()` to strip the tags and
recover structured annotations.

```py
clean_text, annotations = processor.post_process_generation(response, expected="box")
print(clean_text)
print(annotations)
```

Set `expected="point"` to extract point annotations, or leave `expected=None` to collect both points and boxes.

## IsaacVisionConfig

[[autodoc]] IsaacVisionConfig

## IsaacTextConfig

[[autodoc]] IsaacTextConfig

## IsaacConfig

[[autodoc]] IsaacConfig

## IsaacVisionModel

[[autodoc]] IsaacVisionModel

## IsaacTextModel

[[autodoc]] IsaacTextModel
    - forward

## IsaacModel

[[autodoc]] IsaacModel
    - forward

## IsaacForConditionalGeneration

[[autodoc]] IsaacForConditionalGeneration
    - forward

## IsaacProcessor

[[autodoc]] IsaacProcessor

## IsaacImageProcessor

[[autodoc]] IsaacImageProcessor
