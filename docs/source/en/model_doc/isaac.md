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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-24.*

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

## Usage

Isaac uses explicit image placeholders in the rendered prompt. Every occurrence of `processor.image_token` (usually
`<image>`) must have a matching image in the `images` argument.

```py
import torch
from PIL import Image
from transformers import AutoProcessor, IsaacForConditionalGeneration

model_id = "Perceptron/isaac-base"
processor = AutoProcessor.from_pretrained(model_id)
model = IsaacForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

images = [Image.open("chart.png"), Image.open("panel.jpg")]
messages = [
    {"role": "user", "content": "Compare the two figures and explain what changed."},
    {"role": "user", "content": f"{processor.image_token}{processor.image_token}"},
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
).strip()

inputs = processor(text=prompt, images=images, return_tensors="pt")
model_inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
    if value is not None
}

with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

`IsaacProcessor` returns the standard text tensors plus Isaac's batch-major visual tensors:

- `pixel_values`: `(batch_size, max_images, max_patches, patch_dim)`
- `image_grid_thw`: `(batch_size, max_images, 3)`
- `image_metadata`: `(batch_size, max_images, 2)` storing `(offset, length)` for each image slot
- `mm_token_type_ids`: `(batch_size, sequence_length)`

Important notes:

- Pass the full processor output to `generate()`. Isaac uses the multimodal tensors during prefill and handles cached
  decoding internally.
- For fully text-only batches, `pixel_values`, `image_grid_thw`, and `image_metadata` are `None`. When moving inputs to
  the model, keep only non-`None` values as shown above.
- Batched inputs can mix text-only and multimodal samples. For direct processor/model batching, pass images as a nested
  list such as `[[], [image_a], [image_b, image_c]]`.
- `image_grid_thw[batch_idx, image_slot] == (0, 0, 0)` marks a padded empty slot. Real image slots have
  `(T=1, H>0, W>0)`.
- If truncation is enabled, the processor keeps the rightmost part of the multimodal prompt and updates the slot-local
  `image_metadata[..., 0]` and `image_metadata[..., 1]` values automatically.

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

## IsaacVisionTransformer

[[autodoc]] IsaacVisionTransformer

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
