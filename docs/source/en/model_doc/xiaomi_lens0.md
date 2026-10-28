<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-27.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Xiaomi-Lens-0

## Overview

[Xiaomi-Lens-0](https://huggingface.co/papers/2610.23279) is a 4B multimodal model from Xiaomi built on [Qwen3.5](./qwen3_5)-4B that grounds its answers at the pixel level. Besides text, it writes a `<|mask|>` token for every region it refers to, and a lightweight mask decoder turns the hidden state of each `<|mask|>` token into a segmentation mask. The same mechanism covers referring expression comprehension and segmentation, multi-instance grounding, reasoning segmentation, embodied referring, and fine-grained perception, where masks also appear in the reasoning trace (for example one per counted object).

The mask decoder reads a single feature grid from the vision encoder, before the spatial merger. The patch features go through one self-attention layer and two transposed convolutions, and are matched against an embedding of each `<|mask|>` hidden state. A final 4x bilinear upsampling brings the masks to the resolution of the resized image fed to the vision encoder.

The original code can be found [here](https://github.com/SeerRay-Lab/Xiaomi-Lens-0).

## Usage

Boxes are written as `<bbox>[x1, y1, x2, y2]</bbox>` with coordinates normalized to `[0, 1000]`, and masks as `<mask><|mask|></mask>`. A grounded instance is written as `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`. Generate text while collecting final-layer hidden states and image features, then call `decode_masks` to reuse those features without rerunning either backbone. [`~XiaomiLens0Processor.post_process_grounding`] extracts boxes and resizes masks to the original image. Decoding and postprocessing exclude thinking tokens by default.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "SeerRay-Lab/Xiaomi-Lens-0"
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {
                "type": "text",
                "text": 'Locate all instances in the image that match the description "cat".\nReport bbox coordinates and masks.',
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_hidden_states=[model.config.text_config.num_hidden_layers - 1],
        max_new_tokens=1024,
    )
    hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
    mask_logits, mask_token_indices = model.decode_masks(
        outputs.sequences, hidden_states, outputs.image_hidden_states,
        inputs.image_grid_thw, ignore_thinking=True,
    )
results = processor.post_process_grounding(
    outputs.sequences, inputs.image_grid_thw, mask_logits, mask_token_indices, target_sizes=[(480, 640)]
)
print(results[0]["boxes"], results[0]["masks"].shape)
```

Set `return_dict_in_generate=True` and `output_hidden_states=[model.config.text_config.num_hidden_layers - 1]`
to retain the image features and final-layer hidden states needed by `decode_masks`. These options are opt-in;
ordinary generation returns token IDs.

Tips:

- Ask for boxes and masks explicitly with "Report bbox coordinates and masks." at the end of the instruction. Without it, the model answers in plain text.
- The mask decoder sees the image at the resolution chosen by the image processor (up to `max_pixels`). Masks are predicted at that resolution and [`~XiaomiLens0Processor.post_process_grounding`] resizes them to `target_sizes`.

## Training with mask supervision

Pass a full teacher-forced sequence containing the target `<|mask|>` tokens to `forward`, together with
`mask_labels`. Image features and mask-token states retain their gradients, so mask supervision trains
the mask decoder and both backbones in one forward pass.

`mask_labels` is a list in input-image order. Each entry is a binary tensor of shape
`(num_masks, 16 * grid_height, 16 * grid_width)`, aligned with that image's mask tokens in sequence order,
including masks inside thinking blocks. Resize targets to that resolution before calling the model.
For an image with no mask tokens, supply an empty tensor with the same spatial dimensions.
Optional `mask_labels_valid_mask` tensors of matching shapes exclude padded pixels or unsupervised mask tokens;
an all-false mask contributes neither loss nor a mask count.

```python
outputs = model(
    **training_inputs,  # Full input_ids, attention_mask, pixels, grids, and modality IDs.
    labels=language_labels,  # Use -100 for prompt/padding tokens.
    mask_labels=mask_targets,
    use_cache=False,
)
outputs.loss.backward()
```

The loss follows the [original Xiaomi-Lens-0 mask objective](https://github.com/SeerRay-Lab/Xiaomi-Lens-0/blob/main/src/xiaomi_lens/model/losses.py):
Dice loss plus twice the pixel-wise sigmoid binary cross-entropy, averaged over valid masks.
Mask counts are reduced across distributed ranks. Every rank must supply language labels or mask targets.
Ranks with language labels but no mask targets participate with a zero mask count. When some ranks have no
mask supervision, configure the distributed wrapper to handle unused parameters (for DDP,
`find_unused_parameters=True`).
Gradient accumulation averages microbatch losses. Without `mask_labels`, only the language loss is computed;
the mask decoder is skipped and receives no gradients.
The configuration exposes `mask_loss_weight`, `mask_dice_loss_weight`, and `mask_bce_loss_weight`
(defaults: 1, 1, 2). Outputs include the combined `loss` and weighted `mask_loss`,
as well as `mask_logits` and `mask_token_indices`. The `mask_loss_chunk_size` configuration value (default: 8)
sets how many masks per image are processed at a time when computing the loss. Omit language `labels`
for mask-only supervision.

Without mask targets, ordinary generation remains unchanged. For full-sequence mask inference through
`forward`, set `output_masks=True`; incremental cached decoding should use the post-generation
`decode_masks` workflow above.

## XiaomiLens0Config

[[autodoc]] XiaomiLens0Config

## XiaomiLens0TextConfig

[[autodoc]] XiaomiLens0TextConfig

## XiaomiLens0VisionConfig

[[autodoc]] XiaomiLens0VisionConfig

## XiaomiLens0MaskDecoderConfig

[[autodoc]] XiaomiLens0MaskDecoderConfig

## XiaomiLens0Processor

[[autodoc]] XiaomiLens0Processor
    - post_process_grounding

## XiaomiLens0PreTrainedModel

[[autodoc]] XiaomiLens0PreTrainedModel

## XiaomiLens0VisionModel

[[autodoc]] XiaomiLens0VisionModel
    - forward
    - decode_masks

## XiaomiLens0TextModel

[[autodoc]] XiaomiLens0TextModel
    - forward
    - decode_masks

## XiaomiLens0Model

[[autodoc]] XiaomiLens0Model
    - forward
    - decode_masks

## XiaomiLens0ForConditionalGeneration

[[autodoc]] XiaomiLens0ForConditionalGeneration
    - forward
    - decode_masks

## XiaomiLens0GenerateOutput

[[autodoc]] XiaomiLens0GenerateOutput
