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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-30.*
# SAM 3.1 Video

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

## Overview

SAM 3.1 introduces Meta's multiplex video-tracking architecture. In Transformers this support is exposed through [`Sam3_1VideoModel`], which loads the public `sam3.1_multiplex.pt` checkpoint into a dedicated multiplex video model implementation.

Current Transformers support focuses on the public multiplex tracker checkpoint itself:

- [`Sam3_1VideoModel`] provides prompted single-frame interactive decoding and multiplex propagation-head outputs;
- the conversion script verifies parity against the upstream SAM 3.1 implementation by default;
- the high-level `Sam3VideoProcessor` session API from [`Sam3VideoModel`] is not yet available for SAM 3.1.

<div class="warning">
The public [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) repository currently ships the original `sam3.1_multiplex.pt` checkpoint. Convert it first with the provided script, then load the saved directory with [`Sam3_1VideoModel`].
</div>

## Convert the checkpoint

```bash
python src/transformers/models/sam3_1_video/convert_sam3_1_video_to_hf.py \
  --checkpoint_path /path/to/sam3.1_multiplex.pt \
  --output_dir /path/to/sam3.1-video-hf
```

The converter remaps the multiplex checkpoint into [`Sam3_1VideoModel`] and runs an upstream parity check by default. Pass `--no_verify` only if you explicitly want to skip that verification step.

## Usage examples with 🤗 Transformers

### Point-prompted frame inference

```python
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import Sam3ImageProcessor, Sam3_1VideoModel

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> converted_path = "/path/to/sam3.1-video-hf"

>>> model = Sam3_1VideoModel.from_pretrained(converted_path).to(device)
>>> image_processor = Sam3ImageProcessor(size={"height": 1008, "width": 1008})

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
>>> pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

>>> input_points = torch.tensor([[[[500.0, 375.0]]]], device=device)
>>> input_labels = torch.tensor([[[1]]], device=device)

>>> with torch.no_grad():
...     outputs = model(
...         pixel_values=pixel_values,
...         input_points=input_points,
...         input_labels=input_labels,
...         run_propagation_head=True,
...     )

>>> print(outputs.interactive_pred_masks.shape)
torch.Size([1, 1, 288, 288])
>>> print(outputs.interactive_high_res_masks.shape)
torch.Size([1, 1, 1008, 1008])
>>> print(outputs.propagation_masks.shape)
torch.Size([1, 1, 3, 288, 288])
```

### Box prompt

```python
>>> input_boxes = torch.tensor([[[150.0, 250.0, 900.0, 760.0]]], device=device)

>>> with torch.no_grad():
...     outputs = model(
...         pixel_values=pixel_values,
...         input_boxes=input_boxes,
...         multimask_output=False,
...     )

>>> print(outputs.interactive_pred_masks.shape)
torch.Size([1, 1, 288, 288])
>>> print(outputs.interactive_iou_scores.shape)
torch.Size([1, 1, 1])
```

## Sam3_1VideoConfig

[[autodoc]] Sam3_1VideoConfig

## Sam3_1VideoModel

[[autodoc]] Sam3_1VideoModel
    - forward

## Sam3_1ViTModel

[[autodoc]] Sam3_1ViTModel
    - forward
