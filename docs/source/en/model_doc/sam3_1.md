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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-28.*
# SAM 3.1

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

## Overview

SAM 3.1 is the public checkpoint refresh that accompanies Meta's multiplex video-tracking release. In Transformers, image support for SAM 3.1 reuses the existing [`Sam3Model`] detector stack: we extract the detector portion of the public `facebook/sam3.1` checkpoint and load it into the standard SAM 3 image model classes.

This means:

- use [`Sam3Model`] and [`Sam3Processor`] for image inference;
- convert the public `sam3.1_multiplex.pt` checkpoint before loading it with Transformers;
- tracker-only multiplex tensors are ignored on the image path.

<div class="warning">
The public [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) repository currently ships the original `sam3.1_multiplex.pt` checkpoint, not a ready-to-load Transformers weights file. Convert it first with the provided script.
</div>

## Convert the checkpoint

```bash
python src/transformers/models/sam3/convert_sam3_to_hf.py \
  --checkpoint_path /path/to/sam3.1_multiplex.pt \
  --output_path /path/to/sam3.1-image-hf
```

The converter auto-detects the merged SAM 3.1 checkpoint, extracts the detector weights, saves a Transformers [`Sam3Model`] repository, writes the matching processor/tokenizer files alongside it, and verifies the saved checkpoint by default. Pass `--no_verify` only if you explicitly want to skip that save/load forward check.

## Usage examples with 🤗 Transformers

### Text-only prompt

```python
>>> from transformers import Sam3Model, Sam3Processor
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> converted_path = "/path/to/sam3.1-image-hf"

>>> model = Sam3Model.from_pretrained(converted_path).to(device)
>>> processor = Sam3Processor.from_pretrained(converted_path)

>>> image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs["original_sizes"].tolist(),
... )[0]

>>> print(outputs.pred_masks.shape)
torch.Size([1, 200, 288, 288])
>>> print(results["boxes"].shape)
torch.Size([num_detections, 4])
```

### Bounding-box prompt

```python
>>> box_xyxy = [100, 150, 500, 450]
>>> inputs = processor(
...     images=image,
...     input_boxes=[[box_xyxy]],
...     input_boxes_labels=[[1]],
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs["original_sizes"].tolist(),
... )[0]
```

## Sam3Config

[[autodoc]] Sam3Config

## Sam3Processor

[[autodoc]] Sam3Processor
    - __call__
    - post_process_instance_segmentation

## Sam3Model

[[autodoc]] Sam3Model
    - forward
    - get_text_features
    - get_vision_features
