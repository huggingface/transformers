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


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->

*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-02.*

# Efficientvitsam

## Overview

EfficientViT-SAM is the EfficientViT-based Segment Anything Model introduced by MIT HAN Lab in [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](https://arxiv.org/abs/2205.14756) and released in the [`mit-han-lab/efficientvit`](https://github.com/mit-han-lab/efficientvit) repository.

This Transformers implementation follows the upstream MIT `efficientvit/applications/efficientvit_sam/` architecture: an EfficientViT image encoder paired with the SAM prompt encoder and mask decoder. It is compatible with the official `.pt` checkpoints such as `efficientvit_sam_l0.pt`.

## Usage example

```python
from PIL import Image
import requests

from transformers import EfficientvitsamModel, EfficientvitsamProcessor

image = Image.open(
    requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
).convert("RGB")

processor = EfficientvitsamProcessor.from_pretrained("mit-han-lab/efficientvit-sam")
model = EfficientvitsamModel.from_pretrained("mit-han-lab/efficientvit-sam")

inputs = processor(
    images=image,
    input_points=[[[320, 240]]],
    input_labels=[[1]],
    return_tensors="pt",
)

outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    original_sizes=inputs["original_sizes"],
    reshaped_input_sizes=inputs["reshaped_input_sizes"],
)
```

## Notes

- `EfficientvitsamImageProcessor` uses the same resize, normalization, and padding path as the upstream MIT predictor.
- `EfficientvitsamProcessor` scales prompts against the `1024` prompt canvas used by EfficientViT-SAM.
- The converted Hugging Face checkpoints preserve the official MIT module layout and weight semantics.

## EfficientvitsamConfig

[[autodoc]] EfficientvitsamConfig

## EfficientvitsamMaskDecoderConfig

[[autodoc]] EfficientvitsamMaskDecoderConfig

## EfficientvitsamPromptEncoderConfig

[[autodoc]] EfficientvitsamPromptEncoderConfig

## EfficientvitsamVisionConfig

[[autodoc]] EfficientvitsamVisionConfig

## EfficientvitsamVisionEncoderOutput

[[autodoc]] EfficientvitsamVisionEncoderOutput

## EfficientvitsamVisionModel

[[autodoc]] EfficientvitsamVisionModel
    - forward

## EfficientvitsamModel

[[autodoc]] EfficientvitsamModel
    - forward

## EfficientvitsamImageSegmentationOutput

[[autodoc]] EfficientvitsamImageSegmentationOutput

## EfficientvitsamPreTrainedModel

[[autodoc]] EfficientvitsamPreTrainedModel
    - forward

## EfficientvitsamImageProcessor

[[autodoc]] EfficientvitsamImageProcessor

## EfficientvitsamImageProcessorPil

[[autodoc]] EfficientvitsamImageProcessorPil

## EfficientvitsamProcessor

[[autodoc]] EfficientvitsamProcessor
