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

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-03-27 and added to Hugging Face Transformers on 2026-07-09.*

# SAM 3.1 Tracker Video

## Overview

[SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/) introduces **Object Multiplex**, a shared-memory multi-object video tracker.
`Sam31TrackerVideoModel` is the promptable visual segmentation (PVS) video tracker: points, boxes, and masks track specific object instances across frames.

Compared with [`Sam3TrackerVideoModel`], the 3.1 tracker processes up to 16 objects jointly per forward pass, improving multi-object throughput and crowded-scene tracking.

The abstract from the paper is the following:

*We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts...*

## Usage example

### Basic video tracking

```python
from transformers import Sam31TrackerVideoModel, Sam31TrackerVideoProcessor
import torch

model = Sam31TrackerVideoModel.from_pretrained("facebook/sam3.1")
processor = Sam31TrackerVideoProcessor.from_pretrained("facebook/sam3.1")
```

## Resources

- Meta SAM 3.1 release notes: [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- Model card: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)

## Sam31TrackerVideoConfig

[[autodoc]] Sam31TrackerVideoConfig

## Sam31TrackerVideoModel

[[autodoc]] Sam31TrackerVideoModel
    - forward
    - propagate_in_video_iterator

## Sam31TrackerVideoProcessor

[[autodoc]] Sam31TrackerVideoProcessor
