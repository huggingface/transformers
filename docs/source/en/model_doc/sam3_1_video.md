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

# SAM 3.1 Video

## Overview

`Sam31VideoModel` is the full SAM 3.1 **promptable concept segmentation (PCS)** video model: open-vocabulary text prompts detect, segment, and track all matching instances.
It composes a SAM3 detector (heads only) with the multiplex `Sam31TrackerVideoModel` sharing a TriNeck vision encoder.

Object Multiplex yields ~7× faster multi-object inference at 128 objects on H100 versus SAM 3, with improved VOS on 6/7 benchmarks.

## Usage example

```python
from transformers import Sam31VideoModel, Sam3VideoProcessor
import torch

model = Sam31VideoModel.from_pretrained("facebook/sam3.1")
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3.1")
```

## Resources

- [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
- [Meta release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

## Sam31VideoConfig

[[autodoc]] Sam31VideoConfig

## Sam31VideoModel

[[autodoc]] Sam31VideoModel
    - forward
    - propagate_in_video_iterator
