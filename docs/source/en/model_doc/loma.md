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
*This model was contributed to Hugging Face Transformers on 2026-07-28.*


# LoMa

## Overview

LoMa was proposed in [LoMa: Local Feature Matching Revisited](https://huggingface.co/papers/2604.04931) by David Nordström,
Johan Edstedt, Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten Wadenbäck, Michael Felsberg,
and Fredrik Kahl. It is a local feature matcher that refines local descriptors with alternating self- and
cross-attention before selecting mutual matches with a dual-softmax score matrix.

This integration pairs LoMa with the native SuperPoint keypoint detector and LoMa's local descriptor network. The
matching transformer uses learnable Fourier positional encoding for self-attention and leaves cross-attention
position-free, matching the reference architecture. The published LoMa-B checkpoint is available as
[`Falcon7211/loma-b`](https://huggingface.co/Falcon7211/loma-b). The reference repository uses DaD and DeDoDe, so
end-to-end keypoint and descriptor outputs differ from those original frontends.

The original code is available in the [LoMa repository](https://github.com/davnords/LoMa).

## Usage examples

```python
import requests
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForKeypointMatching

checkpoint = "Falcon7211/loma-b"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForKeypointMatching.from_pretrained(checkpoint, dtype="auto", device_map="auto").eval()

url_0 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in (url_0, url_1)]

inputs = image_processor(images=images, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

image_sizes = [[(image.height, image.width) for image in images]]
matches = image_processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
print(matches[0])
```

## LoMaConfig

[[autodoc]] LoMaConfig

## LoMaPreTrainedModel

[[autodoc]] LoMaPreTrainedModel
    - forward

## LoMaForKeypointMatching

[[autodoc]] LoMaForKeypointMatching

## LoMaImageProcessor

[[autodoc]] LoMaImageProcessor
