<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-02-23 and added to Hugging Face Transformers on 2024-07-08 and contributed by [nielsr](https://huggingface.co/nielsr).*

# ZoeDepth

[ZoeD-M12-NK](https://huggingface.co/papers/2302.12288) extends the DPT framework for metric depth estimation by combining relative and metric depth training. It is pre-trained on 12 datasets and fine-tuned on NYU and KITTI datasets using a lightweight head with a metric bins module. During inference, a latent classifier routes input images to the appropriate head. This model achieves state-of-the-art results on NYU Depth v2 and demonstrates zero-shot generalization to eight unseen datasets.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti", dtype="auto")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)],
)
predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
Image.fromarray(depth.astype("uint8"))
```

</hfoption>
</hfoptions>

## ZoeDepthConfig

[[autodoc]] ZoeDepthConfig

## ZoeDepthImageProcessor

[[autodoc]] ZoeDepthImageProcessor
    - preprocess

## ZoeDepthImageProcessorFast

[[autodoc]] ZoeDepthImageProcessorFast
    - preprocess

## ZoeDepthForDepthEstimation

[[autodoc]] ZoeDepthForDepthEstimation
    - forward

