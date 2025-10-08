<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2018-07-26 and added to Hugging Face Transformers on 2023-01-16 and contributed by [nielsr](https://huggingface.co/nielsr).*

# UPerNet

[UPerNet](https://huggingface.co/papers/1807.10221) is a multi-task framework designed for Unified Perceptual Parsing, enabling machines to recognize a wide array of visual concepts from images. It leverages various vision backbones such as ConvNeXt and Swin. The framework is trained using heterogeneous image annotations and demonstrates effective segmentation across a broad range of concepts, facilitating the discovery of visual knowledge in natural scenes.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="openmmlab/upernet-convnext-tiny", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor


processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny", dtype="auto")

image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw)
inputs = processor(images=image, return_tensors="pt",)

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = [(image.height, image.width)]
outputs = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

plt.imshow(outputs[0])
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>

## UperNetConfig

[[autodoc]] UperNetConfig

## UperNetForSemanticSegmentation

[[autodoc]] UperNetForSemanticSegmentation
    - forward

