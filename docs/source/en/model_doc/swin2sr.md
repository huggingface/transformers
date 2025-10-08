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
*This model was released on 2022-09-22 and added to Hugging Face Transformers on 2022-12-16 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Swin2SR

[Swin2SR](https://huggingface.co/papers/2209.11345) enhances SwinIR by integrating Swin Transformer v2 layers, addressing training instability, resolution discrepancies between pre-training and fine-tuning, and data scarcity. This model excels in JPEG compression artifact removal, classical and lightweight image super-resolution, and compressed image super-resolution, achieving top-5 performance in the AIM 2022 Challenge on Super-Resolution of Compressed Image and Video.

<hfoptions id="usage">
<hfoption id="Swin2SRForImageSuperResolution">

```py
import torch
import numpy as np
import requests
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.moveaxis(output, source=0, destination=-1)
output = (output * 255.0).round().astype(np.uint8)
Image.fromarray(output)
```

</hfoption>
</hfoptions>

## Swin2SRImageProcessor

[[autodoc]] Swin2SRImageProcessor
    - preprocess

## Swin2SRImageProcessorFast

[[autodoc]] Swin2SRImageProcessorFast
    - preprocess

## Swin2SRConfig

[[autodoc]] Swin2SRConfig

## Swin2SRModel

[[autodoc]] Swin2SRModel
    - forward

## Swin2SRForImageSuperResolution

[[autodoc]] Swin2SRForImageSuperResolution
    - forward

