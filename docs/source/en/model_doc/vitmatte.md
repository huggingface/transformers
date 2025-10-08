<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-05-24 and added to Hugging Face Transformers on 2023-09-19 and contributed by [nielsr](https://huggingface.co/nielsr).*

# ViTMatte

[ViTMatte](https://huggingface.co/papers/2305.15272) leverages plain Vision Transformers for image matting, combining a hybrid attention mechanism with a convolution neck to balance performance and computation. It also includes a detail capture module using lightweight convolutions to enhance detail accuracy. ViTMatte achieves state-of-the-art results on Composition-1k and Distinctions-646 benchmarks, outperforming existing methods significantly.

<hfoptions id="usage">
<hfoption id="VitMatteForImageMatting">

```py
import torch
from transformers import VitMatteImageProcessor, VitMatteForImageMatting
from PIL import Image
from huggingface_hub import hf_hub_download

processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k", dtype="auto")

filepath = hf_hub_download(
    repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
)
image = Image.open(filepath).convert("RGB")
filepath = hf_hub_download(
    repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
)
trimap = Image.open(filepath).convert("L")
inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

with torch.no_grad():
    alphas = model(**inputs).alphas
print(alphas.shape)
```

</hfoption>
</hfoptions>

## VitMatteConfig

[[autodoc]] VitMatteConfig

## VitMatteImageProcessor

[[autodoc]] VitMatteImageProcessor
    - preprocess

## VitMatteImageProcessorFast

[[autodoc]] VitMatteImageProcessorFast
    - preprocess

## VitMatteForImageMatting

[[autodoc]] VitMatteForImageMatting
    - forward

