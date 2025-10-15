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
*This model was released on 2021-11-11 and added to Hugging Face Transformers on 2022-01-18 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ViTMAE

[ViTMAE](https://huggingface.co/papers/2111.06377v2) demonstrates that masked autoencoders (MAE) are effective self-supervised learners for computer vision. The model uses an asymmetric encoder-decoder architecture, where the encoder processes only the visible patches without mask tokens, and the decoder reconstructs the image from the latent representation and mask tokens. Masking up to 75% of the input image creates a meaningful self-supervisory task, enabling efficient and effective training of large models. This approach leads to high accuracy, with a vanilla ViT-Huge model achieving 87.8% on ImageNet-1K, surpassing supervised pre-training in downstream tasks.

<hfoptions id="usage">
<hfoption id="ViTMAEForPreTraining">

```py
import torch
import requests
from PIL import Image
from transformers import infer_device, ViTImageProcessor, ViTMAEForPreTraining

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
inputs = processor(image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", dtype="auto")
with torch.no_grad():
    outputs = model(**inputs)

reconstruction = outputs.logits
```

</hfoption>
</hfoptions>

## Usage tips

- ViTMAE is typically used in two stages. First, do self-supervised pretraining with [`ViTMAEForPreTraining`]. Then discard the decoder and fine-tune the encoder. After fine-tuning, plug the weights into a model like [`ViTForImageClassification`].
- Use [`ViTImageProcessor`] for input preparation.

## ViTMAEConfig

[[autodoc]] ViTMAEConfig

## ViTMAEModel

[[autodoc]] ViTMAEModel
    - forward

## ViTMAEForPreTraining

[[autodoc]] transformers.ViTMAEForPreTraining
    - forward

