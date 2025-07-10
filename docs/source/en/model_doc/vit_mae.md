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


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ViTMAE

[ViTMAE](https://huggingface.co/papers/2111.06377) is a self-supervised vision model that is pretrained by masking large portions of an image (~75%). An encoder processes the visible image patches and a decoder reconstructs the missing pixels from the encoded patches and mask tokens. After pretraining, the encoder can be reused for downstream tasks like image classification or object detection — often outperforming models trained with supervised learning.

<img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png"
alt="drawing" width="600"/> 

You can find all the original ViTMAE checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=vit-mae) organization.

> [!TIP]
> Click on the ViTMAE models in the right sidebar for more examples of how to apply ViTMAE to vision tasks.

The example below demonstrates how to reconstruct the missing pixels with the [`ViTMAEForPreTraining`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import ViTImageProcessor, ViTMAEForPreTraining

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
inputs = processor(image, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

reconstruction = outputs.logits
```

</hfoption>
</hfoptions>

## Notes
- ViTMAE is typically used in two stages. Self-supervised pretraining with [`ViTMAEForPreTraining`], and then discarding the decoder and fine-tuning the encoder. After fine-tuning, the weights can be plugged into a model like [`ViTForImageClassification`].
- Use [`ViTImageProcessor`] for input preparation.

## Resources

- Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb) to learn how to visualize the reconstructed pixels from [`ViTMAEForPreTraining`].

## ViTMAEConfig

[[autodoc]] ViTMAEConfig

<frameworkcontent>
<pt>

## ViTMAEModel

[[autodoc]] ViTMAEModel
    - forward

## ViTMAEForPreTraining

[[autodoc]] transformers.ViTMAEForPreTraining
    - forward

</pt>
<tf>

## TFViTMAEModel

[[autodoc]] TFViTMAEModel
    - call

## TFViTMAEForPreTraining

[[autodoc]] transformers.TFViTMAEForPreTraining
    - call

</tf>
</frameworkcontent>
