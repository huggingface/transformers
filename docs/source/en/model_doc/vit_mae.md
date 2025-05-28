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

You can find all the original ViTMAE checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=vit-mae) organization.

> [!TIP]
> Click on the ViTMAE models in the right sidebar for more examples of how to apply ViTMAE to vision tasks.

The example below demonstrates how to use ViTMAE with the [`AutoModel`] class.

<hfoptions id="usage">

<!-- This model is not currently supported via pipeline. -->

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import ViTImageProcessor, ViTMAEForPreTraining

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
inputs = processor(image, return_tensors="pt").to("cuda")

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

reconstruction = outputs.logits
```

</hfoption>
<hfoption id="transformers-cli">

<!-- This model is not currently supported via transformers-cli. -->

</hfoption>
</hfoptions>

## Notes
- ViTMAE is used in two stages: self-supervised pretraining with    `ViTMAEForPreTraining`, then finetuning using the encoder with `ViTForImageClassification`.
- The model reconstructs masked image patches (typically 75%) during pretraining and discards the decoder afterward.
- Only visible patches are encoded, and learned mask tokens are used in the decoder.
- Use `ViTImageProcessor` for input preparation.
- For faster inference on supported hardware, SDPA can be enabled via `attn_implementation="sdpa"`.

```python
from transformers import ViTMAEModel
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

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
