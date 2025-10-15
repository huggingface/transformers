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
*This model was released on 2022-04-14 and added to Hugging Face Transformers on 2023-06-20 and contributed by [alihassanijr](https://huggingface.co/alihassanijr).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# Neighborhood Attention Transformer

[Neighborhood Attention Transformer](https://huggingface.co/papers/2204.07143) is a hierarchical vision transformer utilizing Neighborhood Attention, a sliding-window self-attention mechanism. This approach localizes self-attention to neighboring pixels, achieving linear time and space complexity. NATTEN, a Python package with efficient C++ and CUDA kernels, enhances NA's performance, making it up to 40% faster and using 25% less memory compared to Swin Transformer's Window Self Attention. NAT demonstrates competitive results in image classification, object detection, and semantic segmentation, outperforming similar-sized Swin models by 1.9% on ImageNet, 1.0% on MS-COCO, and 2.6% on ADE20K.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="shi-labs/nat-mini-in1k-224", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
model = AutoModelForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- Use the [`AutoImageProcessor`] API to prepare images for the model.
- NAT works as a backbone. When `output_hidden_states = True`, it outputs both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.
- NAT depends on NATTEN's implementation of Neighborhood Attention. Install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`. Building from source takes time to compile. NATTEN doesn't support Windows devices yet.
- Patch size of 4 is the only supported size at the moment.

## NatConfig

[[autodoc]] NatConfig

## NatModel

[[autodoc]] NatModel
    - forward

## NatForImageClassification

[[autodoc]] NatForImageClassification
    - forward

