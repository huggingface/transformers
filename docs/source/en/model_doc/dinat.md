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
*This model was released on 2022-09-29 and added to Hugging Face Transformers on 2022-11-18 and contributed by [alihassanijr](https://huggingface.co/alihassanijr).*

# Dilated Neighborhood Attention Transformer

[Dilated Neighborhood Attention Transformer](https://huggingface.co/papers/2209.15001) extends Neighborhood Attention (NA) by incorporating a Dilated Neighborhood Attention (DiNA) pattern, enhancing global context capture without additional computational cost. DiNAT combines local attention from NA with DiNA's sparse global attention, leading to significant performance improvements over models like NAT, Swin, and ConvNeXt. The large DiNAT variant achieves state-of-the-art results in various vision tasks, including COCO object detection, COCO instance segmentation, ADE20K semantic segmentation, and panoptic segmentation on both COCO and ADE20K datasets.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="shi-labs/dinat-mini-in1k-224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
model = AutoModelForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- DiNAT works as a backbone. When `output_hidden_states = True`, it outputs both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.
- DiNAT depends on NATTEN's implementation of Neighborhood Attention and Dilated Neighborhood Attention. Install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`. Building from source takes time to compile. NATTEN doesn't support Windows devices yet.
- Patch size of 4 is the only supported size at the moment.

## DinatConfig

[[autodoc]] DinatConfig

## DinatModel

[[autodoc]] DinatModel
    - forward

## DinatForImageClassification

[[autodoc]] DinatForImageClassification
    - forward

