<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-06-06 and added to Hugging Face Transformers on 2023-06-02 and contributed by [shehan97](https://huggingface.co/shehan97).*

# MobileViTV2

[MobileViTV2](https://huggingface.co/papers/2206.02680) replaces multi-headed self-attention in MobileViT with separable self-attention, reducing time complexity from O(k²) to O(k) through element-wise operations. This enhancement improves efficiency and latency, making it suitable for resource-constrained devices. MobileViTV2 achieves state-of-the-art results in mobile vision tasks, including ImageNet classification and MS-COCO detection, with a top-1 accuracy of 75.6% and a 3.2× speed increase over MobileViT.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="apple/mobilevitv2-1.0", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0")
model = AutoModelForImageClassification.from_pretrained("apple/mobilevitv2-1.0", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- MobileViTV2 is more like a CNN than a Transformer model. It doesn't work on sequence data but on batches of images. Unlike ViT, there are no embeddings. The backbone model outputs a feature map.
- Use [`MobileViTImageProcessor`] to prepare images for the model. If doing custom preprocessing, the pretrained checkpoints expect images to be in BGR pixel order (not RGB).
- The available image classification checkpoints are pretrained on ImageNet-1k (ILSVRC 2012, 1.3 million images and 1,000 classes).
- The segmentation model uses a DeepLabV3 head. The available semantic segmentation checkpoints are pretrained on PASCAL VOC.

## MobileViTV2Config

[[autodoc]] MobileViTV2Config

## MobileViTV2Model

[[autodoc]] MobileViTV2Model
    - forward

## MobileViTV2ForImageClassification

[[autodoc]] MobileViTV2ForImageClassification
    - forward

## MobileViTV2ForSemanticSegmentation

[[autodoc]] MobileViTV2ForSemanticSegmentation
    - forward

