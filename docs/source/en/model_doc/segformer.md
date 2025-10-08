<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-05-31 and added to Hugging Face Transformers on 2021-10-28 and contributed by [nielsr](https://huggingface.co/nielsr).*

# SegFormer

[SegFormer](https://huggingface.co/papers/2105.15203) unifies Transformers with lightweight MLP decoders for semantic segmentation. It features a hierarchically structured Transformer encoder that outputs multiscale features without positional encoding, and a simple MLP decoder that combines local and global attention. This design achieves superior performance and efficiency, with SegFormer-B4 reaching 50.3% mIoU on ADE20K and SegFormer-B5 achieving 84.0% mIoU on Cityscapes, demonstrating excellent zero-shot robustness.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor


processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", dtype="auto")

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

## SegformerConfig

[[autodoc]] SegformerConfig

## SegformerImageProcessor

[[autodoc]] SegformerImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## SegformerImageProcessorFast

[[autodoc]] SegformerImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## SegformerModel

[[autodoc]] SegformerModel
    - forward

## SegformerDecodeHead

[[autodoc]] SegformerDecodeHead
    - forward

## SegformerForImageClassification

[[autodoc]] SegformerForImageClassification
    - forward

## SegformerForSemanticSegmentation

[[autodoc]] SegformerForSemanticSegmentation
    - forward

