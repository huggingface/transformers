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
*This model was released on 2021-07-13 and added to Hugging Face Transformers on 2022-03-02 and contributed by [francesco](https://huggingface.co/francesco).*

# MaskFormer

[MaskFormer](https://huggingface.co/papers/2107.06278) addresses semantic and instance-level segmentation by employing a mask classification paradigm, eliminating the need for separate approaches. It predicts a set of binary masks, each linked to a global class label, using a unified model, loss, and training procedure. This method simplifies segmentation tasks and achieves superior results, particularly in scenarios with a large number of classes. MaskFormer outperforms per-pixel classification baselines, achieving 55.6 mIoU on ADE20K for semantic segmentation and 52.7 PQ on COCO for panoptic segmentation.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="facebook/maskformer-swin-base-coco", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor


processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = AutoModelForUniversalSegmentation.from_pretrained("facebook/maskformer-swin-base-coco", dtype="auto")

image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw)
inputs = processor(images=image, return_tensors="pt",)

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = [(image.height, image.width)]
outputs = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

plt.imshow(outputs[0]["segmentation"])
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>

## MaskFormer specific outputs

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerModelOutput

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput

## MaskFormerConfig

[[autodoc]] MaskFormerConfig

## MaskFormerImageProcessor

[[autodoc]] MaskFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerImageProcessorFast

[[autodoc]] MaskFormerImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerFeatureExtractor

[[autodoc]] MaskFormerFeatureExtractor
    - __call__
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## MaskFormerModel

[[autodoc]] MaskFormerModel
    - forward

## MaskFormerForInstanceSegmentation

[[autodoc]] MaskFormerForInstanceSegmentation
    - forward

