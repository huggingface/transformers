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
*This model was released on 2021-12-02 and added to Hugging Face Transformers on 2023-01-16 and contributed by [shivi](https://huggingface.co/shivi) and [adirik](https://huggingface.co/adirik).*

# Mask2Former

[Mask2Former](https://huggingface.co/papers/2112.01527) is a unified framework for panoptic, instance, and semantic segmentation. It employs masked attention to extract localized features by constraining cross-attention within predicted mask regions, leading to significant performance improvements over specialized architectures. Mask2Former achieves state-of-the-art results, including 57.8 PQ on COCO for panoptic segmentation, 50.1 AP on COCO for instance segmentation, and 57.7 mIoU on ADE20K for semantic segmentation.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="facebook/mask2former-swin-large-coco-panoptic", dtype="auto")
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


processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = AutoModelForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic", dtype="auto")

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

## Usage tips

- Mask2Former uses the same preprocessing and postprocessing steps as MaskFormer. Use [`Mask2FormerImageProcessor`] or [`AutoImageProcessor`] to prepare images and optional targets for the model.
- To get the final segmentation, call [`post_process_semantic_segmentation`], [`post_process_instance_segmentation`], or [`post_process_panoptic_segmentation`] depending on the task. All three tasks work with [`Mask2FormerForUniversalSegmentation`] output.
- Panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target objects (like sky) together.

## Mask2FormerConfig

[[autodoc]] Mask2FormerConfig

## MaskFormer specific outputs

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerModelOutput

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput

## Mask2FormerModel

[[autodoc]] Mask2FormerModel
    - forward

## Mask2FormerForUniversalSegmentation

[[autodoc]] Mask2FormerForUniversalSegmentation
    - forward

## Mask2FormerImageProcessor

[[autodoc]] Mask2FormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## Mask2FormerImageProcessorFast

[[autodoc]] Mask2FormerImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

