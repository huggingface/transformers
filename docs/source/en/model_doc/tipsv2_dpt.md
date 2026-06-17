<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-04-13 and contributed to Hugging Face Transformers on 2026-06-17.*

# TIPSv2 DPT

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

TIPSv2 (Text-Image Pre-training with Spatial awareness) is a family of
contrastive vision-language encoders proposed in [TIPSv2: Advancing Vision-Language Pretraining with Enhanced
Patch-Text Alignment]((https://huggingface.co/papers/2604.12012)) by Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, Joshua Ainslie, Alex Bewley, Mithun Jacob, René Wagner, Washington Ramos, Krzysztof Choromanski, Mojtaba Seyedhosseini, Howard Zhou, André Araujo.

The abstract from the paper is the following:

*Recent progress in vision-language pretraining has enabled significant improvements to many downstream computer vision applications, such as classification, retrieval, segmentation and depth prediction. However, a fundamental capability that these models still struggle with is aligning dense patch representations with text embeddings of corresponding concepts. In this work, we investigate this critical issue and propose novel techniques to enhance this capability in foundational vision-language models. First, we reveal that a patch-level distillation procedure significantly boosts dense patch-text alignment – surprisingly, the patch-text alignment of the distilled student model strongly surpasses that of the teacher model. This observation inspires us to consider modifications to pretraining recipes, leading us to propose iBOT++, an upgrade to the commonly-used iBOT masked image objective, where unmasked tokens also contribute directly to the loss. This dramatically enhances patch-text alignment of pretrained models. Additionally, to improve vision-language pretraining efficiency and effectiveness, we modify the exponential moving average setup in the learning recipe, and introduce a caption sampling strategy to benefit from synthetic captions at different granularities. Combining these components, we develop TIPSv2, a new family of image-text encoder models suitable for a wide range of downstream applications. Through comprehensive experiments on 9 tasks and 20 datasets, we demonstrate strong performance, generally on par with or better than recent vision encoder models. Code and models are released via our project page at https://gdm-tipsv2.github.io/.*

This model was contributed by [Guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/google-deepmind/tips).

You can find all the original TIPSv2 DPT checkpoints under the [TIPSv2](https://huggingface.co/collections/google/tipsv2) collection.

> [!TIP]
> See [TIPSv2](./tipsv2) for the TIPSv2 vision and language backbones as well as zero shot image classification.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(task="depth-estimation", model="google/tipsv2-dpt-b14", device_map="auto")
out = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
out["depth"]  # PIL Image normalized to [min, max] of predicted_depth
print(out["predicted_depth"].shape)  # (height, width)
```

```python
from transformers import pipeline

pipe = pipeline(task="image-segmentation", model="google/tipsv2-dpt-b14-ade", device_map="auto")
out = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
out[0]["label"] # string class label
out[0]["mask"]  # PIL Image with binary mask set to class id
```

</hfoption>
<hfoption id="AutoModel">

Use [`Tipsv2DptModel`] to run all three tasks (depth, normals, and segmentation) in a single forward pass over a shared backbone.

```python
import torch
from transformers import AutoModel, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-dpt-b14"
model = AutoModel.from_pretrained(model_id, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

outputs.predicted_depth # (batch_size, height, width) tensor with predicted depth in meters
outputs.normals # (batch_size, 3, height, width) tensor with normals in XYZ format (unnormalized)
outputs.segmentation_logits # # (batch_size, config.num_labels, height, width) tensor with segmentation logits

depth_results = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
normal_results = image_processor.post_process_normal_estimation(outputs, target_sizes=[(image.height, image.width)])
segmentation_results = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])

predicted_depth = depth_results[0]["predicted_depth"]  # (height, width) tensor with predicted depth in meters
normals = normal_results[0]["normals"] # (3, height, width) tensor with normals in XYZ format (L2-normalized)
segmentation = segmentation_results[0] # (height, width) tensor with class ids
```

</hfoption>
<hfoption id="Depth estimation">

```python
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-dpt-b14"
model = AutoModelForDepthEstimation.from_pretrained(model_id, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
predicted_depth = results[0]["predicted_depth"] # (height, width) tensor with predicted depth in meters
```

</hfoption>
<hfoption id="Normal estimation">

```python
import torch
from transformers import Tipsv2DptForNormalEstimation, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-dpt-b14"
model = Tipsv2DptForNormalEstimation.from_pretrained(model_id, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_normal_estimation(outputs, target_sizes=[(image.height, image.width)])
normals = results[0]["normals"]  # (3, height, width) tensor with normals in XYZ format (L2-normalized)
```

</hfoption>
<hfoption id="Semantic segmentation">

```python
import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from transformers.utils import load_image


model_id = "google/tipsv2-dpt-b14-ade"
model = AutoModelForSemanticSegmentation.from_pretrained(model_id, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_id)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])
segmentation_map = results[0]  # (height, width) tensor with class ids
```

</hfoption>
</hfoptions>

## Notes

- [`Tipsv2DptModel`] runs a single shared backbone forward pass and produces three outputs simultaneously: `predicted_depth`, `normals`, and `segmentation_logits`. Use it when you need all three tasks for the same image.
- [`Tipsv2DptForDepthEstimation`], [`Tipsv2DptForNormalEstimation`], and [`Tipsv2DptForSemanticSegmentation`] are single-task variants that discard the other heads. Use them for fine-tuning or inference on a single task.
- Call [`~Tipsv2DptImageProcessor.post_process_depth_estimation`], [`~Tipsv2DptImageProcessor.post_process_normal_estimation`], or [`~Tipsv2DptImageProcessor.post_process_semantic_segmentation`] to resize outputs back to the original image resolution. Pass `target_sizes=[(image.height, image.width)]` (height, width) for a single image.
- `post_process_normal_estimation` applies L2 normalization along the channel dimension, yielding unit surface normals in `[-1, 1]` per channel (XYZ convention).
- `predicted_depth` from [`Tipsv2DptModel`] and [`Tipsv2DptForDepthEstimation`] is already in metric scale (meters) — no further rescaling is needed.


## Tipsv2DptConfig

[[autodoc]] Tipsv2DptConfig

## Tipsv2DptImageProcessor

[[autodoc]] Tipsv2DptImageProcessor
    - preprocess
    - post_process_depth_estimation
    - post_process_normal_estimation
    - post_process_semantic_segmentation

## Tipsv2DptModel

[[autodoc]] Tipsv2DptModel
    - forward

## Tipsv2DptForDepthEstimation

[[autodoc]] Tipsv2DptForDepthEstimation
    - forward

## Tipsv2DptForNormalEstimation

[[autodoc]] Tipsv2DptForNormalEstimation
    - forward

## Tipsv2DptForSemanticSegmentation

[[autodoc]] Tipsv2DptForSemanticSegmentation
    - forward
