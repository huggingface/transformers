<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-11-19 and added to Hugging Face Transformers on 2025-11-19.*

# SAM3

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

## Overview

SAM3 (Segment Anything Model 3) was introduced in [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).

SAM3 performs **Promptable Concept Segmentation (PCS)** on images. PCS takes text and/or image exemplars as input (e.g., "yellow school bus"), and predicts instance and semantic masks for **every single object** matching the concept.

The abstract from the paper is the following:

*We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation.*

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan) and [ronghanghu](https://huggingface.co/ronghanghu).

## Usage examples with 🤗 Transformers

### Text-Only Prompts

```python
>>> from transformers import Sam3Processor, Sam3Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")

>>> # Load image
>>> image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> # Segment using text prompt
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]

>>> print(f"Found {len(results['masks'])} objects")
>>> # Results contain:
>>> # - masks: Binary masks resized to original image size
>>> # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
>>> # - scores: Confidence scores
```

### Single Bounding Box Prompt

Segment objects using a bounding box on the visual concept:

```python
>>> # Box in xyxy format: [x1, y1, x2, y2] in pixel coordinates
>>> # Example: laptop region
>>> box_xyxy = [100, 150, 500, 450]
>>> input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
>>> input_boxes_labels = [[1]]  # 1 = positive box

>>> inputs = processor(
...     images=image,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
```

### Multiple Box Prompts (Positive and Negative)

Use multiple boxes with positive and negative labels to refine the concept:

```python
>>> # Load kitchen image
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

>>> # Define two positive boxes (e.g., dial and button on oven)
>>> # Boxes are in xyxy format [x1, y1, x2, y2] in pixel coordinates
>>> box1_xyxy = [59, 144, 76, 163]  # Dial box
>>> box2_xyxy = [87, 148, 104, 159]  # Button box
>>> input_boxes = [[box1_xyxy, box2_xyxy]]
>>> input_boxes_labels = [[1, 1]]  # Both positive

>>> inputs = processor(
...     images=kitchen_image,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
```

### Combined Prompts (Text + Negative Box)

Use text prompts with negative visual prompts to refine the concept:

```python
>>> # Segment "handle" but exclude the oven handle using a negative box
>>> text = "handle"
>>> # Negative box covering oven handle area (xyxy): [40, 183, 318, 204]
>>> oven_handle_box = [40, 183, 318, 204]
>>> input_boxes = [[oven_handle_box]]

>>> inputs = processor(
...     images=kitchen_image,
...     text=text,
...     input_boxes=input_boxes,
...     input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]
>>> # This will segment pot handles but exclude the oven handle
```

### Batched Inference with Text Prompts

Process multiple images with different text prompts efficiently:

```python
>>> cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> images = [
...     Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
...     Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
... ]

>>> # Different text prompt for each image
>>> text_prompts = ["ear", "dial"]

>>> inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results for both images
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )

>>> print(f"Image 1: {len(results[0]['masks'])} objects found")
>>> print(f"Image 2: {len(results[1]['masks'])} objects found")
```

### Batched Mixed Prompts

Use different prompt types for different images in the same batch:

```python
>>> # Image 1: text prompt "laptop"
>>> # Image 2: visual prompt (dial box)
>>> box2_xyxy = [59, 144, 76, 163]

>>> inputs = processor(
...     images=images,
...     text=["laptop", None],  # Only first image has text
...     input_boxes=[None, [box2_xyxy]],  # Only second image has box
...     input_boxes_labels=[None, [1]],  # Positive box for second image
...     return_tensors="pt"
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results for both images
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )
>>> # Both images processed in single forward pass
```

### Semantic Segmentation Output

SAM3 also provides semantic segmentation alongside instance masks:

```python
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Instance segmentation masks
>>> instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

>>> # Semantic segmentation (single channel)
>>> semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

>>> print(f"Instance masks: {instance_masks.shape}")
>>> print(f"Semantic segmentation: {semantic_seg.shape}")
```

### Efficient Multi-Prompt Inference on Single Image

When running multiple text prompts on the same image, pre-compute vision embeddings to avoid redundant computation:

```python
>>> from transformers import Sam3Processor, Sam3Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")

>>> # Load image
>>> image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> # Pre-process image and compute vision embeddings once
>>> img_inputs = processor(images=image, return_tensors="pt").to(device)
>>> with torch.no_grad():
...     vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

>>> # Run multiple text prompts efficiently
>>> text_prompts = ["ear", "eye", "nose"]
>>> all_results = []

>>> for prompt in text_prompts:
...     text_inputs = processor(text=prompt, return_tensors="pt").to(device)
...     with torch.no_grad():
...         outputs = model(vision_embeds=vision_embeds, **text_inputs)
...
...     results = processor.post_process_instance_segmentation(
...         outputs,
...         threshold=0.5,
...         mask_threshold=0.5,
...         target_sizes=img_inputs.get("original_sizes").tolist()
...     )[0]
...     all_results.append({"prompt": prompt, "results": results})

>>> for item in all_results:
...     print(f"Prompt '{item['prompt']}': {len(item['results']['masks'])} objects found")
```

### Efficient Single-Prompt Inference on Multiple Images

When running the same text prompt on multiple images, pre-compute text embeddings to avoid redundant computation:

```python
>>> from transformers import Sam3Processor, Sam3Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")

>>> # Pre-compute text embeddings once
>>> text_prompt = "ear"
>>> text_inputs = processor(text=text_prompt, return_tensors="pt").to(device)
>>> with torch.no_grad():
...     text_embeds = model.get_text_features(**text_inputs)

>>> # Load multiple images
>>> image_urls = [
...     "http://images.cocodataset.org/val2017/000000077595.jpg",
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
... ]
>>> images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

>>> # Run inference on each image reusing text embeddings
>>> # Note: attention_mask must be passed along with text_embeds for proper masking
>>> all_results = []

>>> for image in images:
...     img_inputs = processor(images=image, return_tensors="pt").to(device)
...     with torch.no_grad():
...         outputs = model(
...             pixel_values=img_inputs.pixel_values,
...             text_embeds=text_embeds,
...             attention_mask=text_inputs.attention_mask,
...         )
...
...     results = processor.post_process_instance_segmentation(
...         outputs,
...         threshold=0.5,
...         mask_threshold=0.5,
...         target_sizes=img_inputs.get("original_sizes").tolist()
...     )[0]
...     all_results.append(results)

>>> for i, results in enumerate(all_results):
...     print(f"Image {i+1}: {len(results['masks'])} '{text_prompt}' objects found")
```

### Custom Resolution Inference

<div class="warning">
⚠️ **Performance Note**: Custom resolutions may degrade accuracy. The model is meant to be used at 1008px resolution.
</div>

For faster inference or lower memory usage:

```python
>>> config = Sam3Config.from_pretrained("facebook/sam3")
>>> config.image_size = 560
>>> model = Sam3Model.from_pretrained("facebook/sam3", config=config).to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3", size={"height": 560, "width": 560})
```

### Prompt Label Conventions

SAM3 uses the following label conventions:

**For points and boxes:**

- `1`: Positive prompt (include this region/object)
- `0`: Negative prompt (exclude this region/object)
- `-10`: Padding value for batched inputs

**Coordinate formats:**

- **Input boxes**: `[x1, y1, x2, y2]` (xyxy format) in pixel coordinates
- **Output boxes** (raw): `[x1, y1, x2, y2]` (xyxy format), normalized to [0, 1]
- **Output boxes** (post-processed): `[x1, y1, x2, y2]` (xyxy format) in absolute pixel coordinates

## Sam3Config

[[autodoc]] Sam3Config

## Sam3ViTConfig

[[autodoc]] Sam3ViTConfig

## Sam3VisionConfig

[[autodoc]] Sam3VisionConfig

## Sam3GeometryEncoderConfig

[[autodoc]] Sam3GeometryEncoderConfig

## Sam3DETREncoderConfig

[[autodoc]] Sam3DETREncoderConfig

## Sam3DETRDecoderConfig

[[autodoc]] Sam3DETRDecoderConfig

## Sam3MaskDecoderConfig

[[autodoc]] Sam3MaskDecoderConfig

## Sam3Processor

[[autodoc]] Sam3Processor
    - __call__

## Sam3ImageProcessorFast

[[autodoc]] Sam3ImageProcessorFast
    - preprocess

## Sam3ViTModel

[[autodoc]] Sam3ViTModel
    - forward

## Sam3VisionModel

[[autodoc]] Sam3VisionModel
    - forward

## Sam3Model

[[autodoc]] Sam3Model
    - forward
