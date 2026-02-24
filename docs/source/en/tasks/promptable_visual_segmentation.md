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

# Promptable Visual Segmentation

[[open-in-colab]]

Promptable Visual Segmentation (PVS) is a computer vision task that segments objects in an image based on interactive visual prompts. Unlike automatic segmentation methods, PVS lets you specify **exactly which objects** to segment by providing:

- **Point prompts** with labels (positive points to include, negative points to exclude)
- **Bounding box prompts** (rectangular regions around objects)
- **Combinations** of points and boxes for refined segmentation

For each prompted object, PVS returns:
- Binary segmentation masks
- Quality/confidence scores (IoU predictions)

> [!NOTE]
> This task is supported by the SAM-family models on the Hub: [SAM3Tracker](https://huggingface.co/facebook/sam3), [SAM2](https://huggingface.co/facebook/sam2.1-hiera-large), [SAM](https://huggingface.co/facebook/sam-vit-base), and [EdgeTAM](https://huggingface.co/yonigozlan/EdgeTAM-hf).

In this guide, you will learn how to:

- Use the pipeline for quick inference
- Segment objects with single point clicks
- Refine segmentation with multiple points
- Use bounding boxes as prompts
- Segment multiple objects simultaneously
- Process batches of images efficiently

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install -q transformers
```

## Promptable Visual Segmentation pipeline

The simplest way to try out promptable visual segmentation is to use the [`pipeline`]. Instantiate a pipeline from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?other=sam2):

```python
>>> from transformers import pipeline

>>> segmenter = pipeline("promptable-visual-segmentation", model="facebook/sam2.1-hiera-large")
```

Next, choose an image you'd like to segment objects in. Here we'll use an image from the [COCO dataset](https://cocodataset.org/):

```py
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> image
```

<div class="flex justify-center">
     <img src="http://images.cocodataset.org/val2017/000000077595.jpg" alt="Cats on a couch"/>
</div>

### Single point segmentation

Pass the image and a point prompt. Points are specified as `[[[x, y]]]` coordinates with corresponding labels `[[[1]]]` where `1` means "include this object":

```py
>>> # Click on a cat's body
>>> input_points = [[[[450, 600]]]]  # [batch, objects, points_per_object, coordinates]
>>> input_labels = [[[1]]]  # [batch, objects, points_per_object] - 1=positive click

>>> results = segmenter(image, input_points=input_points, input_labels=input_labels)
>>> results
[[{'score': 0.8731,
   'mask': tensor([[False, False, False, ..., False, False, False],
                   [False, False, False, ..., False, False, False],
                   ...])}]]
```

The results are a list of lists (one inner list per input image). Each object gets multiple mask predictions ranked by quality score:
- `score`: Quality score (typically IoU prediction, 0-1)
- `mask`: Binary segmentation mask (same size as original image)

By default, the model returns 3 masks per prompt, ranked by quality. To get only the best mask:

```py
>>> results = segmenter(image, input_points=input_points, input_labels=input_labels, multimask_output=False)
>>> print(f"Returned {len(results[0])} mask(s)")  # 1 mask
Returned 1 mask(s)
```

### Visualizing results

Let's visualize the segmentation mask:

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> fig, axes = plt.subplots(1, 2, figsize=(15, 5))

>>> # Show original image with point
>>> axes[0].imshow(image)
>>> point_x, point_y = input_points[0][0][0]
>>> axes[0].plot(point_x, point_y, "ro", markersize=10, markeredgewidth=2, markeredgecolor="white")
>>> axes[0].set_title("Input: Image + Point")
>>> axes[0].axis("off")

>>> # Show segmentation result
>>> mask = results[0][0]["mask"].numpy()
>>> score = results[0][0]["score"]

>>> axes[1].imshow(image)
>>> # Create colored overlay
>>> overlay = np.zeros((*mask.shape, 4))
>>> overlay[mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
>>> axes[1].imshow(overlay)
>>> axes[1].set_title(f"Segmentation (score: {score:.3f})")
>>> axes[1].axis("off")

>>> plt.tight_layout()
>>> plt.show()
```

### Multiple points for refinement

You can provide multiple points to refine the segmentation. Use positive points (label=1) to include regions and negative points (label=0) to exclude them:

```py
>>> # First positive point on cat body, second negative point on the couch
>>> input_points = [[[[450, 600], [300, 400]]]]
>>> input_labels = [[[1, 0]]]  # 1=include, 0=exclude

>>> results = segmenter(
...     image,
...     input_points=input_points,
...     input_labels=input_labels,
...     multimask_output=False,
... )
>>> # This will segment the cat while excluding couch regions
```

### Bounding box segmentation

You can also use bounding boxes as prompts. Boxes are specified in `[x1, y1, x2, y2]` format (top-left and bottom-right corners):

```py
>>> # Define a box around the left cat
>>> input_boxes = [[[100, 200, 350, 550]]]  # [batch, objects, 4]

>>> results = segmenter(image, input_boxes=input_boxes, multimask_output=False)
>>> mask = results[0][0]["mask"]
>>> print(f"Segmented object with box prompt, score: {results[0][0]['score']:.3f}")
```

### Multiple objects segmentation

Segment multiple objects in the same image by providing multiple prompts:

```py
>>> # Points for two cats - each cat gets its own point
>>> input_points = [
...     [[[450, 600]], [[200, 300]]]  # Two objects, each with one point
... ]
>>> input_labels = [[[1], [1]]]  # Both positive

>>> results = segmenter(
...     image,
...     input_points=input_points,
...     input_labels=input_labels,
...     multimask_output=False,
... )

>>> print(f"Segmented {len(results[0])} objects")
>>> for i, obj_result in enumerate(results[0]):
...     print(f"Object {i+1}: score={obj_result['score']:.3f}")
```

### Combining points and boxes

For maximum precision, you can combine point and box prompts:

```py
>>> # Box around an object + refinement points
>>> input_boxes = [[[100, 200, 350, 550]]]
>>> input_points = [[[[200, 300], [150, 250]]]]  # Positive and negative points
>>> input_labels = [[[1, 0]]]

>>> results = segmenter(
...     image,
...     input_points=input_points,
...     input_labels=input_labels,
...     input_boxes=input_boxes,
...     multimask_output=False,
... )
```

## Manual inference with model and processor

While the pipeline is convenient, you may want more control over the inference process. Here's how to use the model and processor directly:

```py
>>> from transformers import Sam2Processor, Sam2Model
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
```

Load an image:

```py
>>> url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
```

Prepare inputs and run inference:

```py
>>> input_points = [[[[450, 600]]]]
>>> input_labels = [[[1]]]

>>> inputs = processor(
...     images=image,
...     input_points=input_points,
...     input_labels=input_labels,
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks to original image size
>>> masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(),
...     inputs["original_sizes"],
... )[0]

>>> print(f"Mask shape: {masks.shape}")  # [num_objects, num_masks_per_object, height, width]
>>> print(f"IoU scores: {outputs.iou_scores}")
>>> # Results contain:
>>> # - masks: Segmentation masks (torch.Tensor)
>>> # - iou_scores: Quality predictions for each mask (torch.Tensor)
```

> [!TIP]
> **Pipeline vs Manual Output Format**: The pipeline returns a standardized format (list of lists of dicts with `score` and `mask`) for consistency across transformers. The processor's `post_process_masks()` returns raw tensors for more flexible post-processing.

## Batch processing

You can process multiple images efficiently by batching them together:

```py
>>> cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> images = [
...     Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
...     Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB"),
... ]

>>> # Different prompts for each image
>>> input_points = [
...     [[[450, 600]]],  # Cat image: single point
...     [[[300, 250]]],  # Kitchen image: single point
... ]
>>> input_labels = [[[1]], [[1]]]

>>> inputs = processor(
...     images=images,
...     input_points=input_points,
...     input_labels=input_labels,
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])

>>> for i, image_masks in enumerate(masks):
...     print(f"Image {i+1}: {image_masks.shape[0]} object(s) segmented")
```

## Efficient multi-prompt inference

When running multiple prompts on the same image, pre-compute image embeddings to avoid redundant computation:

```py
>>> # Pre-process image and compute image embeddings once
>>> img_inputs = processor(images=image, return_tensors="pt").to(device)
>>> with torch.no_grad():
...     image_embeddings = model.get_image_features(pixel_values=img_inputs.pixel_values)

>>> # Run multiple prompts efficiently
>>> point_prompts = [
...     [[[[450, 600]]]],  # Point on left cat
...     [[[[200, 300]]]],  # Point on right cat
...     [[[[150, 450]]]],  # Point on couch
... ]
>>> all_results = []

>>> for points in point_prompts:
...     labels = [[[1]]]
...     prompt_inputs = processor(
...         input_points=points,
...         input_labels=labels,
...         original_sizes=img_inputs["original_sizes"],
...         return_tensors="pt",
...     ).to(device)
...
...     with torch.no_grad():
...         outputs = model(
...             input_points=prompt_inputs["input_points"],
...             input_labels=prompt_inputs["input_labels"],
...             image_embeddings=image_embeddings,
...             multimask_output=False,
...         )
...
...     masks = processor.post_process_masks(
...         outputs.pred_masks.cpu(),
...         img_inputs["original_sizes"],
...     )[0]
...     all_results.append({"points": points, "masks": masks, "scores": outputs.iou_scores})

>>> print(f"Processed {len(all_results)} prompts efficiently")
```

This approach significantly speeds up inference when testing multiple points on the same image!

## Advanced usage: Interactive segmentation

PVS is ideal for interactive applications where users click to segment objects. Here's a simple iterative refinement workflow:

```py
>>> def interactive_segment(image, positive_points, negative_points=None):
...     """Segment an object with interactive point clicks."""
...     all_points = positive_points + (negative_points or [])
...     labels = [1] * len(positive_points) + [0] * len(negative_points or [])
...
...     input_points = [[all_points]]
...     input_labels = [[labels]]
...
...     results = segmenter(
...         image,
...         input_points=input_points,
...         input_labels=input_labels,
...         multimask_output=False,
...     )
...     return results[0][0]

>>> # Simulated interactive clicks
>>> # Initial click
>>> result = interactive_segment(image, positive_points=[[450, 600]])
>>> print(f"Initial segmentation score: {result['score']:.3f}")

>>> # Refine with additional positive click
>>> result = interactive_segment(image, positive_points=[[450, 600], [380, 550]])
>>> print(f"Refined segmentation score: {result['score']:.3f}")

>>> # Further refine with negative click to exclude background
>>> result = interactive_segment(
...     image,
...     positive_points=[[450, 600], [380, 550]],
...     negative_points=[[300, 400]],
... )
>>> print(f"Final segmentation score: {result['score']:.3f}")
```

This demonstrates how PVS can be used in interactive tools where users iteratively refine segmentation masks by adding positive and negative clicks!
