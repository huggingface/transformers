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

# Promptable Concept Segmentation

[[open-in-colab]]

Promptable Concept Segmentation (PCS) is a computer vision task that detects and segments **all instances** of objects matching a given concept in an image. Unlike traditional instance segmentation that is limited to a fixed set of object classes, PCS can segment objects based on:

- **Text prompts** (e.g., "yellow school bus", "ear", "dial")
- **Visual prompts** (bounding boxes indicating positive or negative examples)
- **Combined prompts** (text + visual cues)

For each matching object, PCS returns:
- Binary segmentation masks
- Bounding boxes
- Confidence scores

> [!NOTE]
> Currently, [SAM3](https://huggingface.co/facebook/sam3) is the primary model supporting this task on the Hub.

In this guide, you will learn how to:

- Use the pipeline for quick inference
- Segment objects with text prompts
- Segment objects with bounding box prompts
- Combine text and visual prompts for refined segmentation
- Process multiple images in batches

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install -q transformers
```

## Promptable Concept Segmentation pipeline

The simplest way to try out promptable concept segmentation is to use the [`pipeline`]. Instantiate a pipeline from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?other=sam3):

```python
>>> from transformers import pipeline

>>> segmenter = pipeline("promptable-concept-segmentation", model="facebook/sam3")
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

### Text-based segmentation

Pass the image and a text prompt describing the concept you want to segment:

```py
>>> results = segmenter(image, text="ear", threshold=0.5, mask_threshold=0.5)
>>> results
[{'score': 0.8492,
  'label': 'ear',
  'box': {'xmin': 335, 'ymin': 149, 'xmax': 369, 'ymax': 186},
  'mask': tensor([[False, False, False, ..., False, False, False],
                  [False, False, False, ..., False, False, False],
                  ...])},
 {'score': 0.8415,
  'label': 'ear',
  'box': {'xmin': 194, 'ymin': 152, 'xmax': 227, 'ymax': 190},
  'mask': tensor([[False, False, False, ..., False, False, False],
                  ...])},
 ...]
```

The results contain all detected instances of the concept:
- `score`: Confidence score (0-1)
- `label`: The text prompt used
- `box`: Bounding box in `{xmin, ymin, xmax, ymax}` format (absolute pixel coordinates)
- `mask`: Binary segmentation mask (same size as original image)

### Visualizing results

Let's visualize the segmentation masks:

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from matplotlib.patches import Rectangle

>>> fig, ax = plt.subplots(1, 1, figsize=(10, 8))
>>> ax.imshow(image)

>>> # Create a colored overlay for all masks
>>> overlay = np.zeros((*image.size[::-1], 4))
>>> colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))

>>> for i, result in enumerate(results):
...     mask = result["mask"].numpy()
...     box = result["box"]
...     score = result["score"]
...
...     # Add colored mask
...     overlay[mask] = [*colors[i][:3], 0.5]
...
...     # Draw bounding box
...     rect = Rectangle(
...         (box["xmin"], box["ymin"]),
...         box["xmax"] - box["xmin"],
...         box["ymax"] - box["ymin"],
...         linewidth=2,
...         edgecolor=colors[i],
...         facecolor="none",
...     )
...     ax.add_patch(rect)
...     ax.text(box["xmin"], box["ymin"] - 5, f"{score:.2f}", color="white", fontsize=12, weight="bold")

>>> ax.imshow(overlay)
>>> ax.axis("off")
>>> plt.tight_layout()
>>> plt.show()
```

### Box-based segmentation

You can also segment objects using bounding boxes as visual prompts. This is useful when you want to segment specific object instances:

```py
>>> # Load a different image
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> kitchen_image = Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")

>>> # Define a bounding box around a dial (xyxy format: [x1, y1, x2, y2])
>>> box_xyxy = [59, 144, 76, 163]
>>> input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
>>> input_boxes_labels = [[1]]  # 1 = positive box (include objects like this)

>>> results = segmenter(
...     kitchen_image,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     threshold=0.5,
...     mask_threshold=0.5,
... )

>>> print(f"Found {len(results)} objects matching the visual concept")
```

Box labels can be:
- `1`: Positive (find objects similar to this)
- `0`: Negative (exclude objects like this)

### Combined text and visual prompts

For more precise segmentation, combine text prompts with visual examples:

```py
>>> # Segment "handle" but exclude the oven handle using a negative box
>>> text = "handle"
>>> oven_handle_box = [40, 183, 318, 204]  # Box covering oven handle
>>> input_boxes = [[oven_handle_box]]
>>> input_boxes_labels = [[0]]  # 0 = negative (exclude this region)

>>> results = segmenter(
...     kitchen_image,
...     text=text,
...     input_boxes=input_boxes,
...     input_boxes_labels=input_boxes_labels,
...     threshold=0.5,
...     mask_threshold=0.5,
... )
>>> # This will segment pot handles but exclude the oven handle
```

## Manual inference with model and processor

While the pipeline is convenient, you may want more control over the inference process. Here's how to use the model and processor directly:

```py
>>> from transformers import Sam3Processor, Sam3Model
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")
```

Load an image:

```py
>>> url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
```

Prepare inputs and run inference:

```py
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist(),
... )[0]

>>> print(f"Found {len(results['masks'])} objects")
>>> # Results contain:
>>> # - masks: List of binary masks (torch.Tensor)
>>> # - boxes: Bounding boxes in xyxy format (torch.Tensor)
>>> # - scores: Confidence scores (torch.Tensor)
```

> [!TIP]
> **Pipeline vs Manual Output Format**: The pipeline returns a standardized format (list of dicts with `score`, `label`, `box`, `mask`) for consistency across transformers. The processor's `post_process_instance_segmentation()` returns separate tensors (`scores`, `boxes`, `masks`) for more flexible post-processing.

## Batch processing

You can process multiple images efficiently by batching them together:

```py
>>> cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
>>> images = [
...     Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
...     Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB"),
... ]

>>> # Different text prompt for each image
>>> text_prompts = ["ear", "dial"]

>>> inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist(),
... )

>>> for i, result in enumerate(results):
...     print(f"Image {i+1}: {len(result['masks'])} objects found with prompt '{text_prompts[i]}'")
```

## Efficient multi-prompt inference

When running multiple prompts on the same image, pre-compute vision embeddings to avoid redundant computation:

```py
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
...         target_sizes=img_inputs.get("original_sizes").tolist(),
...     )[0]
...     all_results.append({"prompt": prompt, "results": results})

>>> for item in all_results:
...     print(f"Prompt '{item['prompt']}': {len(item['results']['masks'])} objects found")
```

This approach significantly speeds up inference when testing multiple concepts on the same image!
