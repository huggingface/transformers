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

# Instance segmentation

[[open-in-colab]]

Instance segmentation is the computer vision task of detecting objects in an image and segmenting each one at the pixel level. Unlike object detection (which outputs bounding boxes), instance segmentation produces a precise mask for every detected object, allowing you to distinguish individual instances even when they overlap.

In this guide, you will learn how to:

 1. Load an instance segmentation dataset from the Hugging Face Hub.
 2. Fine-tune [RF-DETR-Seg](https://huggingface.co/Roboflow/rf-detr-seg-medium), a transformer-based instance segmentation model, using the Transformers [`Trainer`].
 3. Evaluate your model with mean IoU.
 4. Run inference and visualize predictions.

We'll use the [satellite-building-segmentation](https://huggingface.co/datasets/merve/satellite-building-segmentation) dataset, which contains ~9.6k satellite images annotated with building instance masks.

> [!TIP]
> To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/image-segmentation).

Before you begin, install all the necessary libraries:

```bash
pip install -Uq "transformers>=5.9" datasets torchvision
```

We encourage you to share your model with the community. Log in to your Hugging Face account to upload it to the Hub when training is complete:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load the dataset

The [satellite-building-segmentation](https://huggingface.co/datasets/merve/satellite-building-segmentation) dataset is in native Hugging Face Datasets format, so load it directly with `load_dataset`. Each row contains:

- `image`: the satellite image (PIL)
- `image_id`: unique identifier for the image
- `width` / `height`: image dimensions
- `objects`: a dict of per-instance annotations with `id`, `category`, `bbox`, `area`, `segmentation` (polygon coordinates), and `iscrowd`

```py
>>> from datasets import load_dataset

>>> MODEL_ID = "Roboflow/rf-detr-seg-medium"
>>> DATASET_ID = "merve/satellite-building-segmentation"

>>> ds = load_dataset(DATASET_ID)
>>> train_ds = ds["train"]
>>> valid_ds = ds["validation"]
>>> print(f"Train: {len(train_ds)} images, Valid: {len(valid_ds)} images")
```

Inspect a single example. Each record has an `image`, an `image_id`, and an `objects` dict containing per-instance annotations. Each instance has a `bbox` in `[x, y, width, height]` format and a `segmentation` field with polygon coordinates:

```py
>>> sample = train_ds[0]
>>> print(f"Image ID: {sample['image_id']}")
>>> print(f"Image size: {sample['image'].size}")
>>> print(f"Number of instances: {len(sample['objects']['id'])}")
>>> print(f"\nObjects keys: {list(sample['objects'].keys())}")
>>> print(f"First bbox: {sample['objects']['bbox'][0]}")
>>> print(f"First category: {sample['objects']['category'][0]}")
```

Visualize an example with its ground-truth masks:

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from PIL import Image, ImageDraw

>>> sample = train_ds[0]
>>> image = sample["image"].convert("RGB")

>>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))
>>> axes[0].imshow(image)
>>> axes[0].set_title("Original image")
>>> axes[0].axis("off")

>>> overlay = image.copy()
>>> draw = ImageDraw.Draw(overlay, "RGBA")
>>> objects = sample["objects"]
>>> for seg in objects["segmentation"]:
...     for poly in seg:
...         coords = list(zip(poly[0::2], poly[1::2]))
...         color = tuple(np.random.randint(50, 255, 3)) + (100,)
...         draw.polygon(coords, fill=color, outline="red")

>>> axes[1].imshow(overlay)
>>> axes[1].set_title(f"Ground truth ({len(objects['id'])} buildings)")
>>> axes[1].axis("off")
>>> plt.tight_layout()
>>> plt.show()
```

![Dataset Sample](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/satellite-dataset.png)

## Load the model and image processor

Use [`AutoImageProcessor`] and [`AutoModelForInstanceSegmentation`] to load the RF-DETR-Seg model. When loading the model, pass `id2label` and `label2id` mappings to configure the classification head for the single "building" class. Since the pretrained model was trained on COCO (91 classes), set `ignore_mismatched_sizes=True` to reinitialize the classification head with the correct number of outputs.

The image processor handles all the preprocessing: resizing images while maintaining aspect ratio, normalizing with ImageNet statistics, padding to a uniform size, and — crucially for instance segmentation — converting polygon annotations to binary masks, resizing those masks, and normalizing bounding boxes to the `[cx, cy, w, h]` format in `[0, 1]` range that the model expects.

```py
>>> from transformers import AutoImageProcessor, AutoModelForInstanceSegmentation

>>> id2label = {0: "building"}
>>> label2id = {"building": 0}

>>> image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
>>> model = AutoModelForInstanceSegmentation.from_pretrained(
...     MODEL_ID,
...     id2label=id2label,
...     label2id=label2id,
...     ignore_mismatched_sizes=True,
... )
```

## Preprocess the data

To fine-tune the model, you must preprocess the data to match the format the model expects. The `RfDetrImageProcessor` does all the heavy lifting when you pass it images and COCO-format annotations with `return_segmentation_masks=True`:

1. **Rasterizes** polygon segmentations into binary masks
2. **Resizes** images, bounding boxes, and masks to the model's input size
3. **Normalizes** pixel values with ImageNet mean/std
4. **Converts** bounding boxes from `[x, y, w, h]` to normalized `[cx, cy, w, h]`
5. **Pads** images to a uniform size and creates a `pixel_mask`

The transform reconstructs the COCO-style annotation dicts that the image processor expects from the dataset's `objects` column.

Use [`~datasets.Dataset.with_transform`] to apply preprocessing lazily (on-the-fly when samples are loaded), which avoids storing the entire processed dataset in memory.

```py
>>> from functools import partial
>>> from typing import Any


>>> def transform_batch(examples: dict[str, Any], image_processor) -> dict[str, Any]:
...     """Convert HF dataset rows into COCO-style dicts and pass to the processor."""
...     images, targets = [], []
...     for image, img_id, objects in zip(
...         examples["image"], examples["image_id"], examples["objects"]
...     ):
...         if not objects["id"]:
...             continue
...         annotations = [
...             {
...                 "id": ann_id,
...                 "image_id": img_id,
...                 "category_id": cat,
...                 "bbox": bbox,
...                 "area": area,
...                 "segmentation": seg,
...                 "iscrowd": crowd,
...             }
...             for ann_id, cat, bbox, area, seg, crowd in zip(
...                 objects["id"],
...                 objects["category"],
...                 objects["bbox"],
...                 objects["area"],
...                 objects["segmentation"],
...                 objects["iscrowd"],
...             )
...         ]
...         images.append(image.convert("RGB"))
...         targets.append({"image_id": img_id, "annotations": annotations})
...
...     if not images:
...         return {}
...     return image_processor(
...         images=images, annotations=targets, return_segmentation_masks=True, return_tensors="pt"
...     )


>>> transform = partial(transform_batch, image_processor=image_processor)
>>> train_ds = train_ds.shuffle(seed=42).with_transform(transform)
>>> valid_ds = valid_ds.with_transform(transform)
```

Verify a preprocessed example. It contains `pixel_values` (the normalized image tensor), `pixel_mask` (indicating real pixels vs padding), and `labels` (a dict with `class_labels`, `boxes` in normalized center format, and binary `masks`):

```py
>>> example = train_ds[0]
>>> print(f"pixel_values shape: {example['pixel_values'].shape}")
>>> print(f"pixel_mask shape: {example['pixel_mask'].shape}")
>>> print(f"labels keys: {list(example['labels'].keys())}")
>>> print(f"  class_labels: {example['labels']['class_labels']}")
>>> print(f"  boxes shape: {example['labels']['boxes'].shape}")
>>> print(f"  masks shape: {example['labels']['masks'].shape}")
```

## Data collator

Since each image has a different number of object instances, labels are variable-length dictionaries and cannot be stacked into a single tensor. Define a custom collate function that stacks `pixel_values` and `pixel_mask` normally, but keeps `labels` as a list of per-image dicts:

```py
>>> import torch


>>> def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
...     out = {
...         "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
...         "labels": [x["labels"] for x in batch],
...     }
...     if "pixel_mask" in batch[0]:
...         out["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
...     return out
```

## Define evaluation metric: Mean IoU

To track segmentation quality during training, compute a *union-based mean IoU* at each evaluation epoch. For each image:

1. Filter predicted masks by confidence score (> 0.5)
2. Merge all predicted instance masks into a single binary "buildings" map
3. Merge all ground-truth instance masks into a single binary map
4. Compute Intersection-over-Union between the two maps

This gives a per-image metric of "how well does the model cover the buildings". We average it over the full validation set.

Implement this as a [`Trainer`] subclass that runs an extra inference pass during evaluation to compute the metric:

```py
>>> import torch.nn.functional as F
>>> from transformers import Trainer


>>> @torch.no_grad()
... def compute_mean_iou(pred_masks, gt_masks, target_size):
...     """Union-based IoU: merge all instances per image, then compute IoU."""
...     pred_union = (pred_masks.sigmoid() > 0.5).any(dim=0).float()
...     pred_union = F.interpolate(pred_union[None, None], size=target_size, mode="bilinear")[0, 0] > 0.5
...
...     gt_union = gt_masks.any(dim=0).bool()
...
...     intersection = (pred_union & gt_union).sum().float()
...     union = (pred_union | gt_union).sum().float()
...     return (intersection / union.clamp(min=1)).item()


>>> class SegTrainer(Trainer):
...     """Trainer that computes mean IoU during evaluation."""
...
...     def evaluate(self, eval_dataset=None, **kwargs):
...         metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
...
...         eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
...         model = self.model
...         model.eval()
...         device = next(model.parameters()).device
...
...         ious = []
...         dataloader = self.get_eval_dataloader(eval_ds)
...         for batch in dataloader:
...             pixel_values = batch["pixel_values"].to(device)
...             pixel_mask = batch.get("pixel_mask")
...             if pixel_mask is not None:
...                 pixel_mask = pixel_mask.to(device)
...
...             outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
...             pred_masks = outputs.pred_masks
...             logits = outputs.logits
...
...             for i, gt_label in enumerate(batch["labels"]):
...                 gt_m = gt_label["masks"].to(device)
...                 if gt_m.numel() == 0:
...                     continue
...                 target_h, target_w = gt_m.shape[-2:]
...
...                 scores = logits[i].sigmoid().max(dim=-1).values
...                 keep = scores > 0.5
...                 pm = pred_masks[i][keep]
...                 if pm.numel() == 0:
...                     ious.append(0.0)
...                     continue
...                 ious.append(compute_mean_iou(pm, gt_m, (target_h, target_w)))
...
...         mean_iou = sum(ious) / len(ious) if ious else 0.0
...         metrics["eval_mean_iou"] = mean_iou
...         self.log({"eval_mean_iou": mean_iou})
...         print(f"\n*** Mean IoU: {mean_iou:.4f} ***")
...         return metrics
```

## Training

With the data, model, and metrics ready, set up training.. A few important notes on the [`TrainingArguments`]:

- **`remove_unused_columns=False`**: Required because the default behavior would drop columns before our transform runs.
- **`eval_do_concat_batches=False`**: Instance segmentation labels are variable-length dicts — they cannot be concatenated across batches.
- **`metric_for_best_model="eval_mean_iou"`**: Select the best checkpoint by segmentation quality, not just loss.
- **`fp16=True`**: Mixed precision training significantly speeds up training on modern GPUs.

With a batch size of 16 on an A100 80GB GPU, training takes about 1 hour for 10 epochs. On a T4 (16GB), reduce batch size to 2-4 and increase gradient accumulation.

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="rf-detr-seg-satellite-buildings",
...     num_train_epochs=10,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     learning_rate=1e-4,
...     weight_decay=1e-4,
...     lr_scheduler_type="cosine",
...     warmup_ratio=0.1,
...     fp16=True,
...     dataloader_num_workers=4,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     save_total_limit=2,
...     load_best_model_at_end=True,
...     metric_for_best_model="eval_mean_iou",
...     greater_is_better=True,
...     remove_unused_columns=False,
...     eval_do_concat_batches=False,
...     push_to_hub=True,
... )

>>> trainer = SegTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=valid_ds,
...     processing_class=image_processor,
...     data_collator=collate_fn,
... )

>>> trainer.train()
```

If you set `push_to_hub=True` in the training arguments, the training checkpoints are pushed to the
Hugging Face Hub. Upon training completion, push the final model to the Hub as well by calling the [`~transformers.Trainer.push_to_hub`] method.

```py
>>> trainer.push_to_hub(
...     dataset=DATASET_ID,
...     tags=["instance-segmentation", "rf-detr-seg", "vision", "satellite", "building"],
... )
```

## Inference

Now that you have a fine-tuned model, use it for inference on new satellite images. The workflow is:

1. Preprocess the image with the image processor
2. Run a forward pass through the model
3. Post-process outputs with `post_process_instance_segmentation` to get pixel-level masks

The post-processing step converts the raw query outputs (logits + low-res masks) into full-resolution instance segmentation maps, applying score thresholding and mask binarization.

```py
>>> from datasets import load_dataset

>>> test_ds = load_dataset(DATASET_ID, split="test")
>>> sample = test_ds[0]
>>> image = sample["image"].convert("RGB")

>>> device = next(model.parameters()).device
>>> inputs = image_processor(images=image, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_instance_segmentation(
...     outputs, threshold=0.5, target_sizes=[(image.height, image.width)]
... )[0]

>>> print(f"Detected {len(results['segments_info'])} buildings")
>>> for seg_info in results["segments_info"][:5]:
...     print(f"  Building (score: {seg_info['score']:.3f})")
```

Visualize the predictions. The segmentation map assigns each pixel a segment ID (-1 for background). Overlay each detected building with a random color:

```py
>>> fig, axes = plt.subplots(1, 2, figsize=(14, 6))

>>> axes[0].imshow(image)
>>> axes[0].set_title("Input satellite image")
>>> axes[0].axis("off")

>>> seg_map = results["segmentation"].cpu().numpy()
>>> overlay = np.array(image).copy()
>>> for seg_info in results["segments_info"]:
...     mask = seg_map == seg_info["id"]
...     color = np.random.randint(0, 255, 3)
...     overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)

>>> axes[1].imshow(overlay)
>>> axes[1].set_title(f"Predicted masks ({len(results['segments_info'])} buildings)")
>>> axes[1].axis("off")
>>> plt.tight_layout()
>>> plt.show()
```

![Fine-tuning Result](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/finetuned-results.png)
