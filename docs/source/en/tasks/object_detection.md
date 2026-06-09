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

# Object detection

[[open-in-colab]]

Object detection is the computer vision task of detecting instances (such as humans, buildings, or cars) in an image. Object detection models receive an image as input and output
coordinates of the bounding boxes and associated labels of the detected objects. An image can contain multiple objects,
each with its own bounding box and a label (e.g. it can have a car and a building), and each object can
be present in different parts of an image (e.g. the image can have several cars).
This task is commonly used in autonomous driving for detecting things like pedestrians, road signs, and traffic lights.
Other applications include counting objects in images, image search, and more.

In this guide, you will learn how to:

 1. Finetune [RF-DETR](https://huggingface.co/Roboflow/rf-detr-medium) on the [mobile-ui-design](https://huggingface.co/datasets/merve/mobile-ui-design)
 dataset to detect UI elements in mobile app screenshots.
 2. Use your finetuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/object-detection)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install -q datasets transformers accelerate timm trackio torchmetrics pycocotools
```

You'll use 🤗 Datasets to load a dataset from the Hugging Face Hub and 🤗 Transformers to train your model.

We encourage you to share your model with the community. Log in to your Hugging Face account to upload it to the Hub.
When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

Define global constants, namely the model name and image size. This tutorial uses RF-DETR, but you can select any object detection model in Transformers.

```py
>>> MODEL_NAME = "Roboflow/rf-detr-medium"
```

## Load the mobile-ui-design dataset

The [mobile-ui-design dataset](https://huggingface.co/datasets/merve/mobile-ui-design) contains mobile app screenshots with
annotations for detecting UI elements such as text, images, rectangles, and groups.

Start by loading the dataset, extracting category labels, and creating a train/validation split:

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/mobile-ui-design")

>>> CATEGORIES = sorted(set(
...     cat for split in ds.values() for example in split for cat in example["objects"]["category"]
... ))
>>> label2id = {label: i for i, label in enumerate(CATEGORIES)}
>>> id2label = {i: label for label, i in label2id.items()}
>>> print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
Categories (4): ['group', 'image', 'rectangle', 'text']
```

The dataset uses string category names and bounding boxes in COCO format `(x, y, w, h)`. Convert the
categories to integer ids, compute areas, and filter out degenerate bounding boxes before training:

```py
>>> def prepare_example(example, idx):
...     objects = example["objects"]
...     bboxes = objects["bbox"]
...     categories = objects["category"]
...     img_w, img_h = example["width"], example["height"]
...     good_bboxes, good_cats, good_areas, good_ids = [], [], [], []
...     for i, (bbox, cat) in enumerate(zip(bboxes, categories)):
...         x, y, w, h = bbox
...         if w <= 0 or h <= 0:
...             continue
...         x = max(0.0, min(x, img_w))
...         y = max(0.0, min(y, img_h))
...         w = min(w, img_w - x)
...         h = min(h, img_h - y)
...         if w <= 0 or h <= 0:
...             continue
...         good_bboxes.append([x, y, w, h])
...         good_cats.append(label2id[cat])
...         good_areas.append(w * h)
...         good_ids.append(i)
...     return {
...         "image_id": idx, "image": example["image"],
...         "width": example["width"], "height": example["height"],
...         "objects": {"id": good_ids, "bbox": good_bboxes, "category": good_cats, "area": good_areas},
...     }

>>> ds_prepared = ds["train"].map(prepare_example, with_indices=True, remove_columns=ds["train"].column_names)
>>> ds_prepared = ds_prepared.filter(lambda x: len(x["objects"]["bbox"]) > 0)

>>> split = ds_prepared.train_test_split(test_size=0.15, seed=1337)
>>> train_ds = split["train"]
>>> val_ds = split["test"]
>>> print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")
Train: 6669, Validation: 1177
```

## Preprocess the data

[`AutoImageProcessor`] takes care of processing image data to create `pixel_values`, `pixel_mask`, and
`labels` that the model can train with. The image processor handles resizing, padding, and normalization. On top of that, you can optionally add random data augmentations (see [below](#data-augmentation)) to improve generalization.

```py
>>> import numpy as np
>>> from functools import partial
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
```

The `image_processor` expects annotations in the COCO format: `{'image_id': int, 'annotations': list[Dict]}`. Format each example's annotations and let the processor handle the rest:

```py
>>> def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
...     annotations = []
...     for category, area, bbox in zip(categories, areas, bboxes):
...         annotations.append({
...             "image_id": image_id,
...             "category_id": category,
...             "iscrowd": 0,
...             "area": area,
...             "bbox": list(bbox),
...         })
...     return {"image_id": image_id, "annotations": annotations}

>>> def transform_batch(examples, image_processor):
...     images = []
...     annotations = []
...     for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
...         images.append(np.array(image.convert("RGB")))
...         formatted = format_image_annotations_as_coco(
...             image_id, objects["category"], objects["area"], objects["bbox"]
...         )
...         annotations.append(formatted)
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")
...     result.pop("pixel_mask", None)
...     return result

>>> transform_fn = partial(transform_batch, image_processor=image_processor)
>>> train_ds = train_ds.with_transform(transform_fn)
>>> val_ds = val_ds.with_transform(transform_fn)
```

### Data augmentation

The transform above only resizes and normalizes images. Random augmentations applied to the **training** split usually improve generalization, while the validation split should stay augmentation-free so that evaluation stays deterministic. A common choice is [Albumentations](https://albumentations.ai/), which augments the image and its bounding boxes together. Define a pipeline with `bbox_params` so boxes are transformed consistently with the image, then recompute areas from the augmented boxes:

```py
>>> import albumentations as A

>>> train_augment = A.Compose(
...     [
...         A.Perspective(p=0.1),
...         A.HorizontalFlip(p=0.5),
...         A.RandomBrightnessContrast(p=0.5),
...         A.HueSaturationValue(p=0.1),
...     ],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
... )

>>> def augment_and_transform_batch(examples, image_processor, transform):
...     images = []
...     annotations = []
...     for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
...         image = np.array(image.convert("RGB"))
...         output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
...         images.append(output["image"])
...         areas = [w * h for (_, _, w, h) in output["bboxes"]]
...         formatted = format_image_annotations_as_coco(
...             image_id, output["category"], areas, output["bboxes"]
...         )
...         annotations.append(formatted)
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")
...     result.pop("pixel_mask", None)
...     return result
```

Apply the augmenting transform to the training split only, and keep the plain `transform_fn` for validation:

```py
>>> train_augment_fn = partial(augment_and_transform_batch, image_processor=image_processor, transform=train_augment)
>>> train_ds = train_ds.with_transform(train_augment_fn)
>>> val_ds = val_ds.with_transform(transform_fn)
```

Create a custom `collate_fn` to batch images together:

```py
>>> import torch

>>> def collate_fn(batch):
...     data = {}
...     data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
...     data["labels"] = [x["labels"] for x in batch]
...     if "pixel_mask" in batch[0]:
...         data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
...     return data

```

## Preparing function to compute mAP

Object detection models are commonly evaluated with a set of <a href="https://cocodataset.org/#detection-eval">COCO-style metrics</a>. We are going to use `torchmetrics` to compute `mAP` (mean average precision) and `mAR` (mean average recall) metrics and will wrap it to `compute_metrics` function in order to use in [`Trainer`] for evaluation.

Intermediate format of boxes used for training is `YOLO` (normalized) but we will compute metrics for boxes in `Pascal VOC` (absolute) format in order to correctly handle box areas. Let's define a function that converts bounding boxes to `Pascal VOC` format:

```py
>>> from transformers.image_transforms import center_to_corners_format

>>> def convert_bbox_yolo_to_pascal(boxes, image_size):
...     """
...     Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
...     to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

...     Args:
...         boxes (torch.Tensor): Bounding boxes in YOLO format
...         image_size (tuple[int, int]): Image size in format (height, width)

...     Returns:
...         torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
...     """
...     # convert center to corners format
...     boxes = center_to_corners_format(boxes)

...     # convert to absolute coordinates
...     height, width = image_size
...     boxes = boxes * torch.tensor([[width, height, width, height]])

...     return boxes
```

Then, in `compute_metrics` function we collect `predicted` and `target` bounding boxes, scores and labels from evaluation loop results and pass it to the scoring function.

```py
>>> import numpy as np
>>> from dataclasses import dataclass
>>> from torchmetrics.detection.mean_ap import MeanAveragePrecision


>>> @dataclass
>>> class ModelOutput:
...     logits: torch.Tensor
...     pred_boxes: torch.Tensor


>>> def _get_orig_size(image_target):
...     """Robust orig_size extraction - Trainer serialization can truncate to 1 element."""
...     orig = np.atleast_1d(np.asarray(image_target["orig_size"])).flatten()
...     if len(orig) >= 2:
...         return (int(orig[0]), int(orig[1]))
...     return (int(orig[0]), int(orig[0]))

>>> @torch.no_grad()
>>> def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
...     predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
...     image_sizes = []
...     post_processed_targets = []
...     post_processed_predictions = []
...
...     for batch in targets:
...         batch_sizes = []
...         for image_target in batch:
...             h, w = _get_orig_size(image_target)
...             batch_sizes.append([h, w])
...             boxes = torch.tensor(image_target["boxes"])
...             boxes = convert_bbox_yolo_to_pascal(boxes, (h, w))
...             labels = torch.tensor(image_target["class_labels"])
...             post_processed_targets.append({"boxes": boxes, "labels": labels})
...         image_sizes.append(torch.tensor(batch_sizes))
...
...     for batch, target_sizes in zip(predictions, image_sizes):
...         batch_logits, batch_boxes = batch[1], batch[2]
...         output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
...         post_processed_output = image_processor.post_process_object_detection(
...             output, threshold=threshold, target_sizes=target_sizes
...         )
...         post_processed_predictions.extend(post_processed_output)
...
...     metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
...     metric.update(post_processed_predictions, post_processed_targets)
...     metrics = metric.compute()
...
...     classes = metrics.pop("classes")
...     map_per_class = metrics.pop("map_per_class")
...     mar_100_per_class = metrics.pop("mar_100_per_class")
...     for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
...         class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
...         metrics[f"map_{class_name}"] = class_map
...         metrics[f"mar_100_{class_name}"] = class_mar
...
...     metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
...     return metrics

>>> eval_compute_metrics_fn = partial(
...     compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
... )
```

## Training the detection model

You have done most of the heavy lifting in the previous sections, so now you are ready to train your model!
The images in this dataset are still quite large, even after resizing. This means that finetuning this model will
require at least one GPU.

Training involves the following steps:

1. Load the model with [`AutoModelForObjectDetection`] using the same checkpoint as in the preprocessing.
2. Define your training hyperparameters in [`TrainingArguments`].
3. Pass the training arguments to [`Trainer`] along with the model, dataset, image processor, and data collator.
4. Call [`~Trainer.train`] to finetune your model.

When loading the model from the same checkpoint that you used for the preprocessing, remember to pass the `label2id`
and `id2label` maps that you created earlier from the dataset's metadata. Additionally, we specify `ignore_mismatched_sizes=True` to replace the existing classification head with a new one.

```py
>>> from transformers import AutoModelForObjectDetection

>>> model = AutoModelForObjectDetection.from_pretrained(
...     MODEL_NAME,
...     id2label=id2label,
...     label2id=label2id,
...     ignore_mismatched_sizes=True,
... )
```

In the [`TrainingArguments`] use `output_dir` to specify where to save your model, then configure hyperparameters as you see fit. For `num_train_epochs=5` training will take about 35 minutes on an A100 GPU, increase the number of epochs to get better results.

Important notes:

- Do not remove unused columns because this will drop the image column. Without the image column, you
can't create `pixel_values`. For this reason, set `remove_unused_columns` to `False`.
- Set `eval_do_concat_batches=False` to get proper evaluation results. Images have different number of target boxes, if batches are concatenated we will not be able to determine which boxes belongs to particular image.

If you wish to share your model by pushing to the Hub, set `push_to_hub` to `True` (you must be signed in to Hugging
Face to upload your model).

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="rf_detr_finetuned_mobile_ui",
...     num_train_epochs=5,
...     bf16=True,
...     per_device_train_batch_size=8,
...     dataloader_num_workers=4,
...     learning_rate=5e-5,
...     lr_scheduler_type="cosine",
...     weight_decay=1e-4,
...     max_grad_norm=0.01,
...     metric_for_best_model="eval_map",
...     greater_is_better=True,
...     load_best_model_at_end=True,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     save_total_limit=2,
...     remove_unused_columns=False,
...     report_to="trackio",
...     run_name="mobile-ui-detection",
...     eval_do_concat_batches=False,
...     push_to_hub=True,
... )
```

Finally, bring everything together, and call [`~transformers.Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=val_ds,
...     processing_class=image_processor,
...     data_collator=collate_fn,
...     compute_metrics=eval_compute_metrics_fn,
... )

>>> trainer.train()
```

<div>

  <progress value='2085' max='2085' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [2085/2085 38:39, Epoch 5/5]
</div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Map</th>
      <th>Map 50</th>
      <th>Map 75</th>
      <th>Map Small</th>
      <th>Map Medium</th>
      <th>Map Large</th>
      <th>Mar 1</th>
      <th>Mar 10</th>
      <th>Mar 100</th>
      <th>Mar Small</th>
      <th>Mar Medium</th>
      <th>Mar Large</th>
      <th>Map Group</th>
      <th>Mar 100 Group</th>
      <th>Map Image</th>
      <th>Mar 100 Image</th>
      <th>Map Rectangle</th>
      <th>Mar 100 Rectangle</th>
      <th>Map Text</th>
      <th>Mar 100 Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>9.9234</td>
      <td>0.1303</td>
      <td>0.2236</td>
      <td>0.1478</td>
      <td>0.0909</td>
      <td>0.2030</td>
      <td>0.2524</td>
      <td>0.0421</td>
      <td>0.2520</td>
      <td>0.4683</td>
      <td>0.3113</td>
      <td>0.5607</td>
      <td>0.6782</td>
      <td>0.1244</td>
      <td>0.5122</td>
      <td>0.0958</td>
      <td>0.5035</td>
      <td>0.1285</td>
      <td>0.4328</td>
      <td>0.1725</td>
      <td>0.4413</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>9.8472</td>
      <td>0.1893</td>
      <td>0.3017</td>
      <td>0.2124</td>
      <td>0.1347</td>
      <td>0.2789</td>
      <td>0.3038</td>
      <td>0.0549</td>
      <td>0.2961</td>
      <td>0.5140</td>
      <td>0.3433</td>
      <td>0.5941</td>
      <td>0.7406</td>
      <td>0.1305</td>
      <td>0.5423</td>
      <td>0.1979</td>
      <td>0.5578</td>
      <td>0.1964</td>
      <td>0.4648</td>
      <td>0.2324</td>
      <td>0.4437</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>9.6401</td>
      <td>0.2275</td>
      <td>0.3547</td>
      <td>0.2657</td>
      <td>0.1698</td>
      <td>0.3336</td>
      <td>0.3892</td>
      <td>0.0611</td>
      <td>0.3204</td>
      <td>0.5270</td>
      <td>0.3625</td>
      <td>0.6143</td>
      <td>0.7496</td>
      <td>0.1602</td>
      <td>0.5684</td>
      <td>0.2617</td>
      <td>0.5763</td>
      <td>0.2249</td>
      <td>0.4684</td>
      <td>0.2631</td>
      <td>0.4692</td>
    </tr>
    <tr>
      <td>4</td>
      <td>No log</td>
      <td>9.5770</td>
      <td>0.2733</td>
      <td>0.4068</td>
      <td>0.3133</td>
      <td>0.2100</td>
      <td>0.3867</td>
      <td>0.4343</td>
      <td>0.0668</td>
      <td>0.3456</td>
      <td>0.5593</td>
      <td>0.3875</td>
      <td>0.6393</td>
      <td>0.7725</td>
      <td>0.2013</td>
      <td>0.5941</td>
      <td>0.3158</td>
      <td>0.6065</td>
      <td>0.2733</td>
      <td>0.4998</td>
      <td>0.3028</td>
      <td>0.4756</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10.3700</td>
      <td>11.0500</td>
      <td>0.2827</td>
      <td>0.4193</td>
      <td>0.2913</td>
      <td>0.2021</td>
      <td>0.2814</td>
      <td>0.3763</td>
      <td>0.0609</td>
      <td>0.3403</td>
      <td>0.5668</td>
      <td>0.4138</td>
      <td>0.5669</td>
      <td>0.7317</td>
      <td>0.2092</td>
      <td>0.5979</td>
      <td>0.3334</td>
      <td>0.6295</td>
      <td>0.2793</td>
      <td>0.5245</td>
      <td>0.3089</td>
      <td>0.5151</td>
    </tr>
  </tbody>
</table><p>

If you have set `push_to_hub` to `True` in the `training_args`, the training checkpoints are pushed to the
Hugging Face Hub. Upon training completion, push the final model to the Hub as well by calling the [`~transformers.Trainer.push_to_hub`] method.

```py
>>> trainer.push_to_hub()
```

## Evaluate

```py
>>> from pprint import pprint

>>> metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="test")
>>> pprint(metrics)
{'test_loss': 11.05,
 'test_map': 0.2827,
 'test_map_50': 0.4193,
 'test_map_75': 0.2913,
 'test_map_group': 0.2092,
 'test_map_image': 0.3334,
 'test_map_large': 0.3763,
 'test_map_medium': 0.2814,
 'test_map_rectangle': 0.2793,
 'test_map_small': 0.2021,
 'test_map_text': 0.3089,
 'test_mar_1': 0.0609,
 'test_mar_10': 0.3403,
 'test_mar_100': 0.5668,
 'test_mar_100_group': 0.5979,
 'test_mar_100_image': 0.6295,
 'test_mar_100_rectangle': 0.5245,
 'test_mar_100_text': 0.5151,
 'test_mar_large': 0.7317,
 'test_mar_medium': 0.5669,
 'test_mar_small': 0.4138}
```

These results can be further improved by increasing the number of epochs or adjusting other hyperparameters in [`TrainingArguments`]. Give it a go!

## Inference

Now that you have finetuned a model, evaluated it, and uploaded it to the Hugging Face Hub, you can use it for inference.

```py
>>> import torch
>>> from PIL import Image, ImageDraw
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/mobile-ui-design", split="train")
>>> image = ds[5]["image"].convert("RGB")
```

Load model and image processor from the Hugging Face Hub (skip to use already trained in this session):

```py
>>> model_repo = "merve/rf_detr_finetuned_mobile_ui"

>>> image_processor = AutoImageProcessor.from_pretrained(model_repo)
>>> model = AutoModelForObjectDetection.from_pretrained(model_repo)
>>> model.eval()
```

And detect bounding boxes:

```py
>>> with torch.no_grad():
...     inputs = image_processor(images=[image], return_tensors="pt")
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([[image.size[1], image.size[0]]])
...     results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected text with confidence 0.727 at location [324.02, 340.55, 339.52, 359.12]
Detected rectangle with confidence 0.717 at location [39.97, 705.14, 335.93, 753.54]
Detected text with confidence 0.702 at location [199.94, 473.66, 213.41, 490.6]
Detected text with confidence 0.678 at location [153.14, 474.81, 165.33, 491.0]
Detected text with confidence 0.675 at location [262.67, 718.28, 281.44, 740.81]
Detected rectangle with confidence 0.655 at location [143.57, 242.51, 214.32, 274.26]
Detected text with confidence 0.653 at location [298.68, 637.77, 345.68, 656.26]
```

Let's plot the result:

```py
>>> draw = ImageDraw.Draw(image)

>>> colors = {"group": "blue", "image": "green", "rectangle": "red", "text": "orange"}
>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     x, y, x2, y2 = tuple(box)
...     label_name = model.config.id2label[label.item()]
...     color = colors.get(label_name, "red")
...     draw.rectangle((x, y, x2, y2), outline=color, width=2)
...     draw.text((x, y), f"{label_name} {score:.2f}", fill=color)

>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/mobile_ui_result_5.png" alt="Object detection result on a cart screen"/>
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/mobile_ui_result_50.png" alt="Object detection result on a followers screen"/>
</div>
