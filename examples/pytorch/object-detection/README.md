<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Object detection examples

This directory contains 2 scripts that showcase how to fine-tune any model supported by the [`AutoModelForObjectDetection` API](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForObjectDetection) (such as [DETR](https://huggingface.co/docs/transformers/main/en/model_doc/detr), [DETA](https://huggingface.co/docs/transformers/main/en/model_doc/deta), [Deformable DETR](https://huggingface.co/docs/transformers/main/en/model_doc/deformable_detr)) using PyTorch.

Content:
* [Note on custom data](#note-on-custom-data)
* [PyTorch version, Trainer](#pytorch-version-trainer)
* [PyTorch version, no Trainer](#pytorch-version-no-trainer)
* [Reload and perform inference](#reload-and-perform-inference)
* [Important notes](#important-notes)

## Note on custom data

<!---
In case you'd like to use the script with custom data, there are 2 things required: 1) creating a DatasetDict 2) creating an id2label mapping. Below, these are explained in more detail.

### Creating a `DatasetDict`

The script assumes that you have a `DatasetDict` with 2 columns, "image" and "label", both of type [Image](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Image). This can be created as follows:

```python
from datasets import Dataset, DatasetDict, Image

# your images can of course have a different extension
# semantic segmentation maps are typically stored in the png format
image_paths_train = ["path/to/image_1.jpg/jpg", "path/to/image_2.jpg/jpg", ..., "path/to/image_n.jpg/jpg"]
label_paths_train = ["path/to/annotation_1.png", "path/to/annotation_2.png", ..., "path/to/annotation_n.png"]

# same for validation
# image_paths_validation = [...]
# label_paths_validation = [...]

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
  }
)

# step 3: push to hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
dataset.push_to_hub("name of repo on the hub")

# optionally, you can push to a private repo on the hub
# dataset.push_to_hub("name of repo on the hub", private=True)
```

An example of such a dataset can be seen at [nielsr/ade20k-demo](https://huggingface.co/datasets/nielsr/ade20k-demo).

### Creating an id2label mapping

Besides that, the script also assumes the existence of an `id2label.json` file in the repo, containing a mapping from integers to actual class names. An example of that can be seen [here](https://huggingface.co/datasets/nielsr/ade20k-demo/blob/main/id2label.json). This can be created in Python as follows:

```python
import json
# simple example
id2label = {0: 'cat', 1: 'dog'}
with open('id2label.json', 'w') as fp:
    json.dump(id2label, fp)
```

You can easily upload this by clicking on "Add file" in the "Files and versions" tab of your repo on the hub.
-->

## PyTorch version, Trainer

Based on the script [`run_object_detection.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/run_object_detection.py).

The script leverages the [🤗 Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) to automatically take care of the training for you, running on distributed environments right away.

Here we show how to fine-tune a [DETR](https://huggingface.co/facebook/detr-resnet-50) model on the [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset:

```bash
python run_object_detection.py \
    --model_name_or_path facebook/detr-resnet-50 \
    --dataset_name cppe-5 \
    --do_train true \
    --do_eval true \
    --output_dir detr-finetuned-cppe-5-10k-steps \
    --num_train_epochs 100 \
    --shortest_edge 600 \
    --longest_edge 600 \
    --fp16 true \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --eval_do_concat_batches false \ 
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub true \
    --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps \
    --hub_strategy end \
    --seed 1337
```

> Note:  
`--eval_do_concat_batches false` is required for correct evaluation of detection models;  
`--ignore_mismatched_sizes true` is required to load detection model for finetuning with different number of classes.

The resulting model can be seen here: https://huggingface.co/qubvel-hf/qubvel-hf/detr-resnet-50-finetuned-10k-cppe5. The corresponding Weights and Biases report [here](https://api.wandb.ai/links/qubvel-hf-co/bnm0r5ex). Note that it's always advised to check the original paper to know the details regarding training hyperparameters. Hyperparameters for current example were not tuned. To improve model quality you could try:
 - changing image size parameters (`--shortest_edge`/`--longest_edge`)
 - changing training parameters, such as learning rate, batch size, warmup, optimizer and many more (see [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments))
 - adding more image augmentations (we created a helpful [HF Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo) to choose some)

Note that you can replace the model and dataset by simply setting the `model_name_or_path` and `dataset_name` arguments respectively, with model or dataset from the [hub](https://huggingface.co/). 
For dataset, make sure it provides labels in the same format as [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset and boxes are provided in [YOLO format](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#yolo).


## PyTorch version, no Trainer

Based on the script [`run_object_detection_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_object_detection.py).

The script leverages [🤗 `Accelerate`](https://github.com/huggingface/accelerate), which allows to write your own training loop in PyTorch, but have it run instantly on any (distributed) environment, including CPU, multi-CPU, GPU, multi-GPU and TPU. It also supports mixed precision.

First, run:

```bash
accelerate config
```

and reply to the questions asked regarding the environment on which you'd like to train. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
accelerate launch run_object_detection_no_trainer.py \
    --model_name_or_path "facebook/detr-resnet-50" \
    --dataset_name cppe-5 \
    --output_dir "detr-resnet-50-finetuned" \
    --num_train_epochs 100 \
    --shortest_edge 600 \
    --longest_edge 600 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --checkpointing_steps epoch \
    --learning_rate 5e-5 \
    --ignore_mismatched_sizes \
    --with_tracking \
    --push_to_hub
```

and boom, you're training, possibly on multiple GPUs, logging everything to all trackers found in your environment (like Weights and Biases, Tensorboard) and regularly pushing your model to the hub (with the repo name being equal to `args.output_dir` at your HF username) 🤗

With the default settings, the script fine-tunes a [DETR](https://huggingface.co/facebook/detr-resnet-50) model on the [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset. The resulting model can be seen here: https://huggingface.co/qubvel-hf/detr-resnet-50-finetuned-10k-cppe5-no-trainer. 


## Reload and perform inference

This means that after training, you can easily load your trained model and perform inference as follows::

```python
import requests
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Name of repo on the hub or path to a local folder
model_name = "qubvel-hf/detr-resnet-50-finetuned-10k-cppe5"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

# Load image for inference
url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
image = Image.open(requests.get(url, stream=True).raw)

# Prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post process model predictions 
# this include conversion to Pascal VOC format and filtering non confident boxes
width, height = image.size
target_sizes = torch.tensor([height, width]).unsqueeze(0)  # add batch dim
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

And visualize with the following code:
```python
from PIL import ImageDraw
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

image
```

