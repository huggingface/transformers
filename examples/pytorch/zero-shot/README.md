<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

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

This directory contains 2 scripts that showcase how to fine-tune any model supported by the [`GroundingDinoForObjectDetection` API](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino#transformers.GroundingDinoForObjectDetection) using PyTorch.

Content:
* [PyTorch version, Trainer](#pytorch-version-trainer)
* [PyTorch version, no Trainer](#pytorch-version-no-trainer)
* [Reload and perform inference](#reload-and-perform-inference)
* [Note on custom data](#note-on-custom-data)


## PyTorch version, Trainer

Based on the script [`run_zero_shot_object_detection.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/zero-shot/run_zero_shot_object_detection.py).

The script leverages the [🤗 Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) to automatically take care of the training for you, running on distributed environments right away.

Here we show how to fine-tune a [GroundingDino](https://huggingface.co/IDEA-Research/grounding-dino-tiny) model on the [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset:

```bash
python run_zero_shot_object_detection.py \
    --model_name_or_path IDEA-Research/grounding-dino-tiny \
    --dataset_name cppe-5 \
    --do_train true \
    --do_eval true \
    --output_dir grounding-dino-tiny-finetuned-cppe-5-10k-steps \
    --num_train_epochs 10 \
    --image_square_size 600 \
    --fp16 true \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --include_inputs_for_metrics true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub true \
    --push_to_hub_model_id grounding-dino-tiny-finetuned-cppe-5-10k-steps \
    --hub_strategy end \
    --seed 1337
```

> Note:  
`--eval_do_concat_batches false` is required for correct evaluation of detection models;  
`--ignore_mismatched_sizes true` is required to load detection model for finetuning with different number of classes.

The resulting model can be seen here: https://huggingface.co/danelcsb/grounding-dino-tiny-finetuned-10k-cppe-5-10k-steps.. Note that it's always advised to check the original paper to know the details regarding training hyperparameters. Hyperparameters for current example were not tuned. To improve model quality you could try:
 - changing freeze policy of image backbone and text backbone
 - changing image size parameters (`--shortest_edge`/`--longest_edge`)
 - changing training parameters, such as learning rate, batch size, warmup, optimizer and many more (see [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments))
 - adding more image augmentations (we created a helpful [HF Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo) to choose some)

Note that you can replace the model and dataset by simply setting the `model_name_or_path` and `dataset_name` arguments respectively, with model or dataset from the [hub](https://huggingface.co/). 
For dataset, make sure it provides labels in the same format as [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset and boxes are provided in [COCO format](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco).

Note that zero-shot inference output is not the same output format as object-detection output. In order to compute the evaluation metric performance such as mean average precision, we have to modify the output little bit.

| Train method | Batch size | freeze_text_backbone | freeze_backbone | precision | MSDA kernels | GPU Memory Usage (GB) | Time (s/epoch) |
|--------------|------------|----------------------|-----------------|-----------|--------------|-----------------------|----------------|
| trainer      | 2          | Y                    | Y               | fp16      | Y            | 22.785                | 353            |
| trainer      | 1          | Y                    | Y               | fp32      | Y            | 8.813                 | 429            |
| no_trainer   | 2          | N                    | N               | fp32      | Y            | OOM                   | -              |
| no_trainer   | 1          | N                    | N               | fp32      | N            | 20.441                | 724            |
| no_trainer   | 1          | N                    | N               | fp32      | Y            | 11.243                | 473            |
| no_trainer   | 1          | Y                    | Y               | fp32      | Y            | 11.539                | 386            |

Above table is tested on following device.
- Platform: Linux-5.4.0-167-generic-x86_64-with-glibc2.35
- GPU type: NVIDIA TITAN RTX
- PyTorch version (GPU): 2.2.2

## PyTorch version, no Trainer

Based on the script [`run_zero_shot_object_detection_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/run_zero_shot_object_detection.py).

The script leverages [🤗 `Accelerate`](https://github.com/huggingface/accelerate), which allows to write your own training loop in PyTorch, but have it run instantly on any (distributed) environment, including CPU, multi-CPU, GPU, multi-GPU and TPU. It also supports mixed precision. However, currently multi-GPU evaluation is not working due to following [issue](https://github.com/Lightning-AI/torchmetrics/issues/2477).

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
accelerate launch run_zero_shot_object_detection_no_trainer.py \
    --model_name_or_path "IDEA-Research/grounding-dino-tiny" \
    --dataset_name cppe-5 \
    --output_dir "grounding-dino-tiny-finetuned-cppe-5-10k-steps-no-trainer" \
    --num_train_epochs 10 \
    --image_square_size 600 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --checkpointing_steps epoch \
    --learning_rate 5e-5 \
    --ignore_mismatched_sizes \
    --with_tracking \
    --push_to_hub \
    --freeze_backbone \
    --freeze_text_backbone
```

and boom, you're training, possibly on multiple GPUs, logging everything to all trackers found in your environment (like Weights and Biases, Tensorboard) and regularly pushing your model to the hub (with the repo name being equal to `args.output_dir` at your HF username) 🤗

With the default settings, the script fine-tunes a [GroundingDino](https://huggingface.co/IDEA-Research/grounding-dino-tiny) model on the [CPPE-5](https://huggingface.co/datasets/cppe-5) dataset. The resulting model can be seen here: https://huggingface.co/danelcsb/grounding-dino-tiny-finetuned-10k-cppe-5-no-trainer. 


## Reload and perform inference

This means that after training, you can easily load your trained model and perform inference as follows::

```python
import requests
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Name of repo on the hub or path to a local folder
model_name = "danelcsb/grounding-dino-tiny-finetuned-10k-cppe5"

image_processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)

# Load image for inference
url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
image = Image.open(requests.get(url, stream=True).raw)
text = "Coverall. Face_Shield. Gloves. Goggles. Mask"

# Prepare image for the model
inputs = image_processor(images=image, text=text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post process model predictions 
# this include conversion to Pascal VOC format and filtering non confident boxes
width, height = image.size
target_sizes = torch.tensor([height, width]).unsqueeze(0)  # add batch dim
results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.15, text_threshold=0.1, target_sizes=target_sizes)[0]

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


## Note on custom data

In case you'd like to use the script with custom data, you could prepare your data with the following way:

```bash
custom_dataset/
└── train
    ├── 0001.jpg
    ├── 0002.jpg
    ├── ...
    └── metadata.jsonl
└── validation
    └── ...
└── test
    └── ...
```

Where `metadata.jsonl` is a file with the following structure:
```json
{"file_name": "0001.jpg", "objects": {"bbox": [[302.0, 109.0, 73.0, 52.0]], "categories": [0], "id": [1], "area": [50.0]}}
{"file_name": "0002.jpg", "objects": {"bbox": [[810.0, 100.0, 57.0, 28.0]], "categories": [1], "id": [2], "area": [40.0]}}
...
```
Trining script support bounding boxes in COCO format (x_min, y_min, width, height).

Then, you cat load the dataset with just a few lines of code:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imagefolder", data_dir="custom_dataset/")

# >>> DatasetDict({
# ...     train: Dataset({
# ...         features: ['image', 'objects'],
# ...         num_rows: 2
# ...     })
# ... })

# Push to hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
dataset.push_to_hub("name of repo on the hub")

# optionally, you can push to a private repo on the hub
# dataset.push_to_hub("name of repo on the hub", private=True)
```

And the final step, for training you should provide id2label mapping in the following way:
```python
id2label = {0: "Car", 1: "Bird", ...}
```
Just find it in code and replace for simplicity, or save `json` locally and with the dataset on the hub!

See also: [Dataset Creation Guide](https://huggingface.co/docs/datasets/image_dataset#create-an-image-dataset)
