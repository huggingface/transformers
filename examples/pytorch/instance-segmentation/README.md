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

# Instance segmentation examples

This directory contains 2 scripts that showcase how to fine-tune [MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer) and [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) for instance segmentation using PyTorch.

Content:
* [PyTorch version with Trainer](#pytorch-version-trainer)
* [PyTorch version with Accelerate](#pytorch-version-no-trainer)
* [Reload and perform inference](#reload-and-perform-inference)
* [Note on custom data](#note-on-custom-data)


## PyTorch version, Trainer

Based on the script [`run_instance_segmentation.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py).

The script leverages the [🤗 Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) to automatically take care of the training for you, running on distributed environments right away.

Here we show how to fine-tune a [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) model on a subsample of the [ADE20K](https://huggingface.co/datasets/zhoubolei/scene_parse_150) dataset. For this example we created a [small dataset](https://huggingface.co/datasets/qubvel-hf/ade20k-mini) with ~2k images that contain only "person" and "car" annotations, all other pixels marked as "background".

Here is how `label2id` looks for this dataset:
```python
label2id = {
    "background": 0,
    "person": 1,
    "car": 2,
}
```

Since the `background` label is not an instance and we don't want to predict it, we will specify `do_reduce_labels` to remove it from the data.

You can run the training with the following command:

```bash
python run_instance_segmentation.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former \
    --dataset_name qubvel-hf/ade20k-mini \
    --do_reduce_labels \
    --image_height 256 \
    --image_width 256 \
    --do_train \
    --fp16 \
    --num_train_epochs 40 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub
```

The resulting model can be seen here: https://huggingface.co/qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former. Note that it's always advised to check the original paper to know the details regarding training hyperparameters. Hyperparameters for the current example were not tuned. To improve model quality you could try:
 - changing image size parameters (`--image_height`/`--image_width`)
 - changing training parameters, such as learning rate, batch size, warmup, optimizer and many more (see [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments))
 - adding more image augmentations (we created a helpful [HF Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo) to choose some)

Note that you can replace the model [checkpoint](https://huggingface.co/models?search=maskformer).


## PyTorch version, no Trainer

Based on the script [`run_instance_segmentation_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py).

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
accelerate launch run_instance_segmentation_no_trainer.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former-no-trainer \
    --dataset_name qubvel-hf/ade20k-mini \
    --do_reduce_labels \
    --image_height 256 \
    --image_width 256 \
    --num_train_epochs 40 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --push_to_hub
```

and boom, you're training, possibly on multiple GPUs, logging everything to all trackers found in your environment (like Weights and Biases, Tensorboard) and regularly pushing your model to the hub (with the repo name being equal to `args.output_dir` at your HF username) 🤗

With the default settings, the script fine-tunes a [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) model on the sample of [ADE20K](https://huggingface.co/datasets/qubvel-hf/ade20k-mini) dataset. The resulting model can be seen here: https://huggingface.co/qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former-no-trainer. 


## Reload and perform inference

This means that after training, you can easily load your trained model and perform inference as follows:

```python
import torch
import requests
import matplotlib.pyplot as plt

from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


# Load image
image = Image.open(requests.get("http://farm4.staticflickr.com/3017/3071497290_31f0393363_z.jpg", stream=True).raw)

# Load model and image processor
device = "cuda"
checkpoint = "qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former"

model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, device_map=device)
image_processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)

# Run inference on image
inputs = image_processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post process outputs
outputs = image_processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])

print("Mask shape: ", outputs[0]["segmentation"].shape)
print("Mask values: ", outputs[0]["segmentation"].unique())
for segment in outputs[0]["segments_info"]:
    print("Segment: ", segment)
```

```
Mask shape:  torch.Size([427, 640])
Mask values:  tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.])
Segment:  {'id': 0, 'label_id': 0, 'was_fused': False, 'score': 0.946127}
Segment:  {'id': 1, 'label_id': 1, 'was_fused': False, 'score': 0.961582}
Segment:  {'id': 2, 'label_id': 1, 'was_fused': False, 'score': 0.968367}
Segment:  {'id': 3, 'label_id': 1, 'was_fused': False, 'score': 0.819527}
Segment:  {'id': 4, 'label_id': 1, 'was_fused': False, 'score': 0.655761}
Segment:  {'id': 5, 'label_id': 1, 'was_fused': False, 'score': 0.531299}
Segment:  {'id': 6, 'label_id': 1, 'was_fused': False, 'score': 0.929477}
```

Visualize the results with the following code:
```python
import numpy as np
import matplotlib.pyplot as plt

segmentation = outputs[0]["segmentation"].numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.array(image))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmentation)
plt.axis("off")
plt.show()
```
![Result](https://i.imgur.com/rZmaRjD.png)


## Note on custom data

Here is a short script demonstrating how to create your own dataset for instance segmentation and push it to the hub:

> Note: Annotations should be represented as 3-channel images (similar to [scene_parsing_150](https://huggingface.co/datasets/zhoubolei/scene_parse_150#instance_segmentation-1) dataset), first channel is semantic-segmentation map with values corresponding to `label2id`, second is instance-segmentation map, where each instance has its unique value, third channel should be empty (filled with zeros).

```python
from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage

label2id = {
    "background": 0,
    "person": 1,
    "car": 2,
}

train_split = {
    "image": [<PIL Image 1>, <PIL Image 2>, <PIL Image 3>, ...],
    "annotation": [<PIL Image ann 1>, <PIL Image ann 2>, <PIL Image ann 3>, ...],
}

validation_split = {
    "image": [<PIL Image 101>, <PIL Image 102>, <PIL Image 103>, ...],
    "annotation": [<PIL Image ann 101>, <PIL Image ann 102>, <PIL Image ann 103>, ...],
}

def create_instance_segmentation_dataset(label2id, **splits):
    dataset_dict = {}
    for split_name, split in splits.items():
        split["semantic_class_to_id"] = [label2id] * len(split["image"])
        dataset_split = (
            Dataset.from_dict(split)
            .cast_column("image", DatasetImage())
            .cast_column("annotation", DatasetImage())
        )
        dataset_dict[split_name] = dataset_split
    return DatasetDict(dataset_dict)

dataset = create_instance_segmentation_dataset(label2id, train=train_split, validation=validation_split)
dataset.push_to_hub("qubvel-hf/ade20k-nano")
```

Then, you could use this dataset for fine-tuning by specifying its name with `--dataset_name <your_dataset>`.

See also: [Dataset Creation Guide](https://huggingface.co/docs/datasets/image_dataset#create-an-image-dataset)
