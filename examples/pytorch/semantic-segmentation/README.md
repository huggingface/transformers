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

# Semantic segmentation example

This directory contains a script, `run_semantic_segmentation_no_trainer.py`, that showcases how to fine-tune any model supported by the [`AutoModelForSemanticSegmentation` API](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSemanticSegmentation) (such as [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer), [BEiT](https://huggingface.co/docs/transformers/main/en/model_doc/beit), [DPT]((https://huggingface.co/docs/transformers/main/en/model_doc/dpt))) for semantic segmentation using PyTorch.

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
accelerate launch --output_dir segformer-finetuned-sidewalk --wandb --push_to_hub
```

and boom, you're training, possibly on multiple GPUs, logging everything to Weights and Biases and regularly pushing your model to the hub :)

With the default settings, the script fine-tunes a [SegFormer]((https://huggingface.co/docs/transformers/main/en/model_doc/segformer)) model on the [segments/sidewalk-semantic](segments/sidewalk-semantic) dataset.

The resulting model can be seen here: https://huggingface.co/nielsr/segformer-finetuned-sidewalk.

## Reload and perform inference

This means that after training, you can easily load your trained model as follows:

```python
from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation

model_name = "name_of_repo_on_the_hub_or_path_to_local_folder"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
```

and perform inference as follows:

```python
from PIL import Image
import requests
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# rescale logits to original image size
logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                                    size=image.size[::-1], # (height, width)
                                    mode='bilinear',
                                    align_corners=False)

predicted = logits.argmax(1)
```

For visualization of the segmentation maps, we refer to the [example notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb).

## Important notes

Some datasets, like [`scene_parse_150`](scene_parse_150), contain a "background" label that is not part of the classes. The Scene Parse 150 dataset for instance contains labels between 0 and 150, with 0 being the background class, and 1 to 150 being actual class names (like "tree", "person", etc.). For these kind of datasets, one replaces the background label (0) by 255, which is the `ignore_index` of the PyTorch model's loss function, and reduces all labels by 1. This way, the `labels` are PyTorch tensors containing values between 0 and 149, and 255 for all background/padding.

In case you're training on such a dataset, make sure to set the ``reduce_labels`` flag, which will take care of this.