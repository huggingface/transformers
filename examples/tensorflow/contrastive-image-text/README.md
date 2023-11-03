<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# TFVisionTextDualEncoder and CLIP model training examples

The following example showcases how to train a CLIP-like vision-text dual encoder model
using a pre-trained vision and text encoder.

Such a model can be used for natural language image search and potentially zero-shot image classification.
The model is inspired by [CLIP](https://openai.com/blog/clip/), introduced by Alec Radford et al.
The idea is to train a vision encoder and a text encoder jointly to project the representation of images and their
captions into the same embedding space, such that the caption embeddings are located near the embeddings
of the images they describe.

### Download COCO dataset (2017)
This example uses COCO dataset (2017) through a custom dataset script, which requires users to manually download the
COCO dataset before training.

```bash
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
cd ..
```

Having downloaded COCO dataset manually you should be able to load with the `ydshieh/coc_dataset_script` dataset loading script:

```py
import os
import datasets

COCO_DIR = os.path.join(os.getcwd(), "data")
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
```

### Create a model from a vision encoder model and a text encoder model
We can either load a CLIP-like vision-text dual encoder model from an existing dual encoder model, or
by using a pre-trained vision encoder model and a pre-trained text encoder model.

If you wish to load an existing dual encoder model, please use the `--model_name_or_path` argument. If
you want to use separate pre-trained vision and text models, please use the
`--vision_model_name_or_path` and `--text_model_name_or_path` arguments instead.

### Train the model
Finally, we can run the example script to train the model:

```bash
python examples/tensorflow/contrastive-image-text/run_clip.py \
    --output_dir ./clip-roberta-finetuned \
    --vision_model_name_or_path openai/clip-vit-base-patch32 \
    --text_model_name_or_path roberta-base \
    --data_dir $PWD/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --push_to_hub
```
