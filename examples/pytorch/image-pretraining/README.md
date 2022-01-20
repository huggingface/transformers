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

# Image pretraining examples

This directory contains a script, `run_mae.py`, that can be used to pre-train a Vision Transformer as a masked autoencoder (MAE), as proposed in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). The script can be used to train a `ViTMAEForPreTraining` model in the Transformers library, using PyTorch. After self-supervised pre-training, one can load the weights of the encoder directly into a `ViTForImageClassification`. The MAE method allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data.

The goal for the model is to predict raw pixel values for the masked patches. As the model internally masks patches and learns to reconstruct them, there's no need for any labels. The model uses the mean squared error (MSE) between the reconstructed and original images in the pixel space.

## Using datasets from 🤗 `datasets`

One can use the following command to pre-train a `ViTMAEForPreTraining` model from scratch on the [cifar10](https://huggingface.co/datasets/cifar10) dataset:

```bash
python run_mae.py \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```

Alternatively, one can decide to further pre-train an already pre-trained (or fine-tuned) checkpoint from the [hub](https://huggingface.co/). This can be done by setting the `model_name_or_path` argument to "facebook/vit-mae-base" for example.


## Using your own data

To use your own dataset, the training script expects the following directory structure:

```bash
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

Once you've prepared your dataset, you can can run the script like this:

```bash
python run_mae.py \
    --model_type vit_mae \
    --dataset_name nateraw/image-folder \
    --train_dir <path-to-train-root> \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval
```

### 💡 The above will split the train dir into training and evaluation sets
  - To control the split amount, use the `--train_val_split` flag.
  - To provide your own validation split in its own directory, you can pass the `--validation_dir <path-to-val-root>` flag.


## Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python run_mae.py \
    --push_to_hub \
    --push_to_hub_model_id <name-of-your-model> \
    ...
```