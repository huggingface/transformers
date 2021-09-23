<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Audio classification examples

The following examples showcase how to fine-tune `Wav2Vec2` for audio classification using PyTorch.

## Using datasets from 🤗 `datasets`

Here we show how to fine-tune `Wav2Vec2` on the [Keyword Spotting subset](https://huggingface.co/datasets/beans) of the SUPERB dataset.

👀 See the results here: [anton-l/wav2vec2-base-keyword-spotting](https://huggingface.co/anton-l/wav2vec2-base-keyword-spotting).

```bash
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir ./wav2vec2-base-keyword-spotting \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id wav2vec2-base-keyword-spotting \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --dataloader_num_workers=3 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```

## Using your own data

To use your own dataset, convert your data into a `csv` or `json` format with the 
fields `file` and `label` like so:

```json lines
{"file": "/absolute/path/to/sample0.wav", "label": "cat"}
{"file": "/absolute/path/to/sample1.wav", "label": "dog"}
{"file": "/absolute/path/to/sample2.wav", "label": "bird"}
```

Once you've prepared your dataset, you can run the script like this:

```bash
python run_audio_classification.py \
    --dataset_name my-own-dataset \
    --train_file <path-to-train-file> \
    --validation_file <path-to-validation-file> \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval
```


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
python run_audio_classification.py \
    --push_to_hub \
    --push_to_hub_model_id <name-your-model> \
    ...
```