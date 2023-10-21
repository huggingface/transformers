<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

## Visual Language model training

Fine-tuning (or training from scratch) the library models for visual language modeling on an image-text dataset for Fuyu... Fuyu are trained or fine-tuned using a causal visual language modeling
(CVLM) loss.

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

### Fuyu and causal visual language modeling

The following example fine-tunes Fuyu-8B on facebook/winoground which is a small (800 image-caption pairs) The loss here is that of causal visual language modeling.

```bash
python run_cvlm.py \
    --model_name_or_path adept/fuyu-8b \
    --dataset_name facebook/winoground \
    --text_column_name caption_0 \
    --image_column_name image_0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --low_cpu_mem_usage \
    --output_dir /tmp/test-cvlm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_cvlm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-cvlm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_cvlm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_cvlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-cvlm
```

## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` to the command line. This is currently supported by `run_cvlm.py`.

## Low Cpu Memory Usage

To use low cpu memory mode which can be very useful for LLM, add `--low_cpu_mem_usage` to the command line. This is currently supported by `run_cvlm.py` and `run_cvlm_no_trainer.py`.

## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_cvlm.py --model_type gpt2 --tokenizer_name gpt2 \ --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
[...]
```

This feature is only available in `run_cvlm.py`.
