<!---
Copyright 2020 The HuggingFace Team and Nathan Cooper. All rights reserved.

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

Fine-tuning (or training from scratch) the library models for visual language modeling on an image-text dataset for Fuyu. Fuyu are trained or fine-tuned using a causal visual language modeling
(CVLM) loss.

The example script uses a custom training loop and leverages the 🤗 Accelerate library for the distributed training launcher part. You can easily customize them to your needs if you need extra processing on your datasets. In order for the model to fit into memory, the images are resized to 224 x 224 and the script was tested using FSDP (Fully Sharded Data Parallel) training. This is a technique that shards the model parameters across the replicas so that each replica only holds a portion of the model in memory.

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for training and validation. We give examples of both below.

### Fuyu and causal visual language modeling

The following example fine-tunes Fuyu-8B on facebook/winoground which is a small (800 image-caption pairs) The loss here is that of causal visual language modeling.

```bash
accelerate launch run_fuyu_no_trainer.py \
    --model_name_or_path adept/fuyu-8b \
    --dataset_name facebook/winoground \
    --dataset_caption_column_name caption_0 \
    --dataset_image_column_name image_0 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --low_cpu_mem_usage \
    --output_dir /tmp/test-fuyu-finetune
```

This takes about 3 minutes to train on a 8 x A100 80GB GPUs. It reaches a score of ~1.87 perplexity on the validation set.

## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` to the command line.

## Low Cpu Memory Usage

To use low cpu memory mode which can be very useful for LLM, add `--low_cpu_mem_usage` to the command line.