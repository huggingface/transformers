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

# LayoutLMv2

## Using Huggingface Trainer

Fine-tuning the LayoutLMv2 library model. The main script `run_layoutlmv2.py` leverages the 🤗 Datasets library and the Trainer API. You can easily
customize it to your needs if you need extra processing on your datasets.

It will either run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for
training and validation, you might just need to add some tweaks in the data preprocessing.

The following example fine-tunes LayoutLMv2 model on FUNSD dataset:

```bash
python run_layoutlmv2.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --processor_name microsoft/layoutlmv2-base-uncased \
        --output_dir /tmp/test-layoutlmv2 \
        --dataset_name nielsr/funsd \
        --do_train \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --model_revision no_ocr
```

or just can just run the bash script `run.sh`.

To run on your own training and validation files, use the following command:

```bash
python run_layoutlmv2.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --processor_name microsoft/layoutlmv2-base-uncased \
        --output_dir /tmp/test-layoutlmv2 \
        --train_file training_file_path \
        --validation_file validation_file_path \
        --test_file test_file_path \
        --do_train \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --model_revision no_ocr
```

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks)

## Pytorch version, no Trainer

Based on the script `run_layoutlmv2_no_trainer.py`

Like `run_layoutlmv2.py`, this script allows you to fine-tune LayoutLMv2 model. The main difference is that this
script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like.

It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer
or the dataloaders directly in the script) but still run in a distributed setup, on TPU and supports mixed precision by
the mean of the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. You can use the script normally
after installing it:

```bash
pip install accelerate
```

You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
accelerate launch run_layoutlmv2_no_trainer.py \
        --model_name_or_path microsoft/layoutlmv2-base-uncased \
        --processor_name microsoft/layoutlmv2-base-uncased \
        --output_dir /tmp/test-layoutlmv2 \
        --dataset_name nielsr/funsd \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --model_revision no_ocr
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.
