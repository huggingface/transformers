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

# Token classification

## PyTorch version

Fine-tuning the library models for token classification task such as Named Entity Recognition (NER), Parts-of-speech
tagging (POS) or phrase extraction (CHUNKS). The main script `run_ner.py` leverages the 🤗 Datasets library and the Trainer API. You can easily
customize it to your needs if you need extra processing on your datasets.

It will either run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for
training and validation, you might just need to add some tweaks in the data preprocessing.

### Using your own data

If you use your own data, the script expects the following format of the data -

```bash
{
    "chunk_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
    "id": "0",
    "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pos_tags": [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7],
    "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
}
```

The following example fine-tunes BERT on CoNLL-2003:

```bash
python run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval
```

or just can just run the bash script `run.sh`.

To run on your own training and validation files, use the following command:

```bash
python run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --train_file path_to_train_file \
  --validation_file path_to_validation_file \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval
```

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks), if it doesn't you can still use the old version
of the script.

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

## Old version of the script

You can find the old version of the PyTorch script [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/token-classification/run_ner.py).

## Pytorch version, no Trainer

Based on the script [run_ner_no_trainer.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py).

Like `run_ner.py`, this script allows you to fine-tune any of the models on the [hub](https://huggingface.co/models) on a
token classification task, either NER, POS or CHUNKS tasks or your own data in a csv or a JSON file. The main difference is that this
script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like.

It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer
or the dataloaders directly in the script) but still run in a distributed setup, on TPU and supports mixed precision by
the mean of the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. You can use the script normally
after installing it:

```bash
pip install git+https://github.com/huggingface/accelerate
```

then

```bash
export TASK_NAME=ner

python run_ner_no_trainer.py \
  --model_name_or_path google-bert/bert-base-cased \
  --dataset_name conll2003 \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
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
export TASK_NAME=ner

accelerate launch run_ner_no_trainer.py \
  --model_name_or_path google-bert/bert-base-cased \
  --dataset_name conll2003 \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.
