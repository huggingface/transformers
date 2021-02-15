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

## Sequence to Sequence Training and Evaluation

This directory contains examples for finetuning and evaluating transformers on summarization and translation tasks.
Please tag @patil-suraj with any issues/unexpected behaviors, or send a PR!
For deprecated `bertabs` instructions, see [`bertabs/README.md`](https://github.com/huggingface/transformers/blob/master/examples/research_projects/bertabs/README.md).
For the old `finetune_trainer.py` and related utils, see [`examples/legacy/seq2seq`](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq).

### Supported Architectures

- `BartForConditionalGeneration`
- `MarianMTModel`
- `PegasusForConditionalGeneration`
- `MBartForConditionalGeneration`
- `FSMTForConditionalGeneration`
- `T5ForConditionalGeneration`

`run_seq2seq.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files

Here is an example on a summarization task:
```bash
python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task summarization \
    --dataset_name xsum \
    --output_dir ~/tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

And here is how you would use it on your own files (replace `path_to_csv_or_jsonlines_file`, `text_column_name` and `summary_column_name` by the relevant values):
```bash
python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task summarization \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --output_dir ~/tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column text_column_name \
    --summary_column summary_column_name
```
The training and validation files should have a column for the inputs texts and a column for the summaries.

Here is an example of a translation fine-tuning:
```bash
python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path sshleifer/student_marian_en_ro_6_1 \
    --do_train \
    --do_eval \
    --task translation_en_to_ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO\
    --output_dir ~/tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

And here is how you would use it on your own files (replace `path_to_jsonlines_file`, by the relevant values):
```bash
python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path sshleifer/student_marian_en_ro_6_1 \
    --do_train \
    --do_eval \
    --task translation_en_to_ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO\
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir ~/tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
Here the files are expected to be JSONLINES files, with each input being a dictionary with a key `"translation"` containing one key per language (here `"en"` and `"ro"`).
