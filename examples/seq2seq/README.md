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
- `FSMTForConditionalGeneration` (translation only)
- `T5ForConditionalGeneration`

`run_summarization.py` and `run_translation.py` are lightweight examples of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
and you also will find examples of these below.

### Summarization

Here is an example on a summarization task:
```bash
python examples/seq2seq/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name xsum \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
```

CNN/DailyMail dataset is another commonly used dataset for the task of summarization. To use it replace `--dataset_name xsum` with `--dataset_name cnn_dailymail --dataset_config "3.0.0"`.

And here is how you would use it on your own files, after adjusting the values for the arguments
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python examples/seq2seq/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
```

The task of summarization supports custom CSV and JSONLINES formats.

#### Custom CSV Files

If it's a csv file the training and validation files should have a column for the inputs texts and a column for the summaries.

If the csv file has just two columns as in the following example:

```csv
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```

The first column is assumed to be for `text` and the second is for summary.

If the csv file has multiple columns, you can then specify the names of the columns to use:

```bash
    --text_column text_column_name \
    --summary_column summary_column_name \
```

For example if the columns were:

```csv
id,date,text,summary
```

and you wanted to select only `text` and `summary`, then you'd pass these additional arguments:

```bash
    --text_column text \
    --summary_column summary \
```

#### Custom JSONFILES Files

The second supported format is jsonlines. Here is an example of a jsonlines custom data file.


```json
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

Same as with the CSV files, by default the first value will be used as the text record and the second as the summary record. Therefore you can use any key names for the entries, in this example `text` and `summary` were used.

And as with the CSV files, you can specify which values to select from the file, by explicitly specifying the corresponding key names. In our example this again would be:

```bash
    --text_column text \
    --summary_column summary \
```



### Translation

Here is an example of a translation fine-tuning with T5:

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
```

And the same with MBart:

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path facebook/mbart-large-en-ro  \
    --do_train \
    --do_eval \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
 ```

Note, that depending on the used model additional language-specific command-line arguments are sometimes required. Specifically:

* MBart models require different `--{source,target}_lang` values, e.g. in place of `en` it expects `en_XX`, for `ro` it expects `ro_RO`. The full MBart specification for language codes can be looked up [here](https://huggingface.co/facebook/mbart-large-cc25)
* T5 models can use a `--source_prefix` argument to override the otherwise automated prefix of the form `translate {source_lang} to {target_lang}` for `run_translation.py` and `summarize: ` for `run_summarization.py`

Also, if you switch to a different language pair, make sure to adjust the source and target values in all command line arguments.

And here is how you would use the translation finetuning on your own files, after adjusting the
values for the arguments `--train_file`, `--validation_file` to match your setup:

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
```

The task of translation supports only custom JSONLINES files, with each line being a dictionary with a key `"translation"` and its value another dictionary whose keys is the language pair. For example:

```json
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." } }
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." } }
```
Here the languages are Romanian (`ro`) and English (`en`).

If you want to use a pre-processed dataset that leads to high bleu scores, but for the `en-de` language pair, you can use `--dataset_name wmt14-en-de-pre-processed`, as following:

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --dataset_name wmt14-en-de-pre-processed \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples 500 \
    --max_val_samples 500
 ```
