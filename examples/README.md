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

# Examples

This folder contains actively maintained examples of use of 🤗 Transformers organized along NLP tasks. If you are looking for an example that used to be in this folder, it may have moved to our [research projects](https://github.com/huggingface/transformers/tree/master/examples/research_projects) subfolder (which contains frozen snapshots of research projects) or to the [legacy](https://github.com/huggingface/transformers/tree/master/examples/legacy) subfolder.

While we strive to present as many use cases as possible, the scripts in this folder are just examples. It is expected that they won't work out-of-the box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs. To help you with that, all the PyTorch versions of the examples fully expose the preprocessing of the data. This way, you can easily tweak them.

This is similar if you want the scripts to report another metric than the one they currently use: look at the `compute_metrics` function inside the script. It takes the full arrays of predictions and labels and has to return a dictionary of string keys and float values. Just change it to add (or replace) your own metric to the ones already reported.

Please discuss on the [forum](https://discuss.huggingface.co/) or in an [issue](https://github.com/huggingface/transformers/issues) a feature you would like to implement in an example before submitting a PR: we welcome bug fixes but since we want to keep the examples as simple as possible, it's unlikely we will merge a pull request adding more functionality at the cost of readability.

## Important note

**Important**

To make sure you can successfully run the latest versions of the example scripts, you have to **install the library from source** and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
Then cd in the example folder of your choice and run
```bash
pip install -r requirements.txt
```

To browse the examples corresponding to released versions of 🤗 Transformers, click on the line below and then on your desired version of the library:

<details>
  <summary>Examples for older versions of 🤗 Transformers</summary>

  - [v4.3.3](https://github.com/huggingface/transformers/tree/v4.3.3/examples)
  - [v4.2.2](https://github.com/huggingface/transformers/tree/v4.2.2/examples)
  - [v4.1.1](https://github.com/huggingface/transformers/tree/v4.1.1/examples)
  - [v4.0.1](https://github.com/huggingface/transformers/tree/v4.0.1/examples)
  - [v3.5.1](https://github.com/huggingface/transformers/tree/v3.5.1/examples)
  - [v3.4.0](https://github.com/huggingface/transformers/tree/v3.4.0/examples)
  - [v3.3.1](https://github.com/huggingface/transformers/tree/v3.3.1/examples)
  - [v3.2.0](https://github.com/huggingface/transformers/tree/v3.2.0/examples)
  - [v3.1.0](https://github.com/huggingface/transformers/tree/v3.1.0/examples)
  - [v3.0.2](https://github.com/huggingface/transformers/tree/v3.0.2/examples)
  - [v2.11.0](https://github.com/huggingface/transformers/tree/v2.11.0/examples)
  - [v2.10.0](https://github.com/huggingface/transformers/tree/v2.10.0/examples)
  - [v2.9.1](https://github.com/huggingface/transformers/tree/v2.9.1/examples)
  - [v2.8.0](https://github.com/huggingface/transformers/tree/v2.8.0/examples)
  - [v2.7.0](https://github.com/huggingface/transformers/tree/v2.7.0/examples)
  - [v2.6.0](https://github.com/huggingface/transformers/tree/v2.6.0/examples)
  - [v2.5.1](https://github.com/huggingface/transformers/tree/v2.5.1/examples)
  - [v2.4.0](https://github.com/huggingface/transformers/tree/v2.4.0/examples)
  - [v2.3.0](https://github.com/huggingface/transformers/tree/v2.3.0/examples)
  - [v2.2.0](https://github.com/huggingface/transformers/tree/v2.2.0/examples)
  - [v2.1.1](https://github.com/huggingface/transformers/tree/v2.1.0/examples)
  - [v2.0.0](https://github.com/huggingface/transformers/tree/v2.0.0/examples)
  - [v1.2.0](https://github.com/huggingface/transformers/tree/v1.2.0/examples)
  - [v1.1.0](https://github.com/huggingface/transformers/tree/v1.1.0/examples)
  - [v1.0.0](https://github.com/huggingface/transformers/tree/v1.0.0/examples)
</details>

Alternatively, you can find switch your cloned 🤗 Transformers to a specific version (for instance with v3.5.1) with
```bash
git checkout tags/v3.5.1
```
and run the example command as usual afterward.

## The Big Table of Tasks

Here is the list of all our examples:
- with information on whether they are **built on top of `Trainer`/`TFTrainer`** (if not, they still work, they might
  just lack some features),
- whether or not they leverage the [🤗 Datasets](https://github.com/huggingface/datasets) library.
- links to **Colab notebooks** to walk through the scripts and run them easily,
<!--
Coming soon!
- links to **Cloud deployments** to be able to deploy large-scale trainings in the Cloud with little to no setup.
-->

| Task | Example datasets | Trainer support | TFTrainer support | 🤗 Datasets | Colab
|---|---|:---:|:---:|:---:|:---:|
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)       | WikiText-2      | ✅ | -  | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb)
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/master/examples/multiple-choice)           | SWAG            | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb)
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/master/examples/question-answering)     | SQuAD           | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb)
| [**`summarization`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                     |  XSum           | ✅ | -  | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb)
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/master/examples/text-classification)   | GLUE            | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
| [**`text-generation`**](https://github.com/huggingface/transformers/tree/master/examples/text-generation)           | -               | n/a | n/a | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/master/examples/token-classification) | CoNLL NER       | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
| [**`translation`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                       | WMT             | ✅  | - | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/translation.ipynb)


## Running quick tests

Most examples are equipped with a mechanism to truncate the number of dataset samples to the desired length. This is useful for debugging purposes, for example to quickly check that all stages of the programs can complete, before running the same setup on the full dataset which may take hours to complete.

For example here is how to truncate all three splits to just 50 samples each:
```
examples/token-classification/run_ner.py \
--max_train_samples 50 \
--max_val_samples 50 \
--max_test_samples 50 \
[...]
```

Most example scripts should have the first two command line arguments and some have the third one. You can quickly check if a given example supports any of these by passing a `-h` option, e.g.:
```
examples/token-classification/run_ner.py -h
```

## Resuming training

You can resume training from a previous checkpoint like this:

1. Pass `--output_dir previous_output_dir` without `--overwrite_output_dir` to resume training from the latest checkpoint in `output_dir` (what you would use if the training was interrupted, for instance).
2. Pass `--model_name_or_path path_to_a_specific_checkpoint` to resume training from that checkpoint folder.

Should you want to turn an example into a notebook where you'd no longer have access to the command
line, 🤗 Trainer supports resuming from a checkpoint via `trainer.train(resume_from_checkpoint)`.

1. If `resume_from_checkpoint` is `True` it will look for the last checkpoint in the value of `output_dir` passed via `TrainingArguments`.
2. If `resume_from_checkpoint` is a path to a specific checkpoint it will use that saved checkpoint folder to resume the training from.


## Distributed training and mixed precision

All the PyTorch scripts mentioned above work out of the box with distributed training and mixed precision, thanks to
the [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html). To launch one of them on _n_ GPUS,
use the following command:

```bash
python -m torch.distributed.launch \
    --nproc_per_node number_of_gpu_you_have path_to_script.py \
	--all_arguments_of_the_script
```

As an example, here is how you would fine-tune the BERT large model (with whole word masking) on the text
classification MNLI task using the `run_glue` script, with 8 GPUs:

```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 text-classification/run_glue.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mnli_output/
```

If you have a GPU with mixed precision capabilities (architecture Pascal or more recent), you can use mixed precision
training with PyTorch 1.6.0 or latest, or by installing the [Apex](https://github.com/NVIDIA/apex) library for previous
versions. Just add the flag `--fp16` to your command launching one of the scripts mentioned above!

Using mixed precision training usually results in 2x-speedup for training with the same final results (as shown in
[this table](https://github.com/huggingface/transformers/tree/master/examples/text-classification#mixed-precision-training)
for text classification).

## Running on TPUs

When using Tensorflow, TPUs are supported out of the box as a `tf.distribute.Strategy`.

When using PyTorch, we support TPUs thanks to `pytorch/xla`. For more context and information on how to setup your TPU environment refer to Google's documentation and to the
very detailed [pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

In this repo, we provide a very simple launcher script named
[xla_spawn.py](https://github.com/huggingface/transformers/tree/master/examples/xla_spawn.py) that lets you run our
example scripts on multiple TPU cores without any boilerplate. Just pass a `--num_cores` flag to this script, then your
regular training script with its arguments (this is similar to the `torch.distributed.launch` helper for
`torch.distributed`):

```bash
python xla_spawn.py --num_cores num_tpu_you_have \
    path_to_script.py \
	--all_arguments_of_the_script
```

As an example, here is how you would fine-tune the BERT large model (with whole word masking) on the text
classification MNLI task using the `run_glue` script, with 8 TPUs:

```bash
python xla_spawn.py --num_cores 8 \
    text-classification/run_glue.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mnli_output/
```

## Logging & Experiment tracking

You can easily log and monitor your runs code. The following are currently supported:

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://docs.wandb.ai/integrations/huggingface)
* [Comet ML](https://www.comet.ml/docs/python-sdk/huggingface/)

### Weights & Biases

To use Weights & Biases, install the wandb package with:

```bash
pip install wandb
```

Then log in the command line:

```bash
wandb login
```

If you are in Jupyter or Colab, you should login with:

```python
import wandb
wandb.login()
```

To enable logging to W&B, include `"wandb"` in the `report_to` of your `TrainingArguments` or script. Or just pass along `--report_to all` if you have `wandb` installed.

Whenever you use `Trainer` or `TFTrainer` classes, your losses, evaluation metrics, model topology and gradients (for `Trainer` only) will automatically be logged.

Advanced configuration is possible by setting environment variables:

| Environment Variable | Value |
|---|---|
| WANDB_LOG_MODEL | Log the model as artifact (log the model as artifact at the end of training (`false` by default) |
| WANDB_WATCH | one of `gradients` (default) to log histograms of gradients, `all` to log histograms of both gradients and parameters, or `false` for no histogram logging |
| WANDB_PROJECT | Organize runs by project |

Set run names with `run_name` argument present in scripts or as part of `TrainingArguments`.

Additional configuration options are available through generic [wandb environment variables](https://docs.wandb.com/library/environment-variables).

Refer to related [documentation & examples](https://docs.wandb.ai/integrations/huggingface).

### Comet.ml

To use `comet_ml`, install the Python package with:

```bash
pip install comet_ml
```

or if in a Conda environment:

```bash
conda install -c comet_ml -c anaconda -c conda-forge comet_ml
```
