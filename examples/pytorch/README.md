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

This folder contains actively maintained examples of use of 🤗 Transformers using the PyTorch backend, organized by ML task.

## The Big Table of Tasks

Here is the list of all our examples:
- with information on whether they are **built on top of `Trainer`** (if not, they still work, they might
  just lack some features),
- whether or not they have a version using the [🤗 Accelerate](https://github.com/huggingface/accelerate) library.
- whether or not they leverage the [🤗 Datasets](https://github.com/huggingface/datasets) library.
- links to **Colab notebooks** to walk through the scripts and run them easily,
<!--
Coming soon!
- links to **Cloud deployments** to be able to deploy large-scale trainings in the Cloud with little to no setup.
-->

| Task | Example datasets | Trainer support | 🤗 Accelerate | 🤗 Datasets | Colab
|---|---|:---:|:---:|:---:|:---:|
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) | WikiText-2 | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) | SWAG | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) | SQuAD | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)
| [**`summarization`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) |  XSum | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) | GLUE | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
| [**`text-generation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation) | - | n/a | - | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) | CoNLL NER | ✅ |✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
| [**`translation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) | WMT | ✅ | ✅ |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)
| [**`speech-recognition`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition) | TIMIT | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb)
| [**`multi-lingual speech-recognition`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition) | Common Voice | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb)
| [**`audio-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) | SUPERB KS | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)
| [**`image-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) | CIFAR-10 | ✅ | ✅ |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
| [**`semantic-segmentation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation) | SCENE_PARSE_150 | ✅ | ✅ |✅ | /


## Running quick tests

Most examples are equipped with a mechanism to truncate the number of dataset samples to the desired length. This is useful for debugging purposes, for example to quickly check that all stages of the programs can complete, before running the same setup on the full dataset which may take hours to complete.

For example here is how to truncate all three splits to just 50 samples each:
```
examples/pytorch/token-classification/run_ner.py \
--max_train_samples 50 \
--max_eval_samples 50 \
--max_predict_samples 50 \
[...]
```

Most example scripts should have the first two command line arguments and some have the third one. You can quickly check if a given example supports any of these by passing a `-h` option, e.g.:
```
examples/pytorch/token-classification/run_ner.py -h
```

## Resuming training

You can resume training from a previous checkpoint like this:

1. Pass `--output_dir previous_output_dir` without `--overwrite_output_dir` to resume training from the latest checkpoint in `output_dir` (what you would use if the training was interrupted, for instance).
2. Pass `--resume_from_checkpoint path_to_a_specific_checkpoint` to resume training from that checkpoint folder.

Should you want to turn an example into a notebook where you'd no longer have access to the command
line, 🤗 Trainer supports resuming from a checkpoint via `trainer.train(resume_from_checkpoint)`.

1. If `resume_from_checkpoint` is `True` it will look for the last checkpoint in the value of `output_dir` passed via `TrainingArguments`.
2. If `resume_from_checkpoint` is a path to a specific checkpoint it will use that saved checkpoint folder to resume the training from.


### Upload the trained/fine-tuned model to the Hub

All the example scripts support automatic upload of your final model to the [Model Hub](https://huggingface.co/models) by adding a `--push_to_hub` argument. It will then create a repository with your username slash the name of the folder you are using as `output_dir`. For instance, `"sgugger/test-mrpc"` if your username is `sgugger` and you are working in the folder `~/tmp/test-mrpc`.

To specify a given repository name, use the `--hub_model_id` argument. You will need to specify the whole repository name (including your username), for instance `--hub_model_id sgugger/finetuned-bert-mrpc`. To upload to an organization you are a member of, just use the name of that organization instead of your username: `--hub_model_id huggingface/finetuned-bert-mrpc`.

A few notes on this integration:

- you will need to be logged in to the Hugging Face website locally for it to work, the easiest way to achieve this is to run `huggingface-cli login` and then type your username and password when prompted. You can also pass along your authentication token with the `--hub_token` argument.
- the `output_dir` you pick will either need to be a new folder or a local clone of the distant repository you are using.

## Distributed training and mixed precision

All the PyTorch scripts mentioned above work out of the box with distributed training and mixed precision, thanks to
the [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html). To launch one of them on _n_ GPUs,
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
    --nproc_per_node 8 pytorch/text-classification/run_glue.py \
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
[this table](https://github.com/huggingface/transformers/tree/main/examples/text-classification#mixed-precision-training)
for text classification).

## Running on TPUs

When using Tensorflow, TPUs are supported out of the box as a `tf.distribute.Strategy`.

When using PyTorch, we support TPUs thanks to `pytorch/xla`. For more context and information on how to setup your TPU environment refer to Google's documentation and to the
very detailed [pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

In this repo, we provide a very simple launcher script named
[xla_spawn.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/xla_spawn.py) that lets you run our
example scripts on multiple TPU cores without any boilerplate. Just pass a `--num_cores` flag to this script, then your
regular training script with its arguments (this is similar to the `torch.distributed.launch` helper for
`torch.distributed`):

```bash
python xla_spawn.py --num_cores num_tpu_you_have \
    path_to_script.py \
	--all_arguments_of_the_script
```

As an example, here is how you would fine-tune the BERT large model (with whole word masking) on the text
classification MNLI task using the `run_glue` script, with 8 TPUs (from this folder):

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

## Using Accelerate

Most PyTorch example scripts have a version using the [🤗 Accelerate](https://github.com/huggingface/accelerate) library
that exposes the training loop so it's easy for you to customize or tweak them to your needs. They all require you to
install `accelerate` with the latest development version

```bash
pip install git+https://github.com/huggingface/accelerate
```

Then you can easily launch any of the scripts by running

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
accelerate launch path_to_script.py --args_to_script
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
| WANDB_LOG_MODEL | Log the model as artifact (log the model as artifact at the end of training) (`false` by default) |
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
