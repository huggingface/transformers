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
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) | [WikiText-2](https://huggingface.co/datasets/wikitext) | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) | [SWAG](https://huggingface.co/datasets/swag) | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) | [SQuAD](https://huggingface.co/datasets/squad) | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)
| [**`summarization`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) |  [XSum](https://huggingface.co/datasets/xsum) | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) | [GLUE](https://huggingface.co/datasets/glue) | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
| [**`text-generation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation) | - | n/a | - | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) | [CoNLL NER](https://huggingface.co/datasets/conll2003) | ✅ |✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
| [**`translation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) | [WMT](https://huggingface.co/datasets/wmt17) | ✅ | ✅ |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)
| [**`speech-recognition`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition) | [TIMIT](https://huggingface.co/datasets/timit_asr) | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb)
| [**`multi-lingual speech-recognition`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition) | [Common Voice](https://huggingface.co/datasets/common_voice) | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb)
| [**`audio-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) | [SUPERB KS](https://huggingface.co/datasets/superb) | ✅ | - |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)
| [**`image-pretraining`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining) | [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) | ✅ | - |✅ | /
| [**`image-classification`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) | [CIFAR-10](https://huggingface.co/datasets/cifar10) | ✅ | ✅ |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
| [**`semantic-segmentation`**](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation) | [SCENE_PARSE_150](https://huggingface.co/datasets/scene_parse_150) | ✅ | ✅ |✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb)


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
* [Neptune](https://docs.neptune.ai/integrations-and-supported-tools/model-training/hugging-face)
* [ClearML](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps)

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

### Neptune

First, install the Neptune client library. You can do it with either `pip` or `conda`:

`pip`:

```bash
pip install neptune
```

`conda`:

```bash
conda install -c conda-forge neptune
```

Next, in your model training script, import `NeptuneCallback`:

```python
from transformers.integrations import NeptuneCallback
```

To enable Neptune logging, in your `TrainingArguments`, set the `report_to` argument to `"neptune"`:

```python
training_args = TrainingArguments(
    "quick-training-distilbert-mrpc", 
    evaluation_strategy="steps",
    eval_steps=20,
    report_to="neptune",
)

trainer = Trainer(
    model,
    training_args,
    ...
)
```

**Note:** This method requires saving your Neptune credentials as environment variables (see the bottom of the section).

Alternatively, for more logging options, create a Neptune callback:

```python
neptune_callback = NeptuneCallback()
```

To add more detail to the tracked run, you can supply optional arguments to `NeptuneCallback`.

Some examples:

```python
neptune_callback = NeptuneCallback(
    name = "DistilBERT",
    description = "DistilBERT fine-tuned on GLUE/MRPC",
    tags = ["args-callback", "fine-tune", "MRPC"],  # tags help you manage runs in Neptune
    base_namespace="callback",  # the default is "finetuning"
    log_checkpoints = "best",  # other options are "last", "same", and None
    capture_hardware_metrics = False,  # additional keyword arguments for a Neptune run
)
```

Pass the callback to the Trainer:

```python
training_args = TrainingArguments(..., report_to=None)
trainer = Trainer(
    model,
    training_args,
    ...
    callbacks=[neptune_callback],
)
```

Now, when you start the training with `trainer.train()`, your metadata will be logged in Neptune.

**Note:** Although you can pass your **Neptune API token** and **project name** as arguments when creating the callback, the recommended way is to save them as environment variables:

| Environment variable | Value                                                |
| :------------------- | :--------------------------------------------------- |
| `NEPTUNE_API_TOKEN`  | Your Neptune API token. To find and copy it, click your Neptune avatar and select **Get your API token**. |
| `NEPTUNE_PROJECT` | The full name of your Neptune project (`workspace-name/project-name`). To find and copy it, head to **project settings** &rarr; **Properties**. |

For detailed instructions and examples, see the [Neptune docs](https://docs.neptune.ai/integrations/transformers/).

### ClearML

To use ClearML, install the clearml package with:

```bash
pip install clearml
```

Then [create new credentials]() from the ClearML Server. You can get a free hosted server [here]() or [self-host your own]()!
After creating your new credentials, you can either copy the local snippet which you can paste after running:

```bash
clearml-init
```

Or you can copy the jupyter snippet if you are in Jupyter or Colab:

```python
%env CLEARML_WEB_HOST=https://app.clear.ml
%env CLEARML_API_HOST=https://api.clear.ml
%env CLEARML_FILES_HOST=https://files.clear.ml
%env CLEARML_API_ACCESS_KEY=***
%env CLEARML_API_SECRET_KEY=***
```


To enable logging to ClearML, include `"clearml"` in the `report_to` of your `TrainingArguments` or script. Or just pass along `--report_to all` if you have `clearml` already installed.

Advanced configuration is possible by setting environment variables:

| Environment Variable | Value |
|---|---|
| CLEARML_PROJECT    | Name of the project in ClearML. (default: `"HuggingFace Transformers"`) |
| CLEARML_TASK       | Name of the task in ClearML. (default: `"Trainer"`) |

Additional configuration options are available through generic [clearml environment variables](https://clear.ml/docs/latest/docs/configs/env_vars).
