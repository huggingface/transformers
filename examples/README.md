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

This folder contains actively maintained examples of use of 🤗 Transformers organized along NLP tasks. If you are looking for an example that used to
be in this folder, it may have moved to our [research projects](https://github.com/huggingface/transformers/tree/master/examples/research_projects) subfolder (which contains frozen snapshots of research projects).

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

Alternatively, you can run the version of the examples as they were for your current version of Transformers via (for instance with v3.5.1):
```bash
git checkout tags/v3.5.1
```

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
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)       | Raw text        | ✅ | -  | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/master/examples/multiple-choice)           | SWAG, RACE, ARC | ✅ | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ViktorAlm/notebooks/blob/master/MPC_GPU_Demo_for_TF_and_PT.ipynb)
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/master/examples/question-answering)     | SQuAD           | ✅ | ✅ | ✅ | -
| [**`summarization`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                     | CNN/Daily Mail  | ✅  | - | - | -
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/master/examples/text-classification)   | GLUE, XNLI      | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
| [**`text-generation`**](https://github.com/huggingface/transformers/tree/master/examples/text-generation)           | -               | n/a | n/a | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/master/examples/token-classification) | CoNLL NER       | ✅ | ✅ | ✅ | -
| [**`translation`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                       | WMT             | ✅  | - | - | -


<!--
## One-click Deploy to Cloud (wip)

**Coming soon!**
-->

## Running on TPUs

When using Tensorflow, TPUs are supported out of the box as a `tf.distribute.Strategy`.

When using PyTorch, we support TPUs thanks to `pytorch/xla`. For more context and information on how to setup your TPU environment refer to Google's documentation and to the
very detailed [pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

In this repo, we provide a very simple launcher script named [xla_spawn.py](https://github.com/huggingface/transformers/tree/master/examples/xla_spawn.py) that lets you run our example scripts on multiple TPU cores without any boilerplate.
Just pass a `--num_cores` flag to this script, then your regular training script with its arguments (this is similar to the `torch.distributed.launch` helper for torch.distributed). 
Note that this approach does not work for examples that use `pytorch-lightning`.

For example for `run_glue`:

```bash
python examples/xla_spawn.py --num_cores 8 \
	examples/text-classification/run_glue.py \
	--model_name_or_path bert-base-cased \
	--task_name mnli \
	--data_dir ./data/glue_data/MNLI \
	--output_dir ./models/tpu \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--num_train_epochs 1 \
	--save_steps 20000
```

Feedback and more use cases and benchmarks involving TPUs are welcome, please share with the community.

## Logging & Experiment tracking

You can easily log and monitor your runs code. The following are currently supported:

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://docs.wandb.com/library/integrations/huggingface)
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

Whenever you use `Trainer` or `TFTrainer` classes, your losses, evaluation metrics, model topology and gradients (for `Trainer` only) will automatically be logged.

When using 🤗 Transformers with PyTorch Lightning, runs can be tracked through `WandbLogger`. Refer to related [documentation & examples](https://docs.wandb.com/library/integrations/lightning).

### Comet.ml

To use `comet_ml`, install the Python package with:

```bash
pip install comet_ml
```

or if in a Conda environment:

```bash
conda install -c comet_ml -c anaconda -c conda-forge comet_ml
```
