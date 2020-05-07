# Examples

Version 2.9 of `transformers` introduces a new `Trainer` class for PyTorch, and its equivalent `TFTrainer` for TF 2.

Here is the list of all our examples:
- **grouped by task** (all official examples work for multiple models)
- with information on whether they are **built on top of `Trainer`/`TFTrainer`** (if not, they still work, they might just lack some features),
- whether they also include examples for **`pytorch-lightning`**, which is a great fully-featured, general-purpose training library for PyTorch,
- links to **Colab notebooks** to walk through the scripts and run them easily,
- links to **Cloud deployments** to be able to deploy large-scale trainings in the Cloud with little to no setup.

This is still a work-in-progress – in particular documentation is still sparse – so please **contribute improvements/pull requests.**


## Tasks built on Trainer

| Task | Example datasets | Trainer support | TFTrainer support | pytorch-lightning | Colab | One-click Deploy to Azure (wip) | 
|---|---|:---:|:---:|:---:|:---:|:---:|
| [`language-modeling`](./language-modeling) | Raw text | ✅ | - | - | - | - |
| [`text-classification`](./text-classification) | GLUE, XNLI | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb) | [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure%2Fazure-quickstart-templates%2Fmaster%2F101-storage-account-create%2Fazuredeploy.json) |
| [`token-classification`](./token-classification) | CoNLL NER | ✅ | ✅ | ✅ | - | - |
| [`multiple-choice`](./multiple-choice) | SWAG, RACE, ARC | ✅ | - | - | - | - |



## Other examples and how-to's

| Section | Description |
|---|---|
| [TensorFlow 2.0 models on GLUE](./text-classification) | Examples running BERT TensorFlow 2.0 model on the GLUE tasks. |
| [Running on TPUs](#running-on-tpus) | Examples on running fine-tuning tasks on Google TPUs to accelerate workloads. |
| [Language Model training](./language-modeling) | Fine-tuning (or training from scratch) the library models for language modeling on a text dataset. Causal language modeling for GPT/GPT-2, masked language modeling for BERT/RoBERTa. |
| [Language Generation](./text-generation) | Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet. |
| [GLUE](./text-classification) | Examples running BERT/XLM/XLNet/RoBERTa on the 9 GLUE tasks. Examples feature distributed training as well as half-precision. |
| [SQuAD](./question-answering) | Using BERT/RoBERTa/XLNet/XLM for question answering, examples with distributed training. |
| [Multiple Choice](./multiple-choice) | Examples running BERT/XLNet/RoBERTa on the SWAG/RACE/ARC tasks. |
| [Named Entity Recognition](./token-classification) | Using BERT for Named Entity Recognition (NER) on the CoNLL 2003 dataset, examples with distributed training. |
| [XNLI](./text-classification) | Examples running BERT/XLM on the XNLI benchmark. |
| [Adversarial evaluation of model performances](./adversarial) | Testing a model with adversarial evaluation of natural language inference on the Heuristic Analysis for NLI Systems (HANS) dataset (McCoy et al., 2019.) |

## Important note

**Important**
To make sure you can successfully run the latest versions of the example scripts, you have to install the library from source and install some example-specific requirements.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

## Running on TPUs

Documentation to come.
