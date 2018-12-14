# PyTorch Pretrained Bert

This repository contains an op-for-op PyTorch reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation is provided with [Google's pre-trained models](https://github.com/google-research/bert), examples, notebooks and a command-line interface to load any pre-trained TensorFlow checkpoint for BERT is also provided.

## Content

| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Overview](#overview) | Overview of the package |
| [Usage](#usage) | Quickstart examples |
| [Doc](#doc) |  Detailed documentation |
| [Examples](#examples) | Detailed examples on how to fine-tune Bert |
| [Notebooks](#notebooks) | Introduction on the provided Jupyter Notebooks |
| [TPU](#tpu) | Notes on TPU support and pretraining scripts |
| [Command-line interface](#Command-line-interface) | Convert a TensorFlow checkpoint in a PyTorch dump |

## Installation

This repo was tested on Python 3.5+ and PyTorch 0.4.1/1.0.0

### With pip

PyTorch pretrained bert can be installed by pip as follows:
```bash
pip install pytorch-pretrained-bert
```

### From source

Clone the repository and run:
```bash
pip install [--editable] .
```

A series of tests is included in the [tests folder](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/tests) and can be run using `pytest` (install pytest if needed: `pip install pytest`).

You can run the tests with the command:
```bash
python -m pytest -sv tests/
```

## Overview

This package comprises the following classes that can be imported in Python and are detailed in the [Doc](#doc) section of this readme:

- Eight PyTorch models (`torch.nn.Module`) for Bert with pre-trained weights (in the [`modeling.py`](./pytorch_pretrained_bert/modeling.py) file):
  - [`BertModel`](./pytorch_pretrained_bert/modeling.py#L537) - raw BERT Transformer model (**fully pre-trained**),
  - [`BertForMaskedLM`](./pytorch_pretrained_bert/modeling.py#L691) - BERT Transformer with the pre-trained masked language modeling head on top (**fully pre-trained**),
  - [`BertForNextSentencePrediction`](./pytorch_pretrained_bert/modeling.py#L752) - BERT Transformer with the pre-trained next sentence prediction classifier on top  (**fully pre-trained**),
  - [`BertForPreTraining`](./pytorch_pretrained_bert/modeling.py#L620) - BERT Transformer with masked language modeling head and next sentence prediction classifier on top (**fully pre-trained**),
  - [`BertForSequenceClassification`](./pytorch_pretrained_bert/modeling.py#L814) - BERT Transformer with a sequence classification head on top (BERT Transformer is **pre-trained**, the sequence classification head **is only initialized and has to be trained**),
  - [`BertForMultipleChoice`](./pytorch_pretrained_bert/modeling.py#L880) - BERT Transformer with a multiple choice head on top (used for task like Swag) (BERT Transformer is **pre-trained**, the multiple choice classification head **is only initialized and has to be trained**),
  - [`BertForTokenClassification`](./pytorch_pretrained_bert/modeling.py#L949) - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**, the token classification head **is only initialized and has to be trained**),
  - [`BertForQuestionAnswering`](./pytorch_pretrained_bert/modeling.py#L1015) - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**, the token classification head **is only initialized and has to be trained**).

- Three tokenizers (in the [`tokenization.py`](./pytorch_pretrained_bert/tokenization.py) file):
  - `BasicTokenizer` - basic tokenization (punctuation splitting, lower casing, etc.),
  - `WordpieceTokenizer` - WordPiece tokenization,
  - `BertTokenizer` - perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

- One optimizer (in the [`optimization.py`](./pytorch_pretrained_bert/optimization.py) file):
  - `BertAdam` - Bert version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.

- A configuration class (in the [`modeling.py`](./pytorch_pretrained_bert/modeling.py) file):
  - `BertConfig` - Configuration class to store the configuration of a `BertModel` with utilisities to read and write from JSON configuration files.

The repository further comprises:

- Four examples on how to use Bert (in the [`examples` folder](./examples)):
  - [`extract_features.py`](./examples/extract_features.py) - Show how to extract hidden states from an instance of `BertModel`,
  - [`run_classifier.py`](./examples/run_classifier.py) - Show how to fine-tune an instance of `BertForSequenceClassification` on GLUE's MRPC task,
  - [`run_squad.py`](./examples/run_squad.py) - Show how to fine-tune an instance of `BertForQuestionAnswering` on SQuAD v1.0 task.
  - [`run_swag.py`](./examples/run_swag.py) - Show how to fine-tune an instance of `BertForMultipleChoice` on Swag task.

  These examples are detailed in the [Examples](#examples) section of this readme.

- Three notebooks that were used to check that the TensorFlow and PyTorch models behave identically (in the [`notebooks` folder](./notebooks)):
  - [`Comparing-TF-and-PT-models.ipynb`](./notebooks/Comparing-TF-and-PT-models.ipynb) - Compare the hidden states predicted by `BertModel`,
  - [`Comparing-TF-and-PT-models-SQuAD.ipynb`](./notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb) - Compare the spans predicted by  `BertForQuestionAnswering` instances,
  - [`Comparing-TF-and-PT-models-MLM-NSP.ipynb`](./notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb) - Compare the predictions of the `BertForPretraining` instances.

  These notebooks are detailed in the [Notebooks](#notebooks) section of this readme.

- A command-line interface to convert any TensorFlow checkpoint in a PyTorch dump:

  This CLI is detailed in the [Command-line interface](#Command-line-interface) section of this readme.

## Usage

Here is a quick-start example using `BertTokenizer`, `BertModel` and `BertForMaskedLM` class with Google AI's pre-trained `Bert base uncased` model. See the [doc section](#doc) below for all the details on these classes.

First let's prepare a tokenized input with `BertTokenizer`

```python
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```

Let's see how to use `BertModel` to get hidden states

```python
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12
```

And how to use `BertForMaskedLM`

```python
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
```

## Doc

Here is a detailed documentation of the classes in the package and how to use them:

| Sub-section | Description |
|-|-|
| [Loading Google AI's pre-trained weigths](#Loading-Google-AIs-pre-trained-weigths-and-PyTorch-dump) | How to load Google AI's pre-trained weight or a PyTorch saved instance |
| [PyTorch models](#PyTorch-models) | API of the eight PyTorch model classes: `BertModel`, `BertForMaskedLM`, `BertForNextSentencePrediction`, `BertForPreTraining`, `BertForSequenceClassification`, `BertForMultipleChoice` or `BertForQuestionAnswering` |
| [Tokenizer: `BertTokenizer`](#Tokenizer-BertTokenizer) | API of the `BertTokenizer` class|
| [Optimizer: `BertAdam`](#Optimizer-BertAdam) |  API of the `BertAdam` class |

### Loading Google AI's pre-trained weigths and PyTorch dump

To load one of Google AI's pre-trained models or a PyTorch saved model (an instance of `BertForPreTraining` saved with `torch.save()`), the PyTorch model classes and the tokenizer can be instantiated as

```python
model = BERT_CLASS.from_pretrain(PRE_TRAINED_MODEL_NAME_OR_PATH, cache_dir=None)
```

where

- `BERT_CLASS` is either the `BertTokenizer` class (to load the vocabulary) or one of the eight PyTorch model classes (to load the pre-trained weights): `BertModel`, `BertForMaskedLM`, `BertForNextSentencePrediction`, `BertForPreTraining`, `BertForSequenceClassification`, `BertForTokenClassification`, `BertForMultipleChoice` or `BertForQuestionAnswering`, and
- `PRE_TRAINED_MODEL_NAME_OR_PATH` is either:

  - the shortcut name of a Google AI's pre-trained model selected in the list:

    - `bert-base-uncased`: 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-large-uncased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `bert-base-cased`: 12-layer, 768-hidden, 12-heads , 110M parameters
    - `bert-large-cased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `bert-base-multilingual-uncased`: (Orig, not recommended) 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-base-multilingual-cased`: **(New, recommended)** 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-base-chinese`: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

  - a path or url to a pretrained model archive containing:

    - `bert_config.json` a configuration file for the model, and
    - `pytorch_model.bin` a PyTorch dump of a pre-trained instance `BertForPreTraining` (saved with the usual `torch.save()`)

  If `PRE_TRAINED_MODEL_NAME_OR_PATH` is a shortcut name, the pre-trained weights will be downloaded from AWS S3 (see the links [here](pytorch_pretrained_bert/modeling.py)) and stored in a cache folder to avoid future download (the cache folder can be found at `~/.pytorch_pretrained_bert/`).
- `cache_dir` can be an optional path to a specific directory to download and cache the pre-trained model weights. This option is useful in particular when you are using distributed training: to avoid concurrent access to the same weights you can set for example `cache_dir='./pretrained_model_{}'.format(args.local_rank)` (see the section on distributed training for more information)

`Uncased` means that the text has been lowercased before WordPiece tokenization, e.g., `John Smith` becomes `john smith`. The Uncased model also strips out any accent markers. `Cased` means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging). For information about the Multilingual and Chinese model, see the [Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md) or the original TensorFlow repository.

**When using an `uncased model`, make sure to pass `--do_lower_case` to the training scripts. (Or pass `do_lower_case=True` directly to FullTokenizer if you're using your own script.)**

Example:
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

### PyTorch models

#### 1. `BertModel`

`BertModel` is the basic BERT Transformer model with a layer of summed token, position and sequence embeddings followed by a series of identical self-attention blocks (12 for BERT-base, 24 for BERT-large).

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as *inputs*:
[`modeling.py`](./pytorch_pretrained_bert/modeling.py)
- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts [`extract_features.py`](./examples/extract_features.py), [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`](./examples/run_squad.py)), and
- `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
- `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if some input sequence lengths are smaller than the max input sequence length of the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.
- `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

This model *outputs* a tuple composed of:

- `encoded_layers`: controled by the value of the `output_encoded_layers` argument:

  - `output_all_encoded_layers=True`: outputs a list of the encoded-hidden-states at the end of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
  - `output_all_encoded_layers=False`: outputs only the encoded-hidden-states corresponding to the last attention block, i.e. a single torch.FloatTensor of size [batch_size, sequence_length, hidden_size],

- `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

An example on how to use this class is given in the [`extract_features.py`](./examples/extract_features.py) script which can be used to extract the hidden states of the model for a given input.

#### 2. `BertForPreTraining`

`BertForPreTraining` includes the `BertModel` Transformer followed by the two pre-training heads:

- the masked language modeling head, and
- the next sentence classification head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus two optional labels:

- `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size]
- `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size] with indices selected in [0, 1]. 0 => next sentence is the continuation, 1 => next sentence is a random sentence.

*Outputs*:

- if `masked_lm_labels` and `next_sentence_label` are not `None`: Outputs the total_loss which is the sum of the masked language modeling loss and the next sentence classification loss.
- if `masked_lm_labels` or `next_sentence_label` is `None`: Outputs a tuple comprising

  - the masked language modeling logits, and
  - the next sentence classification logits.

#### 3. `BertForMaskedLM`

`BertForMaskedLM` includes the `BertModel` Transformer followed by the (possibly) pre-trained  masked language modeling head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus optional label:

- `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size]

*Outputs*:

- if `masked_lm_labels` is not `None`: Outputs the masked language modeling loss.
- if `masked_lm_labels` is `None`: Outputs the masked language modeling logits.

#### 4. `BertForNextSentencePrediction`

`BertForNextSentencePrediction` includes the `BertModel` Transformer followed by the next sentence classification head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus an optional label:

- `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size] with indices selected in [0, 1]. 0 => next sentence is the continuation, 1 => next sentence is a random sentence.

*Outputs*:

- if `next_sentence_label` is not `None`: Outputs the next sentence classification loss.
- if `next_sentence_label` is `None`: Outputs the next sentence classification logits.

#### 5. `BertForSequenceClassification`

`BertForSequenceClassification` is a fine-tuning model that includes `BertModel` and a sequence-level (sequence or pair of sequences) classifier on top of the `BertModel`.

The sequence-level classifier is a linear layer that takes as input the last hidden state of the first character in the input sequence (see Figures 3a and 3b in the BERT paper).

An example on how to use this class is given in the [`run_classifier.py`](./examples/run_classifier.py) script which can be used to fine-tune a single sequence (or pair of sequence) classifier using BERT, for example for the MRPC task.

#### 6. `BertForMultipleChoice`

`BertForMultipleChoice` is a fine-tuning model that includes `BertModel` and a linear layer on top of the `BertModel`.

The linear layer outputs a single value for each choice of a multiple choice problem, then all the outputs corresponding to an instance are passed through a softmax to get the model choice.

This implementation is largely inspired by the work of OpenAI in [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) and the answer of Jacob Devlin in the following [issue](https://github.com/google-research/bert/issues/38).

An example on how to use this class is given in the [`run_swag.py`](./examples/run_swag.py) script which can be used to fine-tune a multiple choice classifier using BERT, for example for the Swag task.

#### 7. `BertForTokenClassification`

`BertForTokenClassification` is a fine-tuning model that includes `BertModel` and a token-level classifier on top of the `BertModel`.

The token-level classifier is a linear layer that takes as input the last hidden state of the sequence.

#### 8. `BertForQuestionAnswering`

`BertForQuestionAnswering` is a fine-tuning model that includes `BertModel` with a token-level classifiers on top of the full sequence of last hidden states.

The token-level classifier takes as input the full sequence of the last hidden state and compute several (e.g. two) scores for each tokens that can for example respectively be the score that a given token is a `start_span` and a `end_span` token (see Figures 3c and 3d in the BERT paper).

An example on how to use this class is given in the [`run_squad.py`](./examples/run_squad.py) script which can be used to fine-tune a token classifier using BERT, for example for the SQuAD task.

### Tokenizer: `BertTokenizer`

`BertTokenizer` perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

This class has two arguments:

- `vocab_file`: path to a vocabulary file.
- `do_lower_case`: convert text to lower-case while tokenizing. **Default = True**.

and three methods:

- `tokenize(text)`: convert a `str` in a list of `str` tokens by (1) performing basic tokenization and (2) WordPiece tokenization.
- `convert_tokens_to_ids(tokens)`: convert a list of `str` tokens in a list of `int` indices in the vocabulary.
- `convert_ids_to_tokens(tokens)`: convert a list of `int` indices in a list of `str` tokens in the vocabulary.

Please refer to the doc strings and code in [`tokenization.py`](./pytorch_pretrained_bert/tokenization.py) for the details of the `BasicTokenizer` and `WordpieceTokenizer` classes. In general it is recommended to use `BertTokenizer` unless you know what you are doing.

### Optimizer: `BertAdam`

`BertAdam` is a `torch.optimizer` adapted to be closer to the optimizer used in the TensorFlow implementation of Bert. The differences with PyTorch Adam optimizer are the following:

- BertAdam implements weight decay fix,
- BertAdam doesn't compensate for bias as in the regular Adam optimizer.

The optimizer accepts the following arguments:

- `lr` : learning rate
- `warmup` : portion of `t_total` for the warmup, `-1`  means no warmup. Default : `-1`
- `t_total` : total number of training steps for the learning
    rate schedule, `-1`  means constant learning rate. Default : `-1`
- `schedule` : schedule to use for the warmup (see above). Default : `'warmup_linear'`
- `b1` : Adams b1. Default : `0.9`
- `b2` : Adams b2. Default : `0.999`
- `e` : Adams epsilon. Default : `1e-6`
- `weight_decay:` Weight decay. Default : `0.01`
- `max_grad_norm` : Maximum norm for the gradients (`-1` means no clipping). Default : `1.0`

## Examples

| Sub-section | Description |
|-|-|
| [Training large models: introduction, tools and examples](#Training-large-models-introduction,-tools-and-examples) | How to use gradient-accumulation, multi-gpu training, distributed training, optimize on CPU and 16-bits training to train Bert models |
| [Fine-tuning with BERT: running the examples](#Fine-tuning-with-BERT-running-the-examples) | Running the examples in [`./examples`](./examples/): `extract_classif.py`, `run_classifier.py` and `run_squad.py` |
| [Fine-tuning BERT-large on GPUs](#Fine-tuning-BERT-large-on-GPUs) | How to fine tune `BERT large`|

### Training large models: introduction, tools and examples

BERT-base and BERT-large are respectively 110M and 340M parameters models and it can be difficult to fine-tune them on a single GPU with the recommended batch size for good performance (in most case a batch size of 32).

To help with fine-tuning these models, we have included several techniques that you can activate in the fine-tuning scripts [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`](./examples/run_squad.py): gradient-accumulation, multi-gpu training, distributed training and 16-bits training . For more details on how to use these techniques you can read [the tips on training large batches in PyTorch](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) that I published earlier this month.

Here is how to use these techniques in our scripts:

- **Gradient Accumulation**: Gradient accumulation can be used by supplying a integer greater than 1 to the `--gradient_accumulation_steps` argument. The batch at each step will be divided by this integer and gradient will be accumulated over `gradient_accumulation_steps` steps.
- **Multi-GPU**: Multi-GPU is automatically activated when several GPUs are detected and the batches are splitted over the GPUs.
- **Distributed training**: Distributed training can be activated by supplying an integer greater or equal to 0 to the `--local_rank` argument (see below).
- **16-bits training**: 16-bits training, also called mixed-precision training, can reduce the memory requirement of your model on the GPU by using half-precision training, basically allowing to double the batch size. If you have a recent GPU (starting from NVIDIA Volta architecture) you should see no decrease in speed. A good introduction to Mixed precision training can be found [here](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) and a full documentation is [here](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html). In our scripts, this option can be activated by setting the `--fp16` flag and you can play with loss scaling using the `--loss_scaling` flag (see the previously linked documentation for details on loss scaling). If the loss scaling is too high (`Nan` in the gradients) it will be automatically scaled down until the value is acceptable. The default loss scaling is 128 which behaved nicely in our tests.

Note: To use *Distributed Training*, you will need to run one training script on each of your machines. This can be done for example by running the following command on each server (see [the above mentioned blog post]((https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)) for more details):
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=$THIS_MACHINE_INDEX --master_addr="192.168.1.1" --master_port=1234 run_classifier.py (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)
```
Where `$THIS_MACHINE_INDEX` is an sequential index assigned to each of your machine (0, 1, 2...) and the machine with rank 0 has an IP address `192.168.1.1` and an open port `1234`.

### Fine-tuning with BERT: running the examples

We showcase several fine-tuning examples based on (and extended from) [the original implementation](https://github.com/google-research/bert/):

- a *sequence-level classifier* on the MRPC classification corpus,
- a *token-level classifier* on the question answering dataset SQuAD, and
- a *sequence-level multiple-choice classifier* on the SWAG classification corpus.

#### MRPC

This example code fine-tunes BERT on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running this example you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation results between 84% and 88%.

**Fast run with apex and 16 bit precision: fine-tuning on MRPC in 27 seconds!**
First install apex as indicated [here](https://github.com/NVIDIA/apex).
Then run
```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

#### SQuAD

This example code fine-tunes BERT on the SQuAD dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large) on a single tesla V100 16GB.

The data for SQuAD can be downloaded with the following links and should be saved in a `$SQUAD_DIR` directory.

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previous hyper-parameters gave us the following results:
```bash
{"f1": 88.52381567990474, "exact_match": 81.22043519394512}
```

#### SWAG

The data for SWAG can be downloaded by cloning the following [repository](https://github.com/rowanz/swagaf)

```shell
export SWAG_DIR=/path/to/SWAG

python run_swag.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir /tmp/swag_output/ \
  --gradient_accumulation_steps 4
```

Training with the previous hyper-parameters on a single GPU gave us the following results:
```
eval_accuracy = 0.8062081375587323
eval_loss = 0.5966546792367169
global_step = 13788
loss = 0.06423990014260186
```

## Fine-tuning BERT-large on GPUs

The options we list above allow to fine-tune BERT-large rather easily on GPU(s) instead of the TPU used by the original implementation.

For example, fine-tuning BERT-large on SQuAD can be done on a server with 4 k-80 (these are pretty old now) in 18 hours. Our results are similar to the TensorFlow implementation results (actually slightly higher):
```bash
{"exact_match": 84.56953642384106, "f1": 91.04028647786927}
```
To get these results we used a combination of:
- multi-GPU training (automatically activated on a multi-GPU server),
- 2 steps of gradient accumulation and
- perform the optimization step on CPU to store Adam's averages in RAM.

Here is the full list of hyper-parameters for this run:
```bash
python ./run_squad.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_EVAL \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 24 \
  --gradient_accumulation_steps 2 \
  --optimize_on_cpu
```

If you have a recent GPU (starting from NVIDIA Volta series), you should try **16-bit fine-tuning** (FP16).

Here is an example of hyper-parameters for a FP16 run we tried:
```bash
python ./run_squad.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_EVAL \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 24 \
  --fp16 \
  --loss_scale 128
```

The results were similar to the above FP32 results (actually slightly higher):
```bash
{"exact_match": 84.65468306527909, "f1": 91.238669287002}
```

## Notebooks

We include [three Jupyter Notebooks](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/notebooks) that can be used to check that the predictions of the PyTorch model are identical to the predictions of the original TensorFlow model.

- The first NoteBook ([Comparing-TF-and-PT-models.ipynb](./notebooks/Comparing-TF-and-PT-models.ipynb)) extracts the hidden states of a full sequence on each layers of the TensorFlow and the PyTorch models and computes the standard deviation between them. In the given example, we get a standard deviation of 1.5e-7 to 9e-7 on the various hidden state of the models.

- The second NoteBook ([Comparing-TF-and-PT-models-SQuAD.ipynb](./notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb)) compares the loss computed by the TensorFlow and the PyTorch models for identical initialization of the fine-tuning layer of the `BertForQuestionAnswering` and computes the standard deviation between them. In the given example, we get a standard deviation of 2.5e-7 between the models.

- The third NoteBook ([Comparing-TF-and-PT-models-MLM-NSP.ipynb](./notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb)) compares the predictions computed by the TensorFlow and the PyTorch models for masked token language modeling using the pre-trained masked language modeling model.

Please follow the instructions given in the notebooks to run and modify them.

## Command-line interface

A command-line interface is provided to convert a TensorFlow checkpoint in a PyTorch dump of the `BertForPreTraining` class  (see above).

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`./pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This CLI takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()` (see examples in [`extract_features.py`](./examples/extract_features.py), [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`]((./examples/run_squad.py))).

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

## TPU

TPU support and pretraining scripts

TPU are not supported by the current stable release of PyTorch (0.4.1). However, the next version of PyTorch (v1.0) should support training on TPU and is expected to be released soon (see the recent [official announcement](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud)).

We will add TPU support when this next release is published.

The original TensorFlow code further comprises two scripts for pre-training BERT: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) and [run_pretraining.py](https://github.com/google-research/bert/blob/master/run_pretraining.py).

Since, pre-training BERT is a particularly expensive operation that basically requires one or several TPUs to be completed in a reasonable amout of time (see details [here](https://github.com/google-research/bert#pre-training-with-bert)) we have decided to wait for the inclusion of TPU support in PyTorch to convert these pre-training scripts.
