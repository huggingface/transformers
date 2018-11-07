# PyTorch implementation of Google AI's BERT model with a script to load Google's pre-trained models

## Introduction

This repository contains an op-for-op PyTorch reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)) and a conversion script is provided (see below).

The code to use, in addition, [the Multilingual and Chinese models](https://github.com/google-research/bert/blob/master/multilingual.md) will be added later this week (it's actually just the tokenization code that needs to be updated).

## Loading a TensorFlow checkpoint (e.g. [Google's pre-trained models](https://github.com/google-research/bert#pre-trained-models))

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This script takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()` (see examples in `extract_features.py`, `run_classifier.py` and `run_squad.py`).

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

## PyTorch models for BERT

We included three PyTorch models in this repository that you will find in [`modeling.py`](modeling.py):

- `BertModel` - the basic BERT Transformer model
- `BertForSequenceClassification` - the BERT model with a sequence classification head on top
- `BertForQuestionAnswering` - the BERT model with a token classification head on top

Here are some details on each class.

### 1. `BertModel`

`BertModel` is the basic BERT Transformer model with a layer of summed token, position and sequence embeddings followed by a series of identical self-attention blocks (12 for BERT-base, 24 for BERT-large).

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as inputs:

- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`), and
- `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
- `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.

This model outputs a tuple composed of:

- `all_encoder_layers`: a list of torch.FloatTensor of size [batch_size, sequence_length, hidden_size] which is a list of the full sequences of hidden-states at the end of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), and
- `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

An example on how to use this class is given in the `extract_features.py` script which can be used to extract the hidden states of the model for a given input.

### 2. `BertForSequenceClassification`

`BertForSequenceClassification` is a fine-tuning model that includes `BertModel` and a sequence-level (sequence or pair of sequences) classifier on top of the `BertModel`.

The sequence-level classifier is a linear layer that takes as input the last hidden state of the first character in the input sequence (see Figures 3a and 3b in the BERT paper).

An example on how to use this class is given in the `run_classifier.py` script which can be used to fine-tune a single sequence (or pair of sequence) classifier using BERT, for example for the MRPC task.

### 3. `BertForQuestionAnswering`

`BertForQuestionAnswering` is a fine-tuning model that includes `BertModel` with a token-level classifiers on top of the full sequence of last hidden states.

The token-level classifier takes as input the full sequence of the last hidden state and compute several (e.g. two) scores for each tokens that can for example respectively be the score that a given token is a `start_span` and a `end_span` token (see Figures 3c and 3d in the BERT paper).

An example on how to use this class is given in the `run_squad.py` script which can be used to fine-tune a token classifier using BERT, for example for the SQuAD task.

## Installation, requirements, test

This code was tested on Python 3.5+. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm

To install the dependencies:

````bash
pip install -r ./requirements.txt
````

A series of tests is included in the [tests folder](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/tests) and can be run using `pytest` (install pytest if needed: `pip install pytest`).

You can run the tests with the command:
```bash
python -m pytest -sv tests/
```

## Training on large batches: gradient accumulation, multi-GPU and distributed training

BERT-base and BERT-large are respectively 110M and 340M parameters models and it can be difficult to fine-tune them on a single GPU with the recommended batch size for good performance (in most case a batch size of 32).

To help with fine-tuning these models, we have included three techniques that you can activate in the fine-tuning scripts `run_classifier.py` and `run_squad.py`: gradient-accumulation, multi-gpu and distributed training. For more details on how to use these techniques you can read [the tips on training large batches in PyTorch](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) that I published earlier this month.

Here is how to use these techniques in our scripts:

- **Gradient Accumulation**: Gradient accumulation can be used by supplying a integer greater than 1 to the `--gradient_accumulation_steps` argument. The batch at each step will be divided by this integer and gradient will be accumulated over `gradient_accumulation_steps` steps.
- **Multi-GPU**: Multi-GPU is automatically activated when several GPUs are detected and the batches are splitted over the GPUs.
- **Distributed training**: Distributed training can be activated by suppying an integer greater or equal to 0 to the `--local_rank` argument. To use Distributed training, you will need to run one training script on each of your machines. This can be done for example by running the following command on each server (see the above blog post for more details):

```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=$THIS_MACHINE_INDEX --master_addr="192.168.1.1" --master_port=1234 run_classifier.py (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)
```

Where `$THIS_MACHINE_INDEX` is an sequential index assigned to each of your machine (0, 1, 2...) and the machine with rank 0 has an IP adress `192.168.1.1` and an open port `1234`.

## TPU support and pretraining scripts

TPU are not supported by the current stable release of PyTorch (0.4.1). However, the next version of PyTorch (v1.0) should support training on TPU and is expected to be released soon (see the recent [official announcement](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud)).

We will add TPU support when this next release is published.

The original TensorFlow code further comprises two scripts for pre-training BERT: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) and [run_pretraining.py](https://github.com/google-research/bert/blob/master/run_pretraining.py).

Since, pre-training BERT is a particularly expensive operation that basically requires one or several TPUs to be completed in a reasonable amout of time (see details [here](https://github.com/google-research/bert#pre-training-with-bert)) we have decided to wait for the inclusion of TPU support in PyTorch to convert these pre-training scripts.

## Comparing the PyTorch model and the TensorFlow model predictions

We also include [two Jupyter Notebooks](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/notebooks) that can be used to check that the predictions of the PyTorch model are identical to the predictions of the original TensorFlow model.

- The first NoteBook ([Comparing TF and PT models.ipynb](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing%20TF%20and%20PT%20models.ipynb)) extracts the hidden states of a full sequence on each layers of the TensorFlow and the PyTorch models and computes the sandard deviation between them. In the given example, we get a standard deviation of 1.5e-7 to 9e-7 on the various hidden state of the models.

- The second NoteBook ([Comparing TF and PT models SQuAD predictions.ipynb](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing%20TF%20and%20PT%20models%20SQuAD%20predictions.ipynb)) compares the loss computed by the TensorFlow and the PyTorch models for identical initialization of the fine-tuning layer of the `BertForQuestionAnswering` and computes the sandard deviation between them. In the given example, we get a standard deviation of 2.5e-7 between the models.

Please follow the instructions given in the notebooks to run and modify them. They can also be nice example on how to use the models in a simpler way than the full fine-tuning scripts we provide.

## Fine-tuning with BERT: running the examples

We showcase the same examples as [the original implementation](https://github.com/google-research/bert/): fine-tuning a sequence-level classifier on the MRPC classification corpus and a token-level classifier on the question answering dataset SQuAD.

Before running theses examples you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`. Please also download the `BERT-Base`
checkpoint, unzip it to some directory `$BERT_BASE_DIR`, and convert it to its PyTorch version as explained in the previous section.

This example code fine-tunes `BERT-Base` on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than 10 minutes on a single K-80.

```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation results between 84% and 88%.

The second example fine-tunes `BERT-Base` on the SQuAD question answering task.

The data for SQuAD can be downloaded with the following links and should be saved in a `$SQUAD_DIR` directory.

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --do_train \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --do_predict \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../debug_squad/
```
