# PyTorch implementation of Google AI's BERT model

## Introduction

This is an op-for-op PyTorch reimplementation of the [TensorFlow code](https://github.com/google-research/bert) released by Google AI with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

This implementation can load any pre-trained TensorFlow BERT checkpoint in a PyTorch model (see below).

There are a few differences with the TensorFlow model:

- this PyTorch implementation support multi-GPU and distributed training (see below),
- the current stable version of PyTorch (0.4.1) doesn't support TPU training and as a consequence, the pre-training script are not included in this repo (see below). TPU support is supposed to be available in PyTorch v1.0. We will update the repository with TPU-adapted pre-training scripts at that time. In the meantime, you can use the TensorFlow version to train a model on TPU and import a TensorFlow checkpoint as described below.

## Loading a TensorFlow checkpoint (in particular Google's pre-trained models) in the Pytorch model

You can convert any TensorFlow checkpoint, and in particular the pre-trained weights released by GoogleAI, by using `convert_tf_checkpoint_to_pytorch.py`.

This script takes as input a TensorFlow checkpoint (`bert_model.ckpt`) load it in the PyTorch model and save the model in a standard PyTorch model save file that can be imported using the usual `torch.load()` command (see the `run_classifier.py` script for an example).

TensorFlow pre-trained models can be found in the [original TensorFlow code](https://github.com/google-research/bert). Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin
```

## Multi-GPU and Distributed Training

Multi-GPU is automatically activated in the scripts when multiple GPUs are detected.

Distributed training is activated by suppying a `--local_rank` arguments to the `run_classifier.py` or the `run_squad.py` scripts.

For more information on how to use distributed training with PyTorch, you can read [this simple introduction](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) we wrote earlier this month.

## Fine-tuning with BERT: running the examples

We showcase the same examples as in the original implementation: fine-tuning on the MRPC classification corpus and the question answering dataset SQUAD.

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

The next example fine-tunes `BERT-Base` on the SQuAD question answering task.

The data for SQuAD can be downloaded with the following links and should be saved in a `$SQUAD_DIR` directory.

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_PYTORCH_DIR/pytorch_model.bin \
  --do_train \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=../debug_squad/
```

## Comparing TensorFlow and PyTorch models

We also include [a simple Jupyter Notebook](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/Comparing%20TF%20and%20PT%20models.ipynb) that can be used to check that the predictions of the PyTorch model are identical to the predictions of the original TensorFlow model (within the rounding errors and the differing backend implementations of the operations, in our case we found a standard deviation of about 4e-7 on the last hidden state of the 12th layer). Please follow the instructions in the Notebook to run it.

## Note on pre-training

The original TensorFlow code comprise two scripts that can be used for pre-training BERT: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) and [run_pretraining.py](https://github.com/google-research/bert/blob/master/run_pretraining.py).
As the authors notice, pre-training BERT is particularly expensive and requires TPU to run in a reasonable amout of time (see [here](https://github.com/google-research/bert#pre-training-with-bert)).

We have decided to wait for the up-coming release of PyTorch v1.0 which is expected support training on TPU for porting these scripts (see the recent [official announcement](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud)).

## Requirements

The main dependencies of this code are:

- PyTorch (>= 0.4.0)
- tqdm

To install the dependencies:

````bash
pip install -r ./requirements.txt
````
