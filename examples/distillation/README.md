# DistilBERT

This folder contains the original code used to train DistilBERT as well as examples showcasing how to use DistilBERT.

## What is DistilBERT

DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of Bert's performances as measured on the GLUE language understanding benchmark. DistilBERT is trained using knowledge distillation, a technique to compress a large model called the teacher into a smaller model called the student. By distillating Bert, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilBERT is thus an interesting option to put large-scaled trained Transformer model into production.

For more information on DistilBERT, please refer to our [detailed blog post](https://medium.com/huggingface/smaller-faster-cheaper-lighter-introducing-distilbert-a-distilled-version-of-bert-8cf3380435b5
).

## Setup

This part of the library has only be tested with Python3.6+. There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`. 

**Important note:** The training scripts have been updated to support PyTorch v1.2.0 (there are breakings changes compared to v1.1.0). It is important to note that there is a small internal bug in the current version of PyTorch available on pip that causes a memory leak in our training/distillation. It has been recently fixed and will likely be integrated into the next release. For the moment, we recommend to [compile PyTorch from source](https://github.com/pytorch/pytorch#from-source). Please refer to [issue 1179](https://github.com/huggingface/pytorch-transformers/issues/1179) for more details.

## How to use DistilBERT

PyTorch-Transformers includes two pre-trained DistilBERT models, currently only provided for English (we are investigating the possibility to train and release a multilingual version of DistilBERT):

- `distilbert-base-uncased`: DistilBERT English language model pretrained on the same data used to pretrain Bert (concatenation of the Toronto Book Corpus and full English Wikipedia) using distillation with the supervision of the `bert-base-uncased` version of Bert. The model has 6 layers, 768 dimension and 12 heads, totalizing 66M parameters.
- `distilbert-base-uncased-distilled-squad`: A finetuned version of `distilbert-base-uncased` finetuned using (a second step of) knwoledge distillation on SQuAD 1.0. This model reaches a F1 score of 86.2 on the dev set (for comparison, Bert `bert-base-uncased` version reaches a 88.5 F1 score).

Using DistilBERT is very similar to using BERT. DistilBERT share the same tokenizer as BERT's `bert-base-uncased` even though we provide a link to this tokenizer under the `DistilBertTokenizer` name to have a consistent naming between the library models.

```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

## How to train DistilBERT

In the following, we will explain how you can train your own compressed model.

### A. Preparing the data

The weights we release are trained using a concatenation of Toronto Book Corpus and English Wikipedia (same training data as the English version of BERT).

To avoid processing the data several time, we do it once and for all before the training. From now on, will suppose that you have a text file `dump.txt` which contains one sequence per line (a sequence being composed of one of several coherent sentences).

First, we will binarize the data, i.e. tokenize the data and convert each token in an index in our model's vocabulary.

```bash
python scripts/binarized_data.py \
    --file_path data/dump.txt \
    --bert_tokenizer bert-base-uncased \
    --dump_file data/binarized_text
```

Our implementation of masked language modeling loss follows [XLM](https://github.com/facebookresearch/XLM)'s one and smoothes the probability of masking with a factor that put more emphasis on rare words. Thus we count the occurences of each tokens in the data:

```bash
python scripts/token_counts.py \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts_dump data/token_counts.bert-base-uncased.pickle
```

### B. Training

Training with distillation is really simple once you have pre-processed the data:

```bash
python train.py \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts data/token_counts.bert-base-uncased.pickle \
    --force # overwrites the `dump_path` if it already exists.
```

By default, this will launch a training on a single GPU (even if more are available on the cluster). Other parameters are available in the command line, please look in `train.py` or run `python train.py --help` to list them.

We highly encourage you to use distributed training for training DistilBert as the training corpus is quite large. Here's an example that runs a distributed training on a single node having 4 GPUs:

```bash
export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=<AN_OPEN_PORT>
export MASTER_ADDR=<I.P.>

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --data_file data/binarized_text.bert-base-uncased.pickle \
        --token_counts data/token_counts.bert-base-uncased.pickle \
        --dump_path serialization_dir/my_first_distillation
```

**Tips:** Starting distillated training with good initialization of the model weights is crucial to reach decent performance. In our experiments, we initialized our model from a few layers of the teacher (Bert) itself! Please refer to `scripts/extract_for_distil.py` to create a valid initialization checkpoint and use `--from_pretrained_weights` and `--from_pretrained_config` arguments to use this initialization for the distilled training!

Happy distillation!
