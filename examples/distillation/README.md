# Distil*

This folder contains the original code used to train Distil* as well as examples showcasing how to use DistilBERT and DistilGPT2.

**2019, October 3rd - Update** We release our [NeurIPS workshop paper](https://arxiv.org/abs/1910.01108) explaining our approach on **DistilBERT**. It includes updated results and further experiments. We applied the same method to GPT2 and release the weights of **DistilGPT2**. DistilGPT2 is two times faster and 33% smaller than GPT2.

**2019, September 19th - Update:** We fixed bugs in the code and released an upadted version of the weights trained with a modification of the distillation loss. DistilBERT now reaches 97% of `BERT-base`'s performance on GLUE, and 86.9 F1 score on SQuAD v1.1 dev set (compared to 88.5 for `BERT-base`). We will publish a formal write-up of our approach in the near future!

## What is Distil*

Distil* is a class of compressed models that started with DistilBERT. DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than `bert-base-uncased`, runs 60% faster while preserving 97% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is trained using knowledge distillation, a technique to compress a large model called the teacher into a smaller model called the student. By distillating Bert, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilBERT is thus an interesting option to put large-scaled trained Transformer model into production.

We have applied the same method to GPT2 and release the weights of the compressed model. On the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) benchmark, GPT2 reaches a perplexity on the test set of 15.0 compared to 18.5 for DistilGPT2 (after fine-tuning on the train set).

For more information on DistilBERT, please refer to our [NeurIPS workshop paper](https://arxiv.org/abs/1910.01108). The paper superseeds our [previous blogpost](https://medium.com/huggingface/distilbert-8cf3380435b5) with a different distillation loss and better performances.

Here are the results on the dev sets of GLUE:

| Model      | Macro-score | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2| STS-B| WNLI |
| :---:      |    :---:    | :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:| :---:|
| BERT-base  |  **77.6**   | 48.9 | 84.3 | 88.6 | 89.3 | 89.5 | 71.3 | 91.7 | 91.2 | 43.7 |
| DistilBERT |  **76.8**   | 49.1 | 81.8 | 90.2 | 90.2 | 89.2 | 62.9 | 92.7 | 90.7 | 44.4 |

## Setup

This part of the library has only be tested with Python3.6+. There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`. 

**Important note:** The training scripts have been updated to support PyTorch v1.2.0 (there are breakings changes compared to v1.1.0). It is important to note that there is a small internal bug in the current version of PyTorch available on pip that causes a memory leak in our training/distillation. It has been recently fixed and will likely be integrated into the next release. For the moment, we recommend to [compile PyTorch from source](https://github.com/pytorch/pytorch#from-source). Please refer to [issue 1179](https://github.com/huggingface/transformers/issues/1179) for more details.

## How to use DistilBERT

Transformers includes two pre-trained Distil* models, currently only provided for English (we are investigating the possibility to train and release a multilingual version of DistilBERT):

- `distilbert-base-uncased`: DistilBERT English language model pretrained on the same data used to pretrain Bert (concatenation of the Toronto Book Corpus and full English Wikipedia) using distillation with the supervision of the `bert-base-uncased` version of Bert. The model has 6 layers, 768 dimension and 12 heads, totalizing 66M parameters.
- `distilbert-base-uncased-distilled-squad`: A finetuned version of `distilbert-base-uncased` finetuned using (a second step of) knwoledge distillation on SQuAD 1.0. This model reaches a F1 score of 86.9 on the dev set (for comparison, Bert `bert-base-uncased` version reaches a 88.5 F1 score).
- `distilgpt2`: DistilGPT2 English language model pretrained with the supervision of `gpt2` (the smallest version of GPT2) on [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a reproduction of OpenAI's WebText dataset and . The model has 6 layers, 768 dimension and 12 heads, totalizing 82M (compared to 124M parameters for GPT2). On average, DistilGPT2 is two times faster than GPT2.
- and more to come! 🤗🤗🤗

Using DistilBERT is very similar to using BERT. DistilBERT share the same tokenizer as BERT's `bert-base-uncased` even though we provide a link to this tokenizer under the `DistilBertTokenizer` name to have a consistent naming between the library models.

```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

Similarly, using DistilGPT2 simply consists in calling the GPT2 classes from a different pretrained checkpoint: `model = GPT2Model.from_pretrained('distilgpt2')`.

## How to train Distil*

In the following, we will explain how you can train DistilBERT.

### A. Preparing the data

The weights we release are trained using a concatenation of Toronto Book Corpus and English Wikipedia (same training data as the English version of BERT).

To avoid processing the data several time, we do it once and for all before the training. From now on, will suppose that you have a text file `dump.txt` which contains one sequence per line (a sequence being composed of one of several coherent sentences).

First, we will binarize the data, i.e. tokenize the data and convert each token in an index in our model's vocabulary.

```bash
python scripts/binarized_data.py \
    --file_path data/dump.txt \
    --tokenizer_type bert \
    --tokenizer_name bert-base-uncased \
    --dump_file data/binarized_text
```

Our implementation of masked language modeling loss follows [XLM](https://github.com/facebookresearch/XLM)'s one and smoothes the probability of masking with a factor that put more emphasis on rare words. Thus we count the occurences of each tokens in the data:

```bash
python scripts/token_counts.py \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts_dump data/token_counts.bert-base-uncased.pickle \
    --vocab_size 30522
```

### B. Training

Training with distillation is really simple once you have pre-processed the data:

```bash
python train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-uncased \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --mlm \
    --freeze_pos_embs \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts data/token_counts.bert-base-uncased.pickle \
    --force # overwrites the `dump_path` if it already exists.
```

By default, this will launch a training on a single GPU (even if more are available on the cluster). Other parameters are available in the command line, please look in `train.py` or run `python train.py --help` to list them.

We highly encourage you to use distributed training for training DistilBERT as the training corpus is quite large. Here's an example that runs a distributed training on a single node having 4 GPUs:

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
        --student_type distilbert \
        --student_config training_configs/distilbert-base-uncased.json \
        --teacher_type bert \
        --teacher_name bert-base-uncased \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --mlm \
        --freeze_pos_embs \
        --dump_path serialization_dir/my_first_training \
        --data_file data/binarized_text.bert-base-uncased.pickle \
        --token_counts data/token_counts.bert-base-uncased.pickle
```

**Tips:** Starting distillated training with good initialization of the model weights is crucial to reach decent performance. In our experiments, we initialized our model from a few layers of the teacher (Bert) itself! Please refer to `scripts/extract.py` and `scripts/extract_distilbert.py` to create a valid initialization checkpoint and use `--student_pretrained_weights` argument to use this initialization for the distilled training!

Happy distillation!
