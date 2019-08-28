# DilBERT

This section contains examples showcasing how to use DilBERT and the original code to train DilBERT.

## What is DilBERT?

DilBERT stands for DistiLlation-BERT. DilBERT is a small, fast, cheap and light Transformer model: it has 40% less parameters than `bert-base-uncased`, runs 40% faster while preserving 96% on the language understanding capabilties (as shown on the GLUE benchmark). DilBERT is trained by distillation: a technique to compress a large model called the teacher into a smaller model called the student. By applying this compression technique, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model, while being lighter, smaller and faster. Thus, DilBERT can be an interesting solution to put large Transformer model into production.

For more information on DilBERT, we refer to [our blog post](TODO(Link)).

## How to use DilBERT?

PyTorch-Transformers includes two pre-trained models:
- `dilbert-base-uncased`: The language model pretrained by distillation under the supervision of `bert-base-uncased`. The model has 6 layers, 768 dimension and 12 heads, totalizing 66M parameters.
- `dilbert-base-uncased-distilled-squad`: The `dilbert-base-uncased` finetune by distillation on SQuAD. It reaches a F1 score of 86.2 on the dev set, while `bert-base-uncased` reaches a 88.5 F1 score.

Using DilBERT is really similar to using BERT. DilBERT uses the same tokenizer as BERT and more specifically `bert-base-uncased`. You should only use this tookenizer as the only pre-trained weights available for now are supervised by `bert-base-uncased`.

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = DilBertModel.from_pretrained('dilbert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

## How to train DilBERT?

In the following, we will explain how you can train your own compressed model.

### A. Preparing the data

The weights we release are trained using a concatenation of Toronto Book Corpus and English Wikipedia (same training data as BERT).

To avoid processing the data several time, we do it once and for all before the training. From now on, will suppose that you have a text file `dump.txt` which contains one sequence per line (a sequence being composed of one of several coherent sentences).

First, we will binarize the data: we tokenize the data and associate each token to an id.

```bash
python scripts/binarized_data.py \
    --file_path data/dump.txt \
    --bert_tokenizer bert-base-uncased \
    --dump_file data/binarized_text
```

In the masked language modeling loss, we follow [XLM](https://github.com/facebookresearch/XLM) and smooth the probability of masking with a factor that put more emphasis on rare words. Thus we count the occurences of each tokens in the data:

```bash
python scripts/token_counts.py \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts_dump data/token_counts.bert-base-uncased.pickle
```

### B. Training

Launching a distillation is really simple once you have setup the data:

```bash
python train.py \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.bert-base-uncased.pickle \
    --token_counts data/token_counts.bert-base-uncased.pickle \
    --force # It overwrites the `dump_path` if it already exists.
``` 

By default, this will launch a training on a single GPU (even if more are available on the cluster). Other parameters are available in the command line, please refer to `train.py`.

We also highly encourage using distributed training. Here's an example that launchs a distributed traininng on a single node with 4 GPUs:
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
        --data_file data/dump_concat_wiki_toronto_bk.bert-base-uncased.pickle \
        --token_counts data/token_counts_concat_wiki_toronto_bk.bert-base-uncased.pickle \
        --dump_path serialization_dir/with_transform/last_word
```

**Tips** Start the distillation from some sort of structure initialization is crucial to reach a good final performance. In our experiments, we use initialization from some of the layers of the teacher itself! Please refer to `scripts/extract_for_distil.py` to create a valid initialization checkpoint and add `from_pretrained_weights` and `from_pretrained_config` when launching your distillation!

Happy distillation!
