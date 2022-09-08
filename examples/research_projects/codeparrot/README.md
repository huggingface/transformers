# CodeParrot 🦜
<p align="center">
    <img src="https://huggingface.co/datasets/lvwerra/repo-images/raw/main/code-highlighting-streamlit.png" alt="drawing" width="350"/>
</p>

## What is this about?
This is an open-source effort to train and evaluate code generation models. CodeParrot 🦜 is a GPT-2 model trained from scratch on Python code. The highlights of this project are:
- initialize and train a GPT-2 language model from scratch for code generation
- train a custom tokenizer adapted for Python code
- clean and deduplicate a large (>100GB) dataset with `datasets`
- train with `accelerate` on multiple GPUs using data parallelism and mixed precision
- continuously push checkpoints to the hub with `huggingface_hub`
- stream the dataset with `datasets` during training to avoid disk bottlenecks
- apply the `code_eval` metric in `datasets` to evaluate on [OpenAI's _HumanEval_ benchmark](https://huggingface.co/datasets/openai_humaneval)
- showcase examples for downstream tasks with code models in [examples](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot/examples) folder:
    - Algorithmic complexity prediction
    - Code generation from english text
    - Code explanation
    
## Installation
To install the dependencies simply run the following command:
```bash
pip install -r requirements.txt
```

To reproduce the results you can follow the scripts in the following sections. Note that we don't always show all possible arguments to the scripts. To get the full list of arguments with descriptions you can run the following command on any script:

```bash
python scripts/some_script.py --help
```

Before you run any of the scripts make sure you are logged in and can push to the hub:

```bash
huggingface-cli login
```

Additionally, sure you have git-lfs installed. You can find instructions for how to install it [here](https://git-lfs.github.com/).

## Dataset
The source of the dataset is the GitHub dump available on Google's [BigQuery](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code). The database was queried for all Python files with less than 1MB in size resulting in a 180GB dataset with over 20M files. The dataset is available on the Hugging Face Hub [here](https://huggingface.co/datasets/transformersbook/codeparrot).

### Preprocessing
The raw dataset contains many duplicates. We deduplicated and filtered the dataset using the heuristics proposed in OpenAI's Codex [paper](https://arxiv.org/abs/2107.03374) and some new ones:

- exact deduplication using each file's hash after having removed whistespaces.
- near deduplication using MinHash and Jaccard similarity. MinHash with a Jaccard threshold (default=0.85) is first used to create duplicate clusters. Then these clusters are then reduced to unique files based on the exact Jaccard similarity. See `deduplicate_dataset` in `minhash_deduplication.py` for a detailed description.
- filtering files with max line length > 1000
- filtering files with mean line length > 100
- fraction of alphanumeric characters < 0.25
- containing the word "auto-generated" or similar in the first 5 lines
- filtering with a probability of 0.7 of files with a mention of "test file" or "configuration file" or similar in the first 5 lines
- filtering with a probability of 0.7 of files with high occurence of the keywords "test " or "config" 
- filtering with a probability of 0.7  of files without a mention of the keywords `def` , `for`, `while`  and `class`
- filtering files that use the assignment operator `=` less than 5 times 
- filtering files with ratio between number of characters and number of tokens after tokenization < 1.5 (the average ratio is 3.6)

The script to process the full dataset can be found in `scripts/preprocessing.py`. Executing the script on 16 vCPUs takes roughly 3h and removes 70% of the original dataset. The cleaned [train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train-v2) and [validation](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid-v2) splits are also available on the Hub if you want to skip this step or use the data for another project.

To execute the preprocessing run the following command:
```bash
python scripts/preprocessing.py \
--dataset_name transformersbook/codeparrot \
--output_dir codeparrot-clean
```
During preprocessing the dataset is downloaded and stored locally as well as caches of the computations. Make sure you have more than 500GB free disk space to execute it.

### Pretokenization
The tokenization of the data might be slow during the training especially for small models. We provide code to pretokenize the data beforehand in `scripts/pretokenizing.py`, but this step is optional. The dataset is downloaded and stored locally and the tokenized data is pushed to the hub. The tokenized clean [train](https://huggingface.co/datasets/codeparrot/tokenized-codeparrot-train) and [validation](https://huggingface.co/datasets/codeparrot/tokenized-codeparrot-valid) datasets are available if you want to use them directly.

To execute the pretokenization, for the clean train data for instance, run the following command:
```bash
python scripts/pretokenizing.py \
--dataset_name codeparrot/codeparrot-clean-train \
--tokenized_data_repo tokenized-codeparrot-train
```

## Tokenizer
Before training a new model for code we create a new tokenizer that is efficient at code tokenization. To train the tokenizer you can run the following command: 
```bash
python scripts/bpe_training.py \
    --base_tokenizer gpt2 \
    --dataset_name codeparrot/codeparrot-clean-train
```

_Note:_ We originally trained the tokenizer on the unprocessed train split of the dataset `transformersbook/codeparrot-train`.

## Training
The models are randomly initialized and trained from scratch. To initialize a new model you can run:

```bash
python scripts/initialize_model.py \
--config_name gpt2-large \
--tokenizer_name codeparrot/codeparrot \
--model_name codeparrot \
--push_to_hub True
```
This will initialize a new model with the architecture and configuration of `gpt2-large` and use the tokenizer to appropriately size the input embeddings. Finally, the initilaized model is pushed the hub.

We can either pass the name of a text dataset or a pretokenized dataset which speeds up training a bit.
Now that the tokenizer and model are also ready we can start training the model. The main training script is built with `accelerate` to scale across a wide range of platforms and infrastructure scales. We train two models with [110M](https://huggingface.co/codeparrot/codeparrot-small/) and [1.5B](https://huggingface.co/codeparrot/codeparrot/) parameters for 25-30B tokens on a 16xA100 (40GB) machine which takes 1 day and 1 week, respectively.

First you need to configure `accelerate` and login to Weights & Biases:

```bash
accelerate config
wandb login
```

Note that during the `accelerate` configuration we enabled FP16. Then to train the large model you can run

```bash
accelerate launch scripts/codeparrot_training.py
```

If you want to train the small model you need to make some modifications:

```bash
accelerate launch scripts/codeparrot_training.py \
--model_ckpt codeparrot/codeparrot-small \
--train_batch_size 12 \
--valid_batch_size 12 \
--learning_rate 5e-4 \
--num_warmup_steps 2000 \
--gradient_accumulation 1 \
--gradient_checkpointing False \
--max_train_steps 150000 \
--save_checkpoint_steps 15000
```

Recall that you can see the full set of possible options with descriptions (for all scripts) by running:

```bash
python scripts/codeparrot_training.py --help
```

Instead of streaming the dataset from the hub you can also stream it from disk. This can be helpful for long training runs where the connection can be interrupted sometimes. To stream locally you simply need to clone the datasets and replace the dataset name with their path. In this example we store the data in a folder called `data`: 

```bash
git lfs install
mkdir data
git -C "./data" clone https://huggingface.co/datasets/codeparrot/codeparrot-clean-train
git -C "./data" clone https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid
```

And then pass the paths to the datasets when we run the training script:

```bash
accelerate launch scripts/codeparrot_training.py \
--model_ckpt codeparrot/codeparrot-small \
--dataset_name_train ./data/codeparrot-clean-train \
--dataset_name_valid ./data/codeparrot-clean-valid \
--train_batch_size 12 \
--valid_batch_size 12 \
--learning_rate 5e-4 \
--num_warmup_steps 2000 \
--gradient_accumulation 1 \
--gradient_checkpointing False \
--max_train_steps 150000 \
--save_checkpoint_steps 15000
```

## Evaluation
For evaluating the language modeling loss on the validation set or any other dataset you can use the following command:
```bash
python scripts/validation_loss.py \
--model_ckpt codeparrot/codeparrot \
--dataset_name codeparrot/codeparrot-clean-valid
```
In addition we evaluate the model on OpenAI's _HumanEval_ benchmark. You can run the evaluation with the following command:

```bash
accelerate launch  scripts/human_eval.py --model_ckpt codeparrot/codeparrot \
--do_sample True \
--temperature 0.2 \
--top_p 0.95 \
--n_samples=200 \
--HF_ALLOW_CODE_EVAL="0"
```

The results as well as reference values are shown in the following table:

| Model | pass@1 | pass@10 | pass@100|
|-------|--------|---------|---------|
|CodeParrot 🦜 (110M) | 3.80% | 6.57% | 12.78% |
|CodeParrot 🦜 (1.5B) | 3.99% | 8.69% | 17.88% |
|||||
|Codex (25M)| 3.21% | 7.1% |	12.89%|
|Codex (85M)| 8.22%	| 12.81% | 22.40% |
|Codex (300M)| 13.17%| 20.37% | 36.27% |
|Codex (12B)| 28.81%| 46.81% | 72.31% |
|||||
|GPT-neo (125M)| 0.75% | 1.88% | 2.97% |
|GPT-neo (1.5B)| 4.79% | 7.47% | 16.30% |
|GPT-neo (2.7B)| 6.41% | 11.27% | 21.37% |
|GPT-J (6B)| 11.62% | 15.74% | 27.74% |

The numbers were obtained by sampling with `T = [0.2, 0.6, 0.8]` and picking the best value for each metric. Both CodeParrot 🦜 models are still underfitted and longer training would likely improve the performance.

## Demo
Give the model a shot yourself! There are three demos to interact with CodeParrot 🦜:
- [Code generation](https://huggingface.co/spaces/codeparrot/codeparrot-generation)
- [Code highlighting](https://huggingface.co/spaces/codeparrot/codeparrot-highlighting)
- [Comparison to other code models](https://huggingface.co/spaces/codeparrot/loubnabnl/code-generation-models)

## Training with Megatron
[Megatron](https://github.com/NVIDIA/Megatron-LM) is a framework developed by NVIDIA for training large transformer models. While the CodeParrot code is easy to follow and modify to your needs the Megatron framework lets you train models faster. Below we explain how to use it.

### Setup
You can pull an NVIDIA PyTorch Container that comes with all the required installations from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). See [documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) for more details:

With the following Docker command you can run the container (`xx.xx` denotes your Docker version), and clone [Megatron repository](https://github.com/NVIDIA/Megatron-LM) into it:
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:xx.xx-py3
git clone https://github.com/NVIDIA/Megatron-LM
```

You also need to add the vocabulary file and merges table of the tokenizer that you trained on code into the container. You can also find these files in [vocab.json](https://huggingface.co/codeparrot/codeparrot/raw/main/vocab.json) and [merges.txt](https://huggingface.co/codeparrot/codeparrot/raw/main/merges.txt).
```bash
sudo docker cp vocab.json CONTAINER_ID:/workspace/Megatron-LM
sudo docker cp merges.txt CONTAINER_ID:/workspace/Megatron-LM
```

### Data preprocessing
The training data requires preprocessing. First, you need to convert it into a loose json format, with one json containing a text sample per line. In python this can be done this way:
```python
from datasets import load_dataset

train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')
train_data.to_json("codeparrot_data.json", lines=True)  
```

The data is then tokenized, shuffled and processed into a binary format for training using the following command:
```bash
pip install nltk
cd Megatron-LM
python tools/preprocess_data.py \
       --input codeparrot_data.json \
       --output-prefix codeparrot \
       --vocab vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys content \
       --workers 32 \
       --chunk-size 25 \
       --append-eod
```
This outputs two files `codeparrot_content_document.idx` and `codeparrot_content_document.bin` which are used in the training.

### Training
You can configure the model architecture and training parameters as shown below, or put it in a bash script that you will run. This runs on 8 GPUs the 110M parameter CodeParrot pretraining, with the same settings as before. Note that the data is partitioned by default into a 969:30:1 ratio for training/validation/test sets.
```bash
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/workspace/Megatron-LM/experiments/codeparrot-small
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=codeparrot_content_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS
```
The training takes almost 12 hours in this setting.

### Convert model to `transformers`
After training we want to use the model in `transformers` e.g. to evaluate it on HumanEval. You can convert it to `transformers` following [this](https://huggingface.co/nvidia/megatron-gpt2-345m) tutorial. For instance, after the training is finished you can copy the weights of the last iteration 150k and convert the `model_optim_rng.pt` file to a `pytorch_model.bin` file that is supported by `transformers`.

```bash
mkdir -p nvidia/megatron-codeparrot-small
sudo docker cp CONTAINER_ID:/workspace/Megatron-LM/experiments/codeparrot-small/iter_0150000/mp_rank_00/model_optim_rng.pt nvidia/megatron-codeparrot-small
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/NVIDIA/Megatron-LM.git
export PYTHONPATH=Megatron-LM
python transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py nvidia/megatron-codeparrot-small/model_optim_rng.pt
```
Be careful, you will need to replace the generated vocabulary file and merges table after the conversion, with the original ones if you plan to load the tokenizer from there.

## Further Resources
A detailed description of the project can be found in the chapter "Training Transformers from Scratch" in the upcoming O'Reilly book [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/).

This example was provided by [Leandro von Werra](www.github.com/lvwerra).
