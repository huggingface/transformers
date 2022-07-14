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

- exact deduplication using each file's hash
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

The script to process the full dataset can be found in `scripts/preprocessing.py`. Executing the script on 16 vCPUs takes roughly 3h and removes 70% of the original dataset. The cleaned [train](https://huggingface.co/datasets/loubnabnl/codeparrot-clean-train-v2) and [validation](https://huggingface.co/datasets/loubnabnl/codeparrot-clean-valid-v2) splits are also available on the Hub if you want to skip this step or use the data for another project.

To execute the preprocessing run the following command:
```bash
python scripts/preprocessing.py \
--dataset_name transformersbook/codeparrot \
--output_dir codeparrot-clean
```
During preprocessing the dataset is downloaded and stored locally as well as caches of the computations. Make sure you have more than 500GB free disk space to execute it.

### Pretokenization
The tokenization of the data might be slow during the training especially for small models. We provide code to pretokenize the data beforehand in `scripts/pretokenizing.py`, but this step is optional. The dataset is downloaded and stored locally and the tokenized data is pushed to the hub. The tokenized clean [train](https://huggingface.co/datasets/loubnabnl/tokenized-codeparrot-train) and [validation](https://huggingface.co/datasets/loubnabnl/tokenized-codeparrot-valid) datasets are available if you want to use them directly.

To execute the pretokenization, for the clean train data for instance, run the following command:
```bash
python scripts/pretokenizing.py \
--dataset_name lvwerra/codeparrot-clean-train \
--tokenized_data_repo tokenized-codeparrot-train
```

## Tokenizer
Before training a new model for code we create a new tokenizer that is efficient at code tokenization. To train the tokenizer you can run the following command: 
```bash
python scripts/bpe_training.py \
    --base_tokenizer gpt2 \
    --dataset_name lvwerra/codeparrot-clean-train
```

_Note:_ We originally trained the tokenizer on the unprocessed train split of the dataset `transformersbook/codeparrot-train`.

## Training
The models are randomly initialized and trained from scratch. To initialize a new model you can run:

```bash
python scripts/initialize_model.py \
--config_name gpt2-large \
--tokenizer_name lvwerra/codeparrot \
--model_name codeparrot \
--push_to_hub True
```
This will initialize a new model with the architecture and configuration of `gpt2-large` and use the tokenizer to appropriately size the input embeddings. Finally, the initilaized model is pushed the hub.

We can either pass the name of a text dataset or a pretokenized dataset which speeds up training a bit.
Now that the tokenizer and model are also ready we can start training the model. The main training script is built with `accelerate` to scale across a wide range of platforms and infrastructure scales. We train two models with [110M](https://huggingface.co/lvwerra/codeparrot-small/) and [1.5B](https://huggingface.co/lvwerra/codeparrot/) parameters for 25-30B tokens on a 16xA100 (40GB) machine which takes 1 day and 1 week, respectively.

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
--model_ckpt lvwerra/codeparrot-small \
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
git -C "./data" clone https://huggingface.co/datasets/lvwerra/codeparrot-clean-train
git -C "./data" clone https://huggingface.co/datasets/lvwerra/codeparrot-clean-valid
```

And then pass the paths to the datasets when we run the training script:

```bash
accelerate launch scripts/codeparrot_training.py \
--model_ckpt lvwerra/codeparrot-small \
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
--model_ckpt lvwerra/codeparrot \
--dataset_name lvwerra/codeparrot-clean-valid
```
In addition we evaluate the model on OpenAI's _HumanEval_ benchmark. You can run the evaluation with the following command:

```bash
accelerate launch  scripts/human_eval.py --model_ckpt lvwerra/codeparrot \
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
Give the model a shot yourself! There are two demos to interact with CodeParrot 🦜:
- [Code generation](https://huggingface.co/spaces/lvwerra/codeparrot-generation)
- [Code highlighting](https://huggingface.co/spaces/lvwerra/codeparrot-highlighting)

## Further Resources
A detailed description of the project can be found in the chapter "Training Transformers from Scratch" in the upcoming O'Reilly book [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/).

This example was provided by [Leandro von Werra](www.github.com/lvwerra).
