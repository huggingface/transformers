# CodeParrot 🦜

## What is this about?

This is an open-source effort to train and evaluate code generation models. Codeparrot is a GPT-2 model trained from scratch on Python code. 

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The source of the dataset is the GitHub dump available on Google's [BigQuery](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code). The database was queried for all Python files resulting in a 180GB dataset with over 20M files. The dataset is available on the Hugging Face Hub [here](https://huggingface.co/datasets/transformersbook/codeparrot).

## Preprocessing
The raw dataset contains many duplications so the dataset was deduplicated and filtered using the heuristics proposed in the Codex [paper](https://arxiv.org/abs/2107.03374):

- exact deduplication
- filtering files with max line length > 1000
- filtering files with mean line length > 100
- fraction of alphanumeric characters < 0.25
- containing the word "auto-generated" or similar in the first 5 lines

The script to process the full dataset can be found in `scripts/preprocessing.py`. Executing the script on 16 CPUs takes roughly 3h and removes 70% of the original dataset. The cleaned [train](https://huggingface.co/datasets/lvwerra/codeparrot-clean-train) and [validation](https://huggingface.co/datasets/lvwerra/codeparrot-clean-valid) splits are also available on the Hub.

## Training
The models are randomly initialized and trained from scratch. The initialization script can be found at `scripts/initialize.py`. The main training script is built with 🤗 Accelerate to scale across a wide range of platforms and infrastructure scales.

We train two models with [110M](https://huggingface.co/lvwerra/codeparrot-small/) and [1.5B](https://huggingface.co/lvwerra/codeparrot/) parameters for 30B tokens on 16xA100 (40GB) machine which takes 1 day and 1 week, respectively. The training script can be found in `scripts/codeparrot_training.py`.

## Evaluation

We evaluate the model on OpenAI's _HumanEval_ benchmark. The evaluation script can be found in `scripts/human_eval.py`

| Model | pass@1 | pass@10 | pass@100|
|-------|--------|---------|---------|
|CodeParrot 🦜 (110M) | 0.0% | 0.0% | 0.0% |
|CodeParrot 🦜 (1.5B) | 0.0% | 0.0% | 0.0% |
|||||
|Codex (85M)| 0.0% | 0.0% | 0.0% |
|Codex (300M)| 0.0% | 0.0% | 0.0% |
|Codex (12B)| 0.0% | 0.0% | 0.0% |
|||||
|GPT-neo (125M)| 0.0% | 0.0% | 0.0% |
|GPT-neo (1.5B)| 0.0% | 0.0% | 0.0% |
|GPT-neo (2.7B)| 0.0% | 0.0% | 0.0% |
|GPT-J (6B)| 0.0% | 0.0% | 0.0% |


## Further Resources

A detailed description of the project can be found in the chapter "Training Transformers from Scratch" in the upcoming O'Reilly book [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/).