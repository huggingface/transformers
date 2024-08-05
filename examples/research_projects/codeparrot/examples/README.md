# Examples
In this folder we showcase some examples to use code models for downstream tasks.

## Complexity prediction
In this task we want to predict the complexity of Java programs in [CodeComplex](https://huggingface.co/datasets/codeparrot/codecomplex) dataset. Using Hugging Face `trainer`, we finetuned [multilingual CodeParrot](https://huggingface.co/codeparrot/codeparrot-small-multi) and [UniXcoder](https://huggingface.co/microsoft/unixcoder-base-nine) on it, and we used the latter to build this Java complexity prediction [space](https://huggingface.co/spaces/codeparrot/code-complexity-predictor) on Hugging Face hub.

To fine-tune a model on this dataset you can use the following commands:

```python
python train_complexity_predictor.py \
    --model_ckpt microsoft/unixcoder-base-nine \
    --num_epochs 60 \
    --num_warmup_steps 10 \
    --batch_size 8 \
    --learning_rate 5e-4 
```

## Code generation: text to python
In this task we want to train a model to generate code from english text. We finetuned Codeparrot-small on [github-jupyter-text-to-code](https://huggingface.co/datasets/codeparrot/github-jupyter-text-to-code), a dataset where the samples are a succession of docstrings and their Python code, originally extracted from Jupyter notebooks parsed in this [dataset](https://huggingface.co/datasets/codeparrot/github-jupyter-parsed).

To fine-tune a model on this dataset we use the same [script](https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py) as the pretraining of codeparrot:

```python
accelerate launch scripts/codeparrot_training.py \
    --model_ckpt codeparrot/codeparrot-small \
    --dataset_name_train codeparrot/github-jupyter-text-to-code \
    --dataset_name_valid codeparrot/github-jupyter-text-to-code \
    --train_batch_size 12 \
    --valid_batch_size 12 \
    --learning_rate 5e-4 \
    --num_warmup_steps 100 \
    --gradient_accumulation 1 \
    --gradient_checkpointing False \
    --max_train_steps 3000 \
    --save_checkpoint_steps 200 \
    --save_dir jupyter-text-to-python
```

## Code explanation: python to text
In this task we want to train a model to explain python code. We finetuned Codeparrot-small on [github-jupyter-code-to-text](https://huggingface.co/datasets/codeparrot/github-jupyter-code-to-text), a dataset where the samples are a succession of Python code and its explanation as a docstring, we just inverted the order of text and code pairs in github-jupyter-code-to-text dataset and added the delimiters "Explanation:" and "End of explanation" inside the doctrings.

To fine-tune a model on this dataset we use the same [script](https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py) as the pretraining of codeparrot:

```python
accelerate launch scripts/codeparrot_training.py \
    --model_ckpt codeparrot/codeparrot-small \
    --dataset_name_train codeparrot/github-jupyter-code-to-text \
    --dataset_name_valid codeparrot/github-jupyter-code-to-text \
    --train_batch_size 12 \
    --valid_batch_size 12 \
    --learning_rate 5e-4 \
    --num_warmup_steps 100 \
    --gradient_accumulation 1 \
    --gradient_checkpointing False \
    --max_train_steps 3000 \
    --save_checkpoint_steps 200 \
    --save_dir jupyter-python-to-text
```