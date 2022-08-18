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