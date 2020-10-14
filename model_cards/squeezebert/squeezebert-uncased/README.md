language: en
license: bsd
datasets:
- bookcorpus
- wikipedia
---

# SqueezeBERT pretrained model

This model, `squeezebert-uncased`, is a pretrained model for the English language using a masked language modeling (MLM) and Sentence Order Prediction (SOP) objective.
SqueezeBERT was introduced in [this paper](https://arxiv.org/abs/2006.11316). This model is case-insensitive. The model architecture is similar to BERT-base, but with the pointwise fully-connected layers replaced with [grouped convolutions](https://blog.yani.io/filter-group-tutorial/).
The authors found that SqueezeBERT is 4.3x faster than `bert-base-uncased` on a Google Pixel 3 smartphone.


## Pretraining

### Pretraining data
- [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of thousands of unpublished books
- [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia)

### Pretraining procedure
The model is pretrained using the Masked Language Model (MLM) and Sentence Order Prediction (SOP) tasks.
(Author's note: If you decide to pretrain your own model, and you prefer to train with MLM only, that should work too.)

The SqueezeBERT paper presents 2 approaches to finetuning the model:
> We pretrain SqueezeBERT from scratch (without distillation) using the [LAMB](https://arxiv.org/abs/1904.00962) optimizer, and we employ the hyperparameters recommended by the LAMB authors: a global batch size of 8192, a learning rate of 2.5e-3, and a warmup proportion of 0.28. Following the LAMB paper's recommendations, we pretrain for 56k steps with a maximum sequence length of 128 and then for 6k steps with a maximum sequence length of 512.

## Finetuning

The SqueezeBERT paper results from 2 approaches to finetuning the model:
- "finetuning without bells and whistles" -- after pretraining the SqueezeBERT model, finetune it on each GLUE task
- "finetuning with bells and whistles" -- after pretraining the SqueezeBERT model, finetune it on a MNLI with distillation from a teacher model. Then, use the MNLI-finetuned SqueezeBERT model as a student model to finetune on each of the other GLUE tasks (e.g. RTE, MRPC, …) with distillation from a task-specific teacher model.

A detailed discussion of the hyperparameters used for finetuning is provided in the appendix of the [SqueezeBERT paper](https://arxiv.org/abs/2006.11316).
Note that finetuning SqueezeBERT with distillation is not yet implemented in this repo. If the author (Forrest Iandola - forrest.dnn@gmail.com) gets enough encouragement from the user community, he will add example code to Transformers for finetuning SqueezeBERT with distillation.

This model, `squeezebert/squeezebert-uncased`, has been pretrained but not finetuned. For most text classification tasks, we recommend using squeezebert-mnli-headless as a starting point.

### How to finetune
To try finetuning SqueezeBERT on the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) text classification task, you can run the following command:
```
./utils/download_glue_data.py

python examples/text-classification/run_glue.py \
    --model_name_or_path squeezebert-base-headless \
    --task_name mrpc \
    --data_dir ./glue_data/MRPC \
    --output_dir ./models/squeezebert_mrpc \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --learning_rate 3e-05 \
    --per_device_train_batch_size 16 \
    --save_steps 20000

```

## BibTeX entry and citation info
```
@article{2020_SqueezeBERT,
     author = {Forrest N. Iandola and Albert E. Shaw and Ravi Krishna and Kurt W. Keutzer},
     title = {{SqueezeBERT}: What can computer vision teach NLP about efficient neural networks?},
     journal = {arXiv:2006.11316},
     year = {2020}
}
```
