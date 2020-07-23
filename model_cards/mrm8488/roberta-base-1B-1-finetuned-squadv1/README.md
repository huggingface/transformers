---
language: en
---

# RoBERTa-base (1B-1) + SQuAD v1 ❓

[roberta-base-1B-1](https://huggingface.co/nyu-mll/roberta-base-1B-1) fine-tuned on [SQUAD v1.1 dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Model 🧠

RoBERTa Pretrained on Smaller Datasets

[NYU Machine Learning for Language](https://huggingface.co/nyu-mll) pretrained RoBERTa on smaller datasets (1M, 10M, 100M, 1B tokens). They released 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens). The pretraining data reproduces that of BERT: They combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1.


## Details of the downstream task (Q&A) - Dataset 📚

**S**tanford **Q**uestion **A**nswering **D**ataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD v1.1 contains **100,000+** question-answer pairs on **500+** articles.

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python transformers/examples/question-answering/run_squad.py \
  --model_type roberta \
  --model_name_or_path 'nyu-mll/roberta-base-1B-1' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file /content/dataset/train-v1.1.json \
  --predict_file /content/dataset/dev-v1.1.json \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/output \
  --overwrite_output_dir \
  --save_steps 1000
```

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **72.62** |
| **F1** | **82.19** |



```json
{
'exact': 72.62062440870388,
'f1': 82.19430877136834,
'total': 10570,
'HasAns_exact': 72.62062440870388,
'HasAns_f1': 82.19430877136834,
'HasAns_total': 10570,
'best_exact': 72.62062440870388,
'best_exact_thresh': 0.0,
'best_f1': 82.19430877136834,
'best_f1_thresh': 0.0
}

```

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline

QnA_pipeline = pipeline('question-answering', model='mrm8488/roberta-base-1B-1-finetuned-squadv1')

QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'What has been discovered by scientists from China ?'
})
# Output:

{'answer': 'A new strain of flu', 'end': 19, 'score': 0.04702283976040074, 'start': 0}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)
> Made with <span style="color: #e25555;">&hearts;</span> in Spain
