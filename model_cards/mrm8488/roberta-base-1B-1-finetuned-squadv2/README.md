---
language: en
---

# RoBERTa-base (1B-1) + SQuAD v2 ❓

[roberta-base-1B-1](https://huggingface.co/nyu-mll/roberta-base-1B-1) fine-tuned on [SQUAD v2 dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Model 🧠

RoBERTa Pretrained on Smaller Datasets

[NYU Machine Learning for Language](https://huggingface.co/nyu-mll) pretrained RoBERTa on smaller datasets (1M, 10M, 100M, 1B tokens). They released 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens). The pretraining data reproduces that of BERT: They combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1.


## Details of the downstream task (Q&A) - Dataset 📚

**S**tanford **Q**uestion **A**nswering **D**ataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

 **SQuAD2.0** combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python transformers/examples/question-answering/run_squad.py \
  --model_type roberta \
  --model_name_or_path 'nyu-mll/roberta-base-1B-1' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file /content/dataset/train-v2.0.json \
  --predict_file /content/dataset/dev-v2.0.json \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/output \
  --overwrite_output_dir \
  --save_steps 1000 \
  --version_2_with_negative
```

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **64.86** |
| **F1** | **68.99** |



```json
{
'exact': 64.86145034953255, 
'f1': 68.9902640378272,
'total': 11873,
'HasAns_exact': 64.03508771929825,
'HasAns_f1': 72.3045554860189,
'HasAns_total': 5928,
'NoAns_exact': 65.68544995794785,
'NoAns_f1': 65.68544995794785,
'NoAns_total': 5945,
'best_exact': 64.86987282068559,
'best_exact_thresh': 0.0,
'best_f1': 68.99868650898054,
'best_f1_thresh': 0.0
}
```

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline

QnA_pipeline = pipeline('question-answering', model='mrm8488/roberta-base-1B-1-finetuned-squadv2')

QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'What has been discovered by scientists from China ?'
})
# Output:

{'answer': 'A new strain of flu', 'end': 19, 'score': 0.7145650685380576,'start': 0}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)
> Made with <span style="color: #e25555;">&hearts;</span> in Spain
