---
language: en
datasets:
- squad
---

# MobileBERT + SQuAD (v1.1) 📱❓

[mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) fine-tuned on [SQUAD v2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Model 🧠

**MobileBERT** is a thin version of *BERT_LARGE*, while equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks.

The checkpoint used here is the original MobileBert Optimized Uncased English: (uncased_L-24_H-128_B-512_A-4_F-4_OPT) checkpoint.

More about the model [here](https://arxiv.org/abs/2004.02984)

## Details of the downstream task (Q&A) - Dataset 📚

**S**tanford **Q**uestion **A**nswering **D**ataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD v1.1 contains **100,000+** question-answer pairs on **500+** articles.

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python transformers/examples/question-answering/run_squad.py \
  --model_type bert \
  --model_name_or_path 'google/mobilebert-uncased' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file '/content/dataset/train-v1.1.json' \
  --predict_file '/content/dataset/dev-v1.1.json' \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir '/content/output' \
  --overwrite_output_dir \
  --save_steps 1000
```

It is important to say that this models converges much faster than other ones. So, it is also cheap to fine-tune.

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **82.33** |
| **F1** | **89.64** |
| **Size**| **94 MB** |

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline
QnA_pipeline = pipeline('question-answering', model='mrm8488/mobilebert-uncased-finetuned-squadv1')
QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'Who did identified it ?'
    })
    
# Output: {'answer': 'scientists.', 'end': 106, 'score': 0.7885545492172241, 'start': 96}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
