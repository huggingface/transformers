---
language: en
datasets:
- squad_v2
---

# MobileBERT + SQuAD v2 📱❓

[mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) fine-tuned on [SQUAD v2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) for **Q&A** downstream task.

## Details of the downstream task (Q&A) - Model 🧠

**MobileBERT** is a thin version of *BERT_LARGE*, while equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks.

The checkpoint used here is the original MobileBert Optimized Uncased English: (uncased_L-24_H-128_B-512_A-4_F-4_OPT) checkpoint.

More about the model [here](https://arxiv.org/abs/2004.02984)

## Details of the downstream task (Q&A) - Dataset 📚

**SQuAD2.0** combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python transformers/examples/question-answering/run_squad.py \
  --model_type bert \
  --model_name_or_path 'google/mobilebert-uncased' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file '/content/dataset/train-v2.0.json' \
  --predict_file '/content/dataset/dev-v2.0.json' \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir '/content/output' \
  --overwrite_output_dir \
  --save_steps 1000 \
  --version_2_with_negative
```

It is importatnt to say that this models converges much faster than other ones. So, it is also cheap to fine-tune.

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **75.37** |
| **F1** | **78.48** |
| **Size**| **94 MB** |

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline
QnA_pipeline = pipeline('question-answering', model='mrm8488/mobilebert-uncased-finetuned-squadv2')
QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'Who did identified it ?'
    })
    
# Output: {'answer': 'scientists.', 'end': 106, 'score': 0.41531604528427124, 'start': 96}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
