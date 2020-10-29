---
language: en
datasets:
- squad_v2
---

# SqueezeBERT + SQuAD v2

[squeezebert-uncased](https://huggingface.co/squeezebert/squeezebert-uncased) fine-tuned on [SQUAD v2](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) for **Q&A** downstream task.

## Details of SqueezeBERT

This model, `squeezebert-uncased`, is a pretrained model for the English language using a masked language modeling (MLM) and Sentence Order Prediction (SOP) objective.
SqueezeBERT was introduced in [this paper](https://arxiv.org/abs/2006.11316). This model is case-insensitive. The model architecture is similar to BERT-base, but with the pointwise fully-connected layers replaced with [grouped convolutions](https://blog.yani.io/filter-group-tutorial/).
The authors found that SqueezeBERT is 4.3x faster than `bert-base-uncased` on a Google Pixel 3 smartphone.
More about the model [here](https://arxiv.org/abs/2004.02984)

## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

**SQuAD2.0** combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
python /content/transformers/examples/question-answering/run_squad.py \
  --model_type bert \
  --model_name_or_path squeezebert/squeezebert-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file /content/dataset/train-v2.0.json \
  --predict_file /content/dataset/dev-v2.0.json \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 15 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/output_dir \
  --overwrite_output_dir \
  --version_2_with_negative \
  --save_steps 2000
```

## Test set Results 🧾

| Metric | # Value   |
| ------ | --------- |
| **EM** | **69.98** |
| **F1** | **74.14** |

Model Size: **195 MB** 

### Model in action 🚀

Fast usage with **pipelines**:

```python
from transformers import pipeline
QnA_pipeline = pipeline('question-answering', model='mrm8488/squeezebert-finetuned-squadv2')
QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'Who did identified it ?'
    })
    
# Output: {'answer': 'scientists.', 'end': 106, 'score': 0.9768241047859192, 'start': 96}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
