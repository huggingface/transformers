---
language: en
datasets:
- emo
---

## Emo-MobileBERT: a thin version of BERT LARGE, trained on the EmoContext Dataset from scratch

### Details of MobileBERT

The **MobileBERT** model was presented in [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) by *Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou* and here is the abstract:

Natural Language Processing (NLP) has recently achieved great success by using huge pre-trained models with hundreds of millions of parameters. However, these models suffer from heavy model sizes and high latency such that they cannot be deployed to resource-limited mobile devices. In this paper, we propose MobileBERT for compressing and accelerating the popular BERT model. Like the original BERT, MobileBERT is task-agnostic, that is, it can be generically applied to various downstream NLP tasks via simple fine-tuning. Basically, MobileBERT is a thin version of BERT_LARGE, while equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks. To train MobileBERT, we first train a specially designed teacher model, an inverted-bottleneck incorporated BERT_LARGE model. Then, we conduct knowledge transfer from this teacher to MobileBERT. Empirical studies show that MobileBERT is 4.3x smaller and 5.5x faster than BERT_BASE while achieving competitive results on well-known benchmarks. On the natural language inference tasks of GLUE, MobileBERT achieves a GLUEscore o 77.7 (0.6 lower than BERT_BASE), and 62 ms latency on a Pixel 4 phone. On the SQuAD v1.1/v2.0 question answering task, MobileBERT achieves a dev F1 score of 90.0/79.2 (1.5/2.1 higher than BERT_BASE).

### Details of the downstream task (Emotion Recognition) - Dataset 📚

SemEval-2019 Task 3: EmoContext Contextual Emotion Detection in Text

In this dataset, given a textual dialogue i.e. an utterance along with two previous turns of context, the goal was to infer the underlying emotion of the utterance by choosing from four emotion classes:

 - sad 😢
 - happy 😃
 - angry 😡
 - others

### Model training

The training script is present [here](https://github.com/lordtt13/transformers-experiments/blob/master/Custom%20Tasks/emo-mobilebert.ipynb).

### Pipelining the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")

model = AutoModelForSequenceClassification.from_pretrained("lordtt13/emo-mobilebert")

nlp_sentence_classif = transformers.pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)
nlp_sentence_classif("I've never had such a bad day in my life")
# Output: [{'label': 'sad', 'score': 0.93153977394104}]
```

> Created by [Tanmay Thakur](https://github.com/lordtt13) | [LinkedIn](https://www.linkedin.com/in/tanmay-thakur-6bb5a9154/)
