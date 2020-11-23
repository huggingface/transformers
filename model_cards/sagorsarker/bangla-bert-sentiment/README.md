---
language:
- bn
datasets:
- socian
- bangla-sentiment-benchmark
license: mit
tags:
- bengali
- bengali-sentiment
- sentiment-analysis
---

# bangla-bert-sentiment
`bangla-bert-sentiment` is a pretrained model for bengali **Sentiment Analysis** using [bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) model.

## Datasets Details
This model was trained with two combined datasets
* [socian sentiment data](https://github.com/socian-ai/socian-bangla-sentiment-dataset-labeled)
* [bangla classification dataset](https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP)

|||
|--|--|
|Data Size| 10889 |
|Positive| 4999 |
|Negative| 5890 |
|Train | 8711 |
| Test | 2178 |

## Training Details
Model trained with [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) binary classification script with total of **3 epochs** in `google colab gpu`.


## Evaluation Details
Model evaluate with 2178 sentences

Here is the evaluation result details in table


|Eval Loss | TP | TN | FP | FN | F1 Score |
| -------- | -- | -- | -- | -- | -------- |
| 0.3289 | 880 | 1158 | 59 | 81 | 92.63 |

## Usage

Calculate sentiment from given sentence

```py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-sentiment")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
sentence = "বাংলার ঘরে ঘরে আজ নবান্নের উৎসব"
nlp(sentence)

```

