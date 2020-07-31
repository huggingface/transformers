---
language: multilingual
tags:
- question-answering
datasets:
- squad_v2
---

# Multilingual XLM-RoBERTa large for QA on various languages 

## Overview
**Language model:** xlm-roberta-large  
**Language:** Multilingual  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD dev set - German MLQA - German XQuAD   
**Training run:** [MLFlow link](https://public-mlflow.deepset.ai/#/experiments/124/runs/3a540e3f3ecf4dd98eae8fc6d457ff20)  
**Infrastructure**: 4x Tesla v100

## Hyperparameters

```
batch_size = 32
n_epochs = 3
base_LM_model = "xlm-roberta-large"
max_seq_len = 256
learning_rate = 1e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
``` 

## Performance
Evaluated on the SQuAD 2.0 English dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
  "exact": 79.45759285774446,
  "f1": 83.79259828925511,
  "total": 11873,
  "HasAns_exact": 71.96356275303644,
  "HasAns_f1": 80.6460053117963,
  "HasAns_total": 5928,
  "NoAns_exact": 86.93019343986543,
  "NoAns_f1": 86.93019343986543,
  "NoAns_total": 5945
```

Evaluated on German [MLQA: test-context-de-question-de.json](https://github.com/facebookresearch/MLQA)
```
"exact": 49.34691166703564,
"f1": 66.15582561674236,
"total": 4517,
```

Evaluated on German [XQuAD: xquad.de.json](https://github.com/deepmind/xquad)
```
"exact": 61.51260504201681,
"f1": 78.80206098332569,
"total": 1190,
```

## Usage

### In Transformers
```python
from transformers.pipelines import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

model_name = "deepset/xlm-roberta-large-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### In FARM

```python
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.infer import QAInferencer

model_name = "deepset/xlm-roberta-large-squad2"

# a) Get predictions
nlp = QAInferencer.load(model_name)
QA_input = [{"questions": ["Why is model conversion important?"],
             "text": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."}]
res = nlp.inference_from_dicts(dicts=QA_input, rest_api_schema=True)

# b) Load model & tokenizer
model = AdaptiveModel.convert_from_transformers(model_name, device="cpu", task_type="question_answering")
tokenizer = Tokenizer.load(model_name)
```

### In haystack
For doing QA at scale (i.e. many docs instead of single paragraph), you can load the model also in [haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/xlm-roberta-large-squad2")
# or 
reader = TransformersReader(model="deepset/xlm-roberta-large-squad2",tokenizer="deepset/xlm-roberta-large-squad2")
```


## Authors
Branden Chan: `branden.chan [at] deepset.ai`  
Timo Möller: `timo.moeller [at] deepset.ai`  
Malte Pietsch: `malte.pietsch [at] deepset.ai`  
Tanay Soni: `tanay.soni [at] deepset.ai`

## About us
![deepset logo](https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/deepset_logo.png)

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our work: 
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [FARM](https://github.com/deepset-ai/FARM)
- [Haystack](https://github.com/deepset-ai/haystack/)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Website](https://deepset.ai)

