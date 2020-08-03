---
datasets:
- squad_v2
---

# MiniLM-L12-H384-uncased for QA

## Overview
**Language model:** microsoft/MiniLM-L12-H384-uncased
**Language:** English
**Downstream-task:** Extractive QA
**Training data:** SQuAD 2.0
**Eval data:** SQuAD 2.0
**Code:**  See [example](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py) in [FARM](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering.py)
**Infrastructure**: 1x Tesla v100

## Hyperparameters

```
seed=42
batch_size = 12
n_epochs = 4
base_LM_model = "microsoft/MiniLM-L12-H384-uncased"
max_seq_len = 384
learning_rate = 4e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
grad_acc_steps=4
```

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).
```
"exact": 76.13071675229513,
"f1": 79.49786500219953,
"total": 11873,
"HasAns_exact": 78.35695006747639,
"HasAns_f1": 85.10090269418276,
"HasAns_total": 5928,
"NoAns_exact": 73.91084945332211,
"NoAns_f1": 73.91084945332211,
"NoAns_total": 5945
```

## Usage

### In Transformers
```python
from transformers.pipelines import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

model_name = "deepset/minilm-uncased-squad2"

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
from farm.infer import Inferencer

model_name = "deepset/minilm-uncased-squad2"

# a) Get predictions
nlp = Inferencer.load(model_name, task_type="question_answering")
QA_input = [{"questions": ["Why is model conversion important?"],
             "text": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."}]
res = nlp.inference_from_dicts(dicts=QA_input)

# b) Load model & tokenizer
model = AdaptiveModel.convert_from_transformers(model_name, device="cpu", task_type="question_answering")
tokenizer = Tokenizer.load(model_name)
```

### In haystack
For doing QA at scale (i.e. many docs instead of single paragraph), you can load the model also in [haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2")
# or
reader = TransformersReader(model="deepset/minilm-uncased-squad2",tokenizer="deepset/minilm-uncased-squad2")
```


## Authors
Vaishali Pal `vaishali.pal [at] deepset.ai`
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
