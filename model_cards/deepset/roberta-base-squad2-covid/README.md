# roberta-base-squad2 for QA on COVID-19

## Overview
**Language model:** deepset/roberta-base-squad2  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** [SQuAD-style CORD-19 annotations from 23rd April](https://github.com/deepset-ai/COVID-QA/blob/master/data/question-answering/200423_covidQA.json)  
**Code:**  See [example](https://github.com/deepset-ai/FARM/blob/master/examples/question_answering_crossvalidation.py) in [FARM](https://github.com/deepset-ai/FARM)  
**Infrastructure**: Tesla v100

## Hyperparameters
```
batch_size = 24
n_epochs = 3
base_LM_model = "deepset/roberta-base-squad2"
max_seq_len = 384
learning_rate = 3e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.1
doc_stride = 128
xval_folds = 5
dev_split = 0
no_ans_boost = -100
```

## Performance
5-fold cross-validation on the data set led to the following results:  

**Single EM-Scores:**   [0.222, 0.123, 0.234, 0.159, 0.158]  
**Single F1-Scores:**   [0.476, 0.493, 0.599, 0.461, 0.465]  
**Single top\_3\_recall Scores:**   [0.827, 0.776, 0.860, 0.771, 0.777]  
**XVAL EM:**   0.17890995260663506  
**XVAL f1:**   0.49925444207319924  
**XVAL top\_3\_recall:**   0.8021327014218009

This model is the model obtained from the **third** fold of the cross-validation.

## Usage

### In Transformers
```python
from transformers.pipelines import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

model_name = "deepset/roberta-base-squad2-covid"

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

model_name = "deepset/roberta-base-squad2-covid"

# a) Get predictions
nlp = Inferencer.load(model_name, task_type="question_answering")
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
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-covid")
# or 
reader = TransformersReader(model="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2-covid")
```

## Authors
Branden Chan: `branden.chan [at] deepset.ai`  
Timo Möller: `timo.moeller [at] deepset.ai`  
Malte Pietsch: `malte.pietsch [at] deepset.ai`  
Tanay Soni: `tanay.soni [at] deepset.ai`  
Bogdan Kostić: `bogdan.kostic [at] deepset.ai`  

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