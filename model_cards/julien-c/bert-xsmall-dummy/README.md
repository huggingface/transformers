## How to build a dummy model


```python
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertForMaskedLM
from transformers.modeling_tf_bert import TFBertForMaskedLM
from transformers.tokenization_bert import BertTokenizer


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DIRNAME = "./bert-xsmall-dummy"

config = BertConfig(10, 20, 1, 1, 40)

model = BertForMaskedLM(config)
model.save_pretrained(DIRNAME)

tf_model = TFBertForMaskedLM.from_pretrained(DIRNAME, from_pt=True)
tf_model.save_pretrained(DIRNAME)

# Slightly different for tokenizer.
# tokenizer = BertTokenizer.from_pretrained(DIRNAME)
# tokenizer.save_pretrained()
```
