---
language: tr
---

# Turkish Language Models with Huggingface's Transformers

As R&D Team at Loodos, we release cased and uncased versions of most recent language models for Turkish. More details about pretrained models and evaluations on downstream tasks can be found [here](https://github.com/Loodos/turkish-language-models)

# Turkish ELECTRA-Small-discriminator (cased)

This is ELECTRA-Small model's discriminator which has 12 encoder layers with 256 hidden layer size trained on cased Turkish dataset.

## Usage

Using AutoModel and AutoTokenizer from Transformers, you can import the model as described below.

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("loodos/electra-small-turkish-cased-discriminator")
model = AutoModel.from_pretrained("loodos/electra-small-turkish-cased-discriminator")
```

# Details and Contact

You contact us to ask a question, open an issue or give feedback via our github [repo](https://github.com/Loodos/turkish-language-models).

# Acknowledgments
