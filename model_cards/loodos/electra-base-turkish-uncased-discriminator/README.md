---
language: tr
---

# Turkish Language Models with Huggingface's Transformers

As R&D Team at Loodos, we release cased and uncased versions of most recent language models for Turkish. More details about pretrained models and evaluations on downstream tasks can be found [here (our repo)](https://github.com/Loodos/turkish-language-models).

# Turkish ELECTRA-Base-discriminator (uncased)

This is ELECTRA-Base model's discriminator which has the same structure with BERT-Base trained on uncased Turkish dataset.

## Usage

Using AutoModelWithLMHead and AutoTokenizer from Transformers, you can import the model as described below.

```python
from transformers import AutoModel, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("loodos/electra-base-turkish-uncased-discriminator", do_lower_case=False)

model = AutoModelWithLMHead.from_pretrained("loodos/electra-base-turkish-uncased-discriminator")
 
normalizer = TextNormalization()
normalized_text = normalizer.normalize(text, do_lower_case=True, is_turkish=True)

tokenizer.tokenize(normalized_text)
```

### Notes on Tokenizers
Currently, Huggingface's tokenizers (which were written in Python) have a bug concerning letters "ı, i, I, İ" and non-ASCII Turkish specific letters. There are two reasons.

1- Vocabulary and sentence piece model is created with NFC/NFKC normalization but tokenizer uses NFD/NFKD. NFD/NFKD normalization changes text that contains Turkish characters I-ı, İ-i, Ç-ç, Ö-ö, Ş-ş, Ğ-ğ, Ü-ü. This causes wrong tokenization, wrong training and loss of information. Some tokens are never trained.(like "şanlıurfa", "öğün", "çocuk" etc.) NFD/NFKD normalization is not proper for Turkish.

2- Python's default ```string.lower()``` and ```string.upper()``` make the conversions

- "I" and "İ" to 'i'
- 'i' and 'ı' to 'I'

respectively. However, in Turkish, 'I' and 'İ' are two different letters. 

We opened an [issue](https://github.com/huggingface/transformers/issues/6680) in Huggingface's github repo about this bug. Until it is fixed, in case you want to train your model with uncased data, we provide a simple text normalization module (`TextNormalization()` in the code snippet above) in our [repo](https://github.com/Loodos/turkish-language-models).


## Details and Contact

You contact us to ask a question, open an issue or give feedback via our github [repo](https://github.com/Loodos/turkish-language-models).

## Acknowledgments

Many thanks to TFRC Team for providing us cloud TPUs on Tensorflow Research Cloud to train our models.

