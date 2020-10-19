---
language: pl
tags:
- herbert
license: cc-by-sa-4.0
---
# HerBERT 
**[HerBERT](https://en.wikipedia.org/wiki/Zbigniew_Herbert)** is a BERT-based Language Model trained on Polish Corpora
using MLM and SSO objectives with dynamic masking of whole words.
Model training and experiments were conducted with [transformers](https://github.com/huggingface/transformers) in version 2.9.

## Tokenizer
The training dataset was tokenized into subwords using ``CharBPETokenizer`` a character level byte-pair encoding with
a vocabulary size of 50k tokens. The tokenizer itself was trained with a [tokenizers](https://github.com/huggingface/tokenizers) library. 
We kindly encourage you to use the **Fast** version of tokenizer, namely ``HerbertTokenizerFast``.

## HerBERT usage


Example code:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
model = AutoModel.from_pretrained("allegro/herbert-large-cased")

output = model(
    **tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
    padding='longest',
    add_special_tokens=True,
    return_tensors='pt'
    )
)
```


## License
CC BY-SA 4.0


## Authors
Model was trained by **Allegro Machine Learning Research** team.

You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
