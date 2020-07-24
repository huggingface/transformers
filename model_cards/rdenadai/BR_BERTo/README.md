---
language: pt
tags:
- portuguese
- brazil
- pt_BR
widget:
- text: gostei muito dessa <mask>
---

# BR_BERTo

Portuguese (Brazil) model for text inference.

## Params

Trained on a corpus of 5_258_624 sentences, with 132_807_374 non unique tokens (992_418 unique tokens).

But since my machine doesn`t support bigger model, at the end it has a vocab size of 54_000 tokens. The rest of the parameters are the default used in the HuggingFace tutorial.

[How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)

## Results

```python
fill_mask("gostei muito dessa <mask>")

#[{'sequence': '<s>gostei muito dessa experiência</s>',
#  'score': 0.0719294399023056,
#  'token': 2322,
#  'token_str': 'ĠexperiÃªncia'},
# {'sequence': '<s>gostei muito dessa diferença</s>',
#  'score': 0.05286405608057976,
#  'token': 3472,
#  'token_str': 'ĠdiferenÃ§a'},
# {'sequence': '<s>gostei muito dessa atenção</s>',
#  'score': 0.027575725689530373,
#  'token': 2557,
#  'token_str': 'ĠatenÃ§Ã£o'},
# {'sequence': '<s>gostei muito dessa história</s>',
#  'score': 0.026764703914523125,
#  'token': 1329,
#  'token_str': 'ĠhistÃ³ria'},
# {'sequence': '<s>gostei muito dessa razão</s>',
#  'score': 0.0250675268471241,
#  'token': 3323,
#  'token_str': 'ĠrazÃ£o'},
# {'sequence': '<s>gostei muito dessa resposta</s>',
#  'score': 0.024784332141280174,
#  'token': 2403,
#  'token_str': 'Ġresposta'},
# {'sequence': '<s>gostei muito dessa dose</s>',
#  'score': 0.01720510423183441,
#  'token': 1042,
#  'token_str': 'Ġdose'}]
```
