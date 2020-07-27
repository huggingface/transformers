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

- Vocab size: 220_000
- RobertaForMaskedLM  size : 32
- Num train epochs: 2
- Time to train: ~23hs (on GCP with a Nvidia T4)

I follow the great tutorial from HuggingFace team:

[How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)
