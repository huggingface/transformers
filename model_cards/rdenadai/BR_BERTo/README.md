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

Trained on a corpus of 6_993_330 sentences.

- Vocab size: 150_000
- RobertaForMaskedLM  size : 512
- Num train epochs: 3
- Time to train: ~10days (on GCP with a Nvidia T4)

I follow the great tutorial from HuggingFace team:

[How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)

More infor here:

[BR_BERTo](https://github.com/rdenadai/BR-BERTo)
