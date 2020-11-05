---
language: 
- en
- de
license: apache-2.0
datasets:
- wmt14
tags:
- translation
---

# bert2bert_L-24_wmt_en_de EncoderDecoder model

The model was introduced in 
[this paper](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn and first released in [this repository](https://tfhub.dev/google/bertseq2seq/bert24_en_de/1). 

The model is an encoder-decoder model that was initialized on the `bert-large` checkpoints for both the encoder 
and decoder and fine-tuned on English to German translation on the WMT dataset, which is linked above.

Disclaimer: The model card has been written by the Hugging Face team.

## How to use

You can use this model for translation, *e.g.*

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")

sentence = "Would you like to grab a coffee with me this week?"

input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# Möchten Sie diese Woche einen Kaffee mit mir schnappen?
