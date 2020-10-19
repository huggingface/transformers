---
language: en
license: apache-2.0
datasets:
- discofuse
---

# Roberta2Roberta_L-24_discofuse EncoderDecoder model

The model was introduced in 
[this paper](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn and first released in [this repository](https://tfhub.dev/google/bertseq2seq/roberta24_discofuse/1). 

The model is an encoder-decoder model that was initialized on the `roberta-large` checkpoints for both the encoder 
and decoder and fine-tuned on sentencefusion on the discofuse dataset, which is linked above.

Disclaimer: The model card has been written by the Hugging Face team.

## How to use

You can use this model for sentence fusion, *e.g.*

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_discofuse")

discofuse = """As a run-blocker, Zeitler moves relatively well. Zeitler often struggles at the point of contact in space."""

input_ids = tokenizer(discofuse, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# As a run-blocker, Zeitler moves relatively well. However, Zeitler often struggles at the point of contact in space.  
```
