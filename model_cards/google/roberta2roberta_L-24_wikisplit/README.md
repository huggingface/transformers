---
language: en
license: apache-2.0
---

# Roberta2Roberta_L-24_wikisplit EncoderDecoder model

The model was introduced in 
[this paper](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn and first released in [this repository](https://tfhub.dev/google/bertseq2seq/roberta24_cnndm/1). 

The model is an encoder-decoder model that was initialized on the `roberta-large` checkpoints for both the encoder 
and decoder and fine-tuned on sentence splitting on the [WikiSplit](https://github.com/google-research-datasets/wiki-split) dataset.

Disclaimer: The model card has been written by the Hugging Face team.

## How to use

You can use this model for sentence splitting, *e.g.*

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_wikisplit")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_wikisplit")

long_sentence = """Due to the hurricane, Lobsterfest has been canceled, making Bob very happy about it and he decides to open Bob 's Burgers for customers who were planning on going to Lobsterfest."""

input_ids = tokenizer(long_sentence, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# Due Due hurricane, Lobsterfest has been canceled, making Bob very happy about it. He decides to open B
# ob's Burgers for customers who were planning on going to Lobsterfest.com.
```
