---
language: code
thumbnail:
---

# CodeBERTaJS

CodeBERTaJS is a RoBERTa-like model trained on the [CodeSearchNet](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/) dataset from GitHub for `javaScript` by [Manuel Romero](https://twitter.com/mrm8488)

The **tokenizer** is a Byte-level BPE tokenizer trained on the corpus using Hugging Face `tokenizers`.

Because it is trained on a corpus of code (vs. natural language), it encodes the corpus efficiently (the sequences are between 33% to 50% shorter, compared to the same corpus tokenized by gpt2/roberta).

The (small) **model** is a 6-layer, 84M parameters, RoBERTa-like Transformer model – that’s the same number of layers & heads as DistilBERT – initialized from the default initialization settings and trained from scratch on the full `javascript` corpus (120M after preproccessing) for 2 epochs.

## Quick start: masked language modeling prediction

```python
JS_CODE = """
async function createUser(req, <mask>) {
  if (!validUser(req.body.user)) {
	  return res.status(400);
  }
  user = userService.createUser(req.body.user);
  return res.json(user);
}
""".lstrip()
```

### Does the model know how to complete simple JS/express like code?

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="mrm8488/codeBERTaJS",
    tokenizer="mrm8488/codeBERTaJS"
)

fill_mask(JS_CODE)

## Top 5 predictions:
#
'res' # prob  0.069489665329
'next'
'req'
'user'
',req'
```

### Yes! That was easy 🎉 Let's try with another example

```python
JS_CODE_= """
function getKeys(obj) {
  keys = [];
  for (var [key, value] of Object.entries(obj)) {
     keys.push(<mask>);
  }
  return keys
}
""".lstrip()
```

Results:

```python
'obj', 'key', ' value', 'keys', 'i'
```

> Not so bad! Right token was predicted as second option! 🎉

## This work is heavely inspired on [codeBERTa](https://github.com/huggingface/transformers/blob/master/model_cards/huggingface/CodeBERTa-small-v1/README.md) by huggingface team

<br>

## CodeSearchNet citation

<details>

```bibtex
@article{husain_codesearchnet_2019,
	title = {{CodeSearchNet} {Challenge}: {Evaluating} the {State} of {Semantic} {Code} {Search}},
	shorttitle = {{CodeSearchNet} {Challenge}},
	url = {http://arxiv.org/abs/1909.09436},
	urldate = {2020-03-12},
	journal = {arXiv:1909.09436 [cs, stat]},
	author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
	month = sep,
	year = {2019},
	note = {arXiv: 1909.09436},
}
```

</details>

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
