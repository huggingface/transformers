---
language: code
thumbnail: https://hf-dinosaur.huggingface.co/CodeBERTa/CodeBERTa.png
---

# CodeBERTa

CodeBERTa is a RoBERTa-like model trained on the [CodeSearchNet](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/) dataset from GitHub.

Supported languages:

```shell
"go"
"java"
"javascript"
"php"
"python"
"ruby"
```

The **tokenizer** is a Byte-level BPE tokenizer trained on the corpus using Hugging Face `tokenizers`.

Because it is trained on a corpus of code (vs. natural language), it encodes the corpus efficiently (the sequences are between 33% to 50% shorter, compared to the same corpus tokenized by gpt2/roberta).

The (small) **model** is a 6-layer, 84M parameters, RoBERTa-like Transformer model – that’s the same number of layers & heads as DistilBERT – initialized from the default initialization settings and trained from scratch on the full corpus (~2M functions) for 5 epochs.

### Tensorboard for this training ⤵️

[![tb](https://hf-dinosaur.huggingface.co/CodeBERTa/tensorboard.png)](https://tensorboard.dev/experiment/irRI7jXGQlqmlxXS0I07ew/#scalars)

## Quick start: masked language modeling prediction

```python
PHP_CODE = """
public static <mask> set(string $key, $value) {
	if (!in_array($key, self::$allowedKeys)) {
		throw new \InvalidArgumentException('Invalid key given');
	}
	self::$storedValues[$key] = $value;
}
""".lstrip()
```

### Does the model know how to complete simple PHP code?

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="huggingface/CodeBERTa-small-v1",
    tokenizer="huggingface/CodeBERTa-small-v1"
)

fill_mask(PHP_CODE)

## Top 5 predictions:
# 
' function' # prob 0.9999827146530151
'function'  # 
' void'     # 
' def'      # 
' final'    # 
```

### Yes! That was easy 🎉 What about some Python (warning: this is going to be meta)

```python
PYTHON_CODE = """
def pipeline(
    task: str,
    model: Optional = None,
    framework: Optional[<mask>] = None,
    **kwargs
) -> Pipeline:
	pass
""".lstrip()
```

Results:
```python
'framework', 'Framework', ' framework', 'None', 'str'
```

> This program can auto-complete itself! 😱

### Just for fun, let's try to mask natural language (not code):

```python
fill_mask("My name is <mask>.")

# {'sequence': '<s> My name is undefined.</s>', 'score': 0.2548016905784607, 'token': 3353}
# {'sequence': '<s> My name is required.</s>', 'score': 0.07290805131196976, 'token': 2371}
# {'sequence': '<s> My name is null.</s>', 'score': 0.06323737651109695, 'token': 469}
# {'sequence': '<s> My name is name.</s>', 'score': 0.021919190883636475, 'token': 652}
# {'sequence': '<s> My name is disabled.</s>', 'score': 0.019681859761476517, 'token': 7434}
```

This (kind of) works because code contains comments (which contain natural language).

Of course, the most frequent name for a Computer scientist must be undefined 🤓.


## Downstream task: [programming language identification](https://huggingface.co/huggingface/CodeBERTa-language-id)

See the model card for **[`huggingface/CodeBERTa-language-id`](https://huggingface.co/huggingface/CodeBERTa-language-id)** 🤯.

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
