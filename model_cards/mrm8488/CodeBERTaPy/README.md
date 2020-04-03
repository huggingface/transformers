---
language: code
thumbnail:
---

# CodeBERTaPy

CodeBERTaPy is a RoBERTa-like model trained on the [CodeSearchNet](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/) dataset from GitHub for `python` by [Manuel Romero](https://twitter.com/mrm8488)

The **tokenizer** is a Byte-level BPE tokenizer trained on the corpus using Hugging Face `tokenizers`.

Because it is trained on a corpus of code (vs. natural language), it encodes the corpus efficiently (the sequences are between 33% to 50% shorter, compared to the same corpus tokenized by gpt2/roberta).

The (small) **model** is a 6-layer, 84M parameters, RoBERTa-like Transformer model – that’s the same number of layers & heads as DistilBERT – initialized from the default initialization settings and trained from scratch on the full `python` corpus for 4 epochs.

## Quick start: masked language modeling prediction

```python
PYTHON_CODE = """
fruits = ['apples', 'bananas', 'oranges']
for idx, <mask> in enumerate(fruits):
  print("index is %d and value is %s" % (idx, val))
""".lstrip()
```

### Does the model know how to complete simple Python code?

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="mrm8488/CodeBERTaPy",
    tokenizer="mrm8488/CodeBERTaPy"
)

fill_mask(PYTHON_CODE)

## Top 5 predictions:

'val' # prob  0.980728805065155
'value'
'idx'
',val'
'_'
```

### Yes! That was easy 🎉 Let's try with another Flask like example

```python
PYTHON_CODE2 = """
@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(<mask>)

if __name__ == '__main__':
    app.run()
""".lstrip()


fill_mask(PYTHON_CODE2)

## Top 5 predictions:

'name' # prob  0.9961813688278198
' name'
'url'
'description'
'self'
```

### Yeah! It works 🎉 Let's try with another Tensorflow/Keras like example

```python
PYTHON_CODE3="""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.<mask>(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
""".lstrip()


fill_mask(PYTHON_CODE3)

## Top 5 predictions:

'Dense' # prob   0.4482928514480591
'relu'
'Flatten'
'Activation'
'Conv'
```

> Great! 🎉

## This work is heavely inspired on [CodeBERTa](https://github.com/huggingface/transformers/blob/master/model_cards/huggingface/CodeBERTa-small-v1/README.md) by huggingface team

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
