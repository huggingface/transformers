---
language: english
tags:
- exbert

license: mit
---


# GPT-2

Pretrained model on English language using a causal language modeling (CLM) objective. It was introduced in
[this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
and first released at [this page](https://openai.com/blog/better-language-models/).

Disclaimer: The team releasing GPT-2 also wrote a
[model card](https://github.com/openai/gpt-2/blob/master/model_card.md) for their model. Content from this model card
has been written by the Hugging Face team to complete the information they provided and give specific examples of bias.

## Model description

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This
means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots
of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely,
it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence,
shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the
predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a
prompt.

## Intended uses & limitations

You can use the raw model for text generation or fine-tune it to a downstream task. See the
[model hub](https://huggingface.co/models?filter=gpt2) to look for fine-tuned versions on a task that interests you.

### How to use

You can use this model directly with a pipeline for text generation. Since the generation relies on some randomness, we
set a seed for reproducibility:

```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='gpt2')
>>> set_seed(42)
>>> generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

[{'generated_text': "Hello, I'm a language model, a language for thinking, a language for expressing thoughts."},
 {'generated_text': "Hello, I'm a language model, a compiler, a compiler library, I just want to know how I build this kind of stuff. I don"},
 {'generated_text': "Hello, I'm a language model, and also have more than a few of your own, but I understand that they're going to need some help"},
 {'generated_text': "Hello, I'm a language model, a system model. I want to know my language so that it might be more interesting, more user-friendly"},
 {'generated_text': 'Hello, I\'m a language model, not a language model"\n\nThe concept of "no-tricks" comes in handy later with new'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import GPT2Tokenizer, TFGPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

### Limitations and bias

The training data used for this model has not been released as a dataset one can browse. We know it contains a lot of
unfiltered content from the internet, which is far from neutral. As the openAI team themselves point out in their
[model card](https://github.com/openai/gpt-2/blob/master/model_card.md#out-of-scope-use-cases):

> Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases
> that require the generated text to be true.
>
> Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do
> not recommend that they be deployed into systems that interact with humans > unless the deployers first carry out a
> study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race,
> and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar
> levels of caution around use cases that are sensitive to biases around human attributes.

Here's an example of how the model can have biased predictions:

```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='gpt2')
>>> set_seed(42)
>>> generator("The White man worked as a", max_length=10, num_return_sequences=5)

[{'generated_text': 'The White man worked as a mannequin for'},
 {'generated_text': 'The White man worked as a maniser of the'},
 {'generated_text': 'The White man worked as a bus conductor by day'},
 {'generated_text': 'The White man worked as a plumber at the'},
 {'generated_text': 'The White man worked as a journalist. He had'}]

>>> set_seed(42)
>>> generator("The Black man worked as a", max_length=10, num_return_sequences=5)

[{'generated_text': 'The Black man worked as a man at a restaurant'},
 {'generated_text': 'The Black man worked as a car salesman in a'},
 {'generated_text': 'The Black man worked as a police sergeant at the'},
 {'generated_text': 'The Black man worked as a man-eating monster'},
 {'generated_text': 'The Black man worked as a slave, and was'}]
```

This bias will also affect all fine-tuned versions of this model.

## Training data

The OpenAI team wanted to train this model on a corpus as large as possible. To build it, they scraped all the web
pages from outbound links on Reddit which received at least 3 karma. Note that all Wikipedia pages were removed from
this dataset, so the model was not trained on any part of Wikipedia. The resulting dataset (called WebText) weights
40GB of texts but has not been publicly released. You can find a list of the top 1,000 domains present in WebText
[here](https://github.com/openai/gpt-2/blob/master/domains.txt).

## Training procedure

### Preprocessing

The texts are tokenized using a byte-level version of Byte Pair Encoding (BPE) (for unicode characters) and a
vocabulary size of 50,257. The inputs are sequences of 1024 consecutive tokens.

The larger model was trained on 256 cloud TPU v3 cores. The training duration was not disclosed, nor were the exact
details of training.

## Evaluation results

The model achieves the following results without any fine-tuning (zero-shot):

| Dataset  | LAMBADA | LAMBADA | CBT-CN | CBT-NE | WikiText2 | PTB    | enwiki8 | text8  | WikiText103 | 1BW   |
|:--------:|:-------:|:-------:|:------:|:------:|:---------:|:------:|:-------:|:------:|:-----------:|:-----:|
| (metric) | (PPL)   | (ACC)   | (ACC)  | (ACC)  | (PPL)     | (PPL)  | (BPB)   | (BPC)  | (PPL)       | (PPL) |
|          | 35.13   | 45.99   | 87.65  | 83.4   | 29.41     | 65.85  | 1.16    | 1,17   | 37.50       | 75.20 |


### BibTeX entry and citation info

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

<a href="https://huggingface.co/exbert/?model=gpt2">
	<img width="300px" src="https://hf-dinosaur.huggingface.co/exbert/button.png">
</a>
