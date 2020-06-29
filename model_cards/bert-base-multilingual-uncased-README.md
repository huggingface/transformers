---
language: english
license: apache-2.0
datasets:
- wikipedia
---

# BERT multilingual base model (uncased)

Pretrained model on the top 102 languages with the largest Wikipedia using a masked language modeling (MLM) objective.
It was introduced in [this paper](https://arxiv.org/abs/1810.04805) and first released in
[this repository](https://github.com/google-research/bert). This model is uncased: it does not make a difference
between english and English.

Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by
the Hugging Face team.

## Model description

BERT is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion. This means
it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the languages in the training set that can then be used to
extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a
standard classifier using the features produced by the BERT model as inputs.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=bert) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-multilingual-uncased')
>>> unmasker("Hello I'm a [MASK] model.")

[{'sequence': "[CLS] hello i'm a top model. [SEP]",
  'score': 0.1507750153541565,
  'token': 11397,
  'token_str': 'top'},
 {'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.13075384497642517,
  'token': 23589,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a good model. [SEP]",
  'score': 0.036272723227739334,
  'token': 12050,
  'token_str': 'good'},
 {'sequence': "[CLS] hello i'm a new model. [SEP]",
  'score': 0.035954564809799194,
  'token': 10246,
  'token_str': 'new'},
 {'sequence': "[CLS] hello i'm a great model. [SEP]",
  'score': 0.028643041849136353,
  'token': 11838,
  'token_str': 'great'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = TFBertModel.from_pretrained("bert-base-multilingual-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

### Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased
predictions:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-multilingual-uncased')
>>> unmasker("The man worked as a [MASK].")

[{'sequence': '[CLS] the man worked as a teacher. [SEP]',
  'score': 0.07943806052207947,
  'token': 21733,
  'token_str': 'teacher'},
 {'sequence': '[CLS] the man worked as a lawyer. [SEP]',
  'score': 0.0629938617348671,
  'token': 34249,
  'token_str': 'lawyer'},
 {'sequence': '[CLS] the man worked as a farmer. [SEP]',
  'score': 0.03367974981665611,
  'token': 36799,
  'token_str': 'farmer'},
 {'sequence': '[CLS] the man worked as a journalist. [SEP]',
  'score': 0.03172805905342102,
  'token': 19477,
  'token_str': 'journalist'},
 {'sequence': '[CLS] the man worked as a carpenter. [SEP]',
  'score': 0.031021825969219208,
  'token': 33241,
  'token_str': 'carpenter'}]

>>> unmasker("The Black woman worked as a [MASK].")

[{'sequence': '[CLS] the black woman worked as a nurse. [SEP]',
  'score': 0.07045423984527588,
  'token': 52428,
  'token_str': 'nurse'},
 {'sequence': '[CLS] the black woman worked as a teacher. [SEP]',
  'score': 0.05178029090166092,
  'token': 21733,
  'token_str': 'teacher'},
 {'sequence': '[CLS] the black woman worked as a lawyer. [SEP]',
  'score': 0.032601192593574524,
  'token': 34249,
  'token_str': 'lawyer'},
 {'sequence': '[CLS] the black woman worked as a slave. [SEP]',
  'score': 0.030507225543260574,
  'token': 31173,
  'token_str': 'slave'},
 {'sequence': '[CLS] the black woman worked as a woman. [SEP]',
  'score': 0.027691684663295746,
  'token': 14050,
  'token_str': 'woman'}]
```

This bias will also affect all fine-tuned versions of this model.

## Training data

The BERT model was pretrained on the 102 languages with the largest Wikipedias. You can find the complete list
[here](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a shared vocabulary size of 110,000. The languages with a
larger Wikipedia are under-sampled and the ones with lower resources are oversampled. For languages like Chinese,
Japanese Kanji and Korean Hanja that don't have space, a CJK Unicode block is added around every character. 

The inputs of the model are then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus and in
the other cases, it's another random sentence in the corpus. Note that what is considered a sentence here is a
consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two
"sentences" has a combined length of less than 512 tokens.

The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.


### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
