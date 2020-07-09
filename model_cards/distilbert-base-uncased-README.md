---
language: english
tags:
- exbert
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# DistilBERT base model (uncased)

This model is a distilled version of the [BERT base mode](https://huggingface.co/distilbert-base-uncased). It was
introduced in [this paper](https://arxiv.org/abs/1910.01108). The code for the distillation process can be found
[here](https://github.com/huggingface/transformers/tree/master/examples/distillation). This model is uncased: it does
not make a difference between english and English.

## Model description

DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a
self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only,
with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic
process to generate inputs and labels from those texts using the BERT base model. More precisely, it was pretrained
with three objectives:

- Distillation loss: the model was trained to return the same probabilities as the BERT base model.
- Masked language modeling (MLM): this is part of the original training loss of the BERT base model. When taking a
  sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the
  model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that
  usually see the words one after the other, or from autoregressive models like GPT which internally mask the future
  tokens. It allows the model to learn a bidirectional representation of the sentence.
- Cosine embedding loss: the model was also trained to generate hidden states as close as possible as the BERT base
  model.

This way, the model learns the same inner representation of the English language than its teacher model, while being
faster for inference or downstream tasks.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=distilbert) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
>>> unmasker("Hello I'm a [MASK] model.")

[{'sequence': "[CLS] hello i'm a role model. [SEP]",
  'score': 0.05292855575680733,
  'token': 2535,
  'token_str': 'role'},
 {'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.03968575969338417,
  'token': 4827,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a business model. [SEP]",
  'score': 0.034743521362543106,
  'token': 2449,
  'token_str': 'business'},
 {'sequence': "[CLS] hello i'm a model model. [SEP]",
  'score': 0.03462274372577667,
  'token': 2944,
  'token_str': 'model'},
 {'sequence': "[CLS] hello i'm a modeling model. [SEP]",
  'score': 0.018145186826586723,
  'token': 11643,
  'token_str': 'modeling'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import DistilBertTokenizer, TFDistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

### Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased
predictions. It also inherits some of
[the bias of its teacher model](https://huggingface.co/bert-base-uncased#limitations-and-bias). 

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
>>> unmasker("The White man worked as a [MASK].")

[{'sequence': '[CLS] the white man worked as a blacksmith. [SEP]',
  'score': 0.1235365942120552,
  'token': 20987,
  'token_str': 'blacksmith'},
 {'sequence': '[CLS] the white man worked as a carpenter. [SEP]',
  'score': 0.10142576694488525,
  'token': 10533,
  'token_str': 'carpenter'},
 {'sequence': '[CLS] the white man worked as a farmer. [SEP]',
  'score': 0.04985016956925392,
  'token': 7500,
  'token_str': 'farmer'},
 {'sequence': '[CLS] the white man worked as a miner. [SEP]',
  'score': 0.03932540491223335,
  'token': 18594,
  'token_str': 'miner'},
 {'sequence': '[CLS] the white man worked as a butcher. [SEP]',
  'score': 0.03351764753460884,
  'token': 14998,
  'token_str': 'butcher'}]

>>> unmasker("The Black woman worked as a [MASK].")

[{'sequence': '[CLS] the black woman worked as a waitress. [SEP]',
  'score': 0.13283951580524445,
  'token': 13877,
  'token_str': 'waitress'},
 {'sequence': '[CLS] the black woman worked as a nurse. [SEP]',
  'score': 0.12586183845996857,
  'token': 6821,
  'token_str': 'nurse'},
 {'sequence': '[CLS] the black woman worked as a maid. [SEP]',
  'score': 0.11708822101354599,
  'token': 10850,
  'token_str': 'maid'},
 {'sequence': '[CLS] the black woman worked as a prostitute. [SEP]',
  'score': 0.11499975621700287,
  'token': 19215,
  'token_str': 'prostitute'},
 {'sequence': '[CLS] the black woman worked as a housekeeper. [SEP]',
  'score': 0.04722772538661957,
  'token': 22583,
  'token_str': 'housekeeper'}]
```

This bias will also affect all fine-tuned versions of this model.

## Training data

DistilBERT pretrained on the same data as BERT, which is [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset
consisting of 11,038 unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia)
(excluding lists, tables and headers).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

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

### Pretraining

The model was trained on 8 16 GB V100 for 90 hours. See the
[training code](https://github.com/huggingface/transformers/tree/master/examples/distillation) for all hyperparameters
details.

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Average |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:-------:|
|      | 82.2 | 88.5 | 89.2 | 91.3  | 51.3 | 85.8  | 87.5 | 59.9 | 77.0    |


### BibTeX entry and citation info

```bibtex
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
```

<a href="https://huggingface.co/exbert/?model=distilbert-base-uncased">
	<img width="300px" src="https://hf-dinosaur.huggingface.co/exbert/button.png">
</a>
