---
language: greek
thumbnail: https://github.com/nlpaueb/GreekBERT/raw/master/greek-bert-logo.png
---

# GreekBERT

A Greek version of BERT pre-trained language model.

<img src="https://github.com/nlpaueb/GreekBERT/raw/master/greek-bert-logo.png" width="600"/> 


## Pre-training corpora

The pre-training corpora of `bert-base-greek-uncased-v1` include:

* The Greek part of [Wikipedia](https://el.wikipedia.org/wiki/Βικιπαίδεια:Αντίγραφα_της_βάσης_δεδομένων),
* The Greek part of [European Parliament Proceedings Parallel Corpus](https://www.statmt.org/europarl/), and
* The Greek part of [OSCAR](https://traces1.inria.fr/oscar/), a cleansed version of [Common Crawl](https://commoncrawl.org).

Future release will also include:

* The entire corpus of Greek legislation, as published by the [National Publication Office](http://www.et.gr),  
* The entire corpus of EU legislation (Greek translation), as published in [Eur-Lex](https://eur-lex.europa.eu/homepage.html?locale=en).

## Pre-training details

* We trained BERT using the official code provided in Google BERT's github repository (https://github.com/google-research/bert). We then used [Hugging Face](https://huggingface.co)'s [Transformers](https://github.com/huggingface/transformers) conversion script to convert the TF checkpoint and vocabulary in the desirable format in order to be able to load the model in two lines of code for both PyTorch and TF2 users.
* We released a model similar to the English `bert-base-uncased` model (12-layer, 768-hidden, 12-heads, 110M parameters).
* We chose to follow the same training set-up: 1 million training steps with batches of 256 sequences of length 512 with an initial learning rate 1e-4.
* We were able to use a single Google Cloud TPU v3-8 provided for free from [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc), while also utilizing [GCP research credits](https://edu.google.com/programs/credits/research). Huge thanks to both Google programs for supporting us!


## Requirements

We published `bert-base-greek-uncased-v1` as part of [Hugging Face](https://huggingface.co)'s [Transformers](https://github.com/huggingface/transformers) repository. So, you need to install the transfomers library through pip along with PyTorch or Tensorflow 2.

```
pip install transfomers
pip install (torch|tensorflow)
```

## Pre-process text (Deaccent - Lower)

In order to use `bert-base-greek-uncased-v1`, you have to pre-process texts to lowercase letters and remove all Greek diacritics.

```python

import unicodedata

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

accented_string = "Αυτή είναι η Ελληνική έκδοση του BERT."
unaccented_string = strip_accents_and_lowercase(accented_string)

print(unaccented_string) # αυτη ειναι η ελληνικη εκδοση του bert.

```

## Load Pretrained Model 

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
```

## Use Pretrained Model as a Language Model

```python
import torch
from transformers import *

# Load model and tokenizer
tokenizer_greek = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
lm_model_greek = AutoModelWithLMHead.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

# ================ EXAMPLE 1 ================
text_1 = 'O ποιητής έγραψε ένα [MASK] .'
# EN: 'The poet wrote a [MASK].'
input_ids = tokenizer_greek.encode(text_1)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'o', 'ποιητης', 'εγραψε', 'ενα', '[MASK]', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 5].max(0)[1].item()))
# the most plausible prediction for [MASK] is "song"

# ================ EXAMPLE 2 ================
text_2 = 'Είναι ένας [MASK] άνθρωπος.'
# EN: 'He is a [MASK] person.'
input_ids = tokenizer_greek.encode(text_1)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 3].max(0)[1].item()))
# the most plausible prediction for [MASK] is "good"

# ================ EXAMPLE 3 ================
text_3 = 'Είναι ένας [MASK] άνθρωπος και κάνει συχνά [MASK].'
# EN: 'He is a [MASK] person he does frequently [MASK].'
input_ids = tokenizer_greek.encode(text_3)
print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', 'και', 'κανει', 'συχνα', '[MASK]', '.', '[SEP]']
outputs = lm_model_greek(torch.tensor([input_ids]))[0]
print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 8].max(0)[1].item()))
# the most plausible prediction for the second [MASK] is "trips"
```

## Evaluation on downstream tasks

TBA

## Author

Ilias Chalkidis on behalf of [AUEB's Natural Language Processing Group](http://nlp.cs.aueb.gr)

| Github: [@ilias.chalkidis](https://github.com/seolhokim) | Twitter: [@KiddoThe2B](https://twitter.com/KiddoThe2B) |

## About Us

[AUEB's Natural Language Processing Group](http://nlp.cs.aueb.gr) develops algorithms, models, and systems that allow computers to process and generate natural language texts.

The group's current research interests include:
* question answering systems for databases, ontologies, document collections, and the Web, especially biomedical question answering,
* natural language generation from databases and ontologies, especially Semantic Web ontologies,
text classification, including filtering spam and abusive content,
* information extraction and opinion mining, including legal text analytics and sentiment analysis,
* natural language processing tools for Greek, for example parsers and named-entity recognizers,
machine learning in natural language processing, especially deep learning.

The group is part of the Information Processing Laboratory of the Department of Informatics of the Athens University of Economics and Business.
