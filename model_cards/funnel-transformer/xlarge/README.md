---
language: en
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
- gigaword
---

# Funnel Transformer xlarge model (B10-10-10 with decoder)

Pretrained model on English language using a similar objective objective as [ELECTRA](https://huggingface.co/transformers/model_doc/electra.html). It was introduced in
[this paper](https://arxiv.org/pdf/2006.03236.pdf) and first released in
[this repository](https://github.com/laiguokun/Funnel-Transformer). This model is uncased: it does not make a difference
between english and English.

Disclaimer: The team releasing Funnel Transformer did not write a model card for this model so this model card has been
written by the Hugging Face team.

## Model description

Funnel Transformer is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it
was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. 

More precisely, a small language model corrupts the input texts and serves as a generator of inputs for this model, and
the pretraining objective is to predict which token is an original and which one has been replaced, a bit like a GAN training.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the BERT model as inputs.

## Intended uses & limitations

You can use the raw model to extract a vector representation of a given text, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=funnel-transformer) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use


Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import FunnelTokenizer, FunnelModel
tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge")
model = FunneModel.from_pretrained("funnel-transformer/xlarge")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import FunnelTokenizer, TFFunnelModel
tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge")
model = TFFunnelModel.from_pretrained("funnel-transformer/xlarge")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

The BERT model was pretrained on:
- [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books,
- [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers),
- [Clue Web](https://lemurproject.org/clueweb12/), a dataset of 733,019,372 English web pages,
- [GigaWord](https://catalog.ldc.upenn.edu/LDC2011T07), an archive of newswire text data,
- [Common Crawl](https://commoncrawl.org/), a dataset of raw web pages.


### BibTeX entry and citation info

```bibtex
@misc{dai2020funneltransformer,
    title={Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing},
    author={Zihang Dai and Guokun Lai and Yiming Yang and Quoc V. Le},
    year={2020},
    eprint={2006.03236},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

