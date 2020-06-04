---
language: turkish
license: mit
---

# 🤗 + 📚 dbmdz Turkish BERT model

In this repository the MDZ Digital Library team (dbmdz) at the Bavarian State
Library open sources an uncased model for Turkish 🎉

# 🇹🇷 BERTurk

BERTurk is a community-driven uncased BERT model for Turkish.

Some datasets used for pretraining and evaluation are contributed from the
awesome Turkish NLP community, as well as the decision for the model name: BERTurk.

## Stats

The current version of the model is trained on a filtered and sentence
segmented version of the Turkish [OSCAR corpus](https://traces1.inria.fr/oscar/),
a recent Wikipedia dump, various [OPUS corpora](http://opus.nlpl.eu/) and a
special corpus provided by [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/).

The final training corpus has a size of 35GB and 44,04,976,662 tokens.

Thanks to Google's TensorFlow Research Cloud (TFRC) we could train an uncased model
on a TPU v3-8 for 2M steps.

For this model we use a vocab size of 128k.

## Model weights

Currently only PyTorch-[Transformers](https://github.com/huggingface/transformers)
compatible weights are available. If you need access to TensorFlow checkpoints,
please raise an issue!

| Model                                  | Downloads
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `dbmdz/bert-base-turkish-128k-uncased` | [`config.json`](https://cdn.huggingface.co/dbmdz/bert-base-turkish-128k-uncased/config.json) • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/bert-base-turkish-128k-uncased/pytorch_model.bin) • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/bert-base-turkish-128k-uncased/vocab.txt)

## Usage

With Transformers >= 2.3 our BERTurk uncased model can be loaded like:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
```

## Results

For results on PoS tagging or NER tasks, please refer to
[this repository](https://github.com/stefan-it/turkish-bert).

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our BERT models just open an issue
[here](https://github.com/dbmdz/berts/issues/new) 🤗

# Acknowledgments

Thanks to [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/) for providing us
additional large corpora for Turkish. Many thanks to Reyyan Yeniterzi for providing
us the Turkish NER dataset for evaluation.

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
