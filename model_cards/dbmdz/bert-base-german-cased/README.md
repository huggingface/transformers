---
language: german
---

# 🤗 + 📚 dbmdz German BERT models

In this repository the MDZ Digital Library team (dbmdz) at the Bavarian State
Library open sources another German BERT models 🎉

# German BERT

## Stats

In addition to the recently released [German BERT](https://deepset.ai/german-bert)
model by [deepset](https://deepset.ai/) we provide another German-language model.

The source data for the model consists of a recent Wikipedia dump, EU Bookshop corpus,
Open Subtitles, CommonCrawl, ParaCrawl and News Crawl. This results in a dataset with
a size of 16GB and 2,350,234,427 tokens.

For sentence splitting, we use [spacy](https://spacy.io/). Our preprocessing steps
(sentence piece model for vocab generation) follow those used for training
[SciBERT](https://github.com/allenai/scibert). The model is trained with an initial
sequence length of 512 subwords and was performed for 1.5M steps.

This release includes both cased and uncased models.

## Model weights

Currently only PyTorch-[Transformers](https://github.com/huggingface/transformers)
compatible weights are available. If you need access to TensorFlow checkpoints,
please raise an issue!

| Model                            | Downloads
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `bert-base-german-dbmdz-cased`   | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json) • [`pytorch_model.bin`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin) • [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt)
| `bert-base-german-dbmdz-uncased` | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json) • [`pytorch_model.bin`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin) • [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt)

## Usage

With Transformers >= 2.3 our German BERT models can be loaded like:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
```

## Results

For results on downstream tasks like NER or PoS tagging, please refer to
[this repository](https://github.com/stefan-it/fine-tuned-berts-seq).

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our BERT models just open an issue
[here](https://github.com/dbmdz/berts/issues/new) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
