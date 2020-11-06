---
language: it
license: mit
datasets:
- wikipedia
---

# 🤗 + 📚 dbmdz BERT and ELECTRA models

In this repository the MDZ Digital Library team (dbmdz) at the Bavarian State
Library open sources Italian BERT and ELECTRA models 🎉

# Italian BERT

The source data for the Italian BERT model consists of a recent Wikipedia dump and
various texts from the [OPUS corpora](http://opus.nlpl.eu/) collection. The final
training corpus has a size of 13GB and 2,050,057,573 tokens.

For sentence splitting, we use NLTK (faster compared to spacy).
Our cased and uncased models are training with an initial sequence length of 512
subwords for ~2-3M steps.

For the XXL Italian models, we use the same training data from OPUS and extend
it with data from the Italian part of the [OSCAR corpus](https://traces1.inria.fr/oscar/).
Thus, the final training corpus has a size of 81GB and 13,138,379,147 tokens.

Note: Unfortunately, a wrong vocab size was used when training the XXL models.
This explains the mismatch of the "real" vocab size of 31102, compared to the
vocab size specified in `config.json`. However, the model is working and all
evaluations were done under those circumstances.
See [this issue](https://github.com/dbmdz/berts/issues/7) for more information.

The Italian ELECTRA model was trained on the "XXL" corpus for 1M steps in total using a batch
size of 128. We pretty much following the ELECTRA training procedure as used for
[BERTurk](https://github.com/stefan-it/turkish-bert/tree/master/electra).

## Model weights

Currently only PyTorch-[Transformers](https://github.com/huggingface/transformers)
compatible weights are available. If you need access to TensorFlow checkpoints,
please raise an issue!

| Model                                                | Downloads
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `dbmdz/bert-base-italian-cased`                      | [`config.json`](https://cdn.huggingface.co/dbmdz/bert-base-italian-cased/config.json)                                               • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/bert-base-italian-cased/pytorch_model.bin)                      • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/bert-base-italian-cased/vocab.txt)
| `dbmdz/bert-base-italian-uncased`                    | [`config.json`](https://cdn.huggingface.co/dbmdz/bert-base-italian-uncased/config.json)                                             • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/bert-base-italian-uncased/pytorch_model.bin)                    • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/bert-base-italian-uncased/vocab.txt)
| `dbmdz/bert-base-italian-xxl-cased`                  | [`config.json`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-cased/config.json)                                           • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-cased/pytorch_model.bin)                  • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-cased/vocab.txt)
| `dbmdz/bert-base-italian-xxl-uncased`                | [`config.json`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-uncased/config.json)                                         • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-uncased/pytorch_model.bin)                • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/bert-base-italian-xxl-uncased/vocab.txt)
| `dbmdz/electra-base-italian-xxl-cased-discriminator` | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/electra-base-italian-xxl-cased-discriminator/config.json) • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/electra-base-italian-xxl-cased-discriminator/pytorch_model.bin) • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/electra-base-italian-xxl-cased-discriminator/vocab.txt)
| `dbmdz/electra-base-italian-xxl-cased-generator`     | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/electra-base-italian-xxl-cased-generator/config.json)     • [`pytorch_model.bin`](https://cdn.huggingface.co/dbmdz/electra-base-italian-xxl-cased-generator/pytorch_model.bin)     • [`vocab.txt`](https://cdn.huggingface.co/dbmdz/electra-base-italian-xxl-cased-generator/vocab.txt)

## Results

For results on downstream tasks like NER or PoS tagging, please refer to
[this repository](https://github.com/stefan-it/italian-bertelectra).

## Usage

With Transformers >= 2.3 our Italian BERT models can be loaded like:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/bert-base-italian-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
```

To load the (recommended) Italian XXL BERT models, just use:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/bert-base-italian-xxl-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
```

To load the Italian XXL ELECTRA model (discriminator), just use:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/electra-base-italian-xxl-cased-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelWithLMHead.from_pretrained(model_name)
```

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our BERT/ELECTRA models just open an issue
[here](https://github.com/dbmdz/berts/issues/new) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
