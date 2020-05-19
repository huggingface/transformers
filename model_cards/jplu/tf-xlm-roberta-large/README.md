# Tensorflow XLM-RoBERTa

In this repository you will find different versions of the XLM-RoBERTa model for Tensorflow.

## XLM-RoBERTa

[XLM-RoBERTa](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/) is a scaled cross lingual sentence encoder. It is trained on 2.5T of data across 100 languages data filtered from Common Crawl. XLM-R achieves state-of-the-arts results on multiple cross lingual benchmarks.

## Model Weights

| Model                            | Downloads
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `jplu/tf-xlm-roberta-base`   | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-base/config.json) • [`tf_model.h5`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-base/tf_model.h5)
| `jplu/tf-xlm-roberta-large`   | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-large/config.json) • [`tf_model.h5`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-xlm-roberta-large/tf_model.h5)

## Usage

With Transformers >= 2.4 the Tensorflow models of XLM-RoBERTa can be loaded like:

```python
from transformers import TFXLMRobertaModel

model = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")
```
Or
```
model = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-large")
```

## Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/jplu).

## Acknowledgments

Thanks to all the Huggingface team for the support and their amazing library!
