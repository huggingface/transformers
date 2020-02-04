# Tensorflow CamemBERT

In this repository you will find different versions of the CamemBERT model for Tensorflow.

## CamemBERT

[CamemBERT](https://camembert-model.fr/) is a state-of-the-art language model for French based on the RoBERTa architecture pretrained on the French subcorpus of the newly available multilingual corpus OSCAR.

## Model Weights

| Model                            | Downloads
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `jplu/tf-camembert-base`   | [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-camembert-base/config.json) • [`tf_model.h5`](https://s3.amazonaws.com/models.huggingface.co/bert/jplu/tf-camembert-base/tf_model.h5)

## Usage

With Transformers >= 2.4 the Tensorflow models of CamemBERT can be loaded like:

```python
from transformers import TFCamembertModel

model = TFCamembertModel.from_pretrained("jplu/tf-camembert-base")
```

## Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/jplu).

## Acknowledgments

Thanks to all the Huggingface team for the support and their amazing library!
