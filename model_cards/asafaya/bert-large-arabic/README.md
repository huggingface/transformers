---
language: arabic
---

# Arabic BERT Large Model

Pretrained BERT Large language model for Arabic

_If you use this model in your work, please cite this paper (to appear in 2020):_

```
@inproceedings{
  title={KUISAIL at SemEval-2020 Task 12: BERT-CNN for Offensive Speech Identification in Social Media},
  author={Safaya, Ali and Abdullatif, Moutasem and Yuret, Deniz},
  booktitle={Proceedings of the International Workshop on Semantic Evaluation (SemEval)},
  year={2020}
}
```

## Pretraining Corpus

`arabic-bert-large` model was pretrained on ~8.2 Billion words:

- Arabic version of [OSCAR](https://traces1.inria.fr/oscar/) - filtered from [Common Crawl](http://commoncrawl.org/)
- Recent dump of Arabic [Wikipedia](https://dumps.wikimedia.org/backup-index.html)

and other Arabic resources which sum up to ~95GB of text.

__Notes on training data:__

- Our final version of corpus contains some non-Arabic words inlines, which we did not remove from sentences since that would affect some tasks like NER.
- Although non-Arabic characters were lowered as a preprocessing step, since Arabic characters does not have upper or lower case, there is no cased and uncased version of the model.
- The corpus and vocabulary set are not restricted to Modern Standard Arabic, they contain some dialectical Arabic too.

## Pretraining details

- This model was trained using Google BERT's github [repository](https://github.com/google-research/bert) on a single TPU v3-8 provided for free from [TFRC](https://www.tensorflow.org/tfrc).
- Our pretraining procedure follows training settings of bert with some changes: trained for 3M training steps with batchsize of 128, instead of 1M with batchsize of 256.

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-large-arabic")
model = AutoModel.from_pretrained("asafaya/bert-large-arabic")
```

## Results

For further details on the models performance or any other queries, please refer to [Arabic-BERT](https://github.com/alisafaya/Arabic-BERT)

## Acknowledgement

Thanks to Google for providing free TPU for the training process and for Huggingface for hosting this model on their servers 😊


