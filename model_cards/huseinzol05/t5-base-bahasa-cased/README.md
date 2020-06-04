---
language: malay
---

# Bahasa T5 Model

Pretrained T5 base language model for Malay and Indonesian. 

## Pretraining Corpus

`t5-base-bahasa-cased` model was pretrained on multiple tasks. Below is list of tasks we trained on,

1. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [local Wikipedia](https://github.com/huseinzol05/Malaya-Dataset#wikipedia-1).
2. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [local news](https://github.com/huseinzol05/Malaya-Dataset#public-news).
3. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [local parliament text](https://github.com/huseinzol05/Malaya-Dataset#parliament).
4. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [IIUM Confession](https://github.com/huseinzol05/Malaya-Dataset#iium-confession).
5. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [Wattpad](https://github.com/huseinzol05/Malaya-Dataset#wattpad).
6. [Unsupervised](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1875) on [Academia PDF](https://github.com/huseinzol05/Malaya-Dataset#academia-pdf).
7. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [local Wikipedia](https://github.com/huseinzol05/Malaya-Dataset#wikipedia-1).
8. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [local news](https://github.com/huseinzol05/Malaya-Dataset#public-news).
9. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [local parliament text](https://github.com/huseinzol05/Malaya-Dataset#parliament).
10. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [IIUM Confession](https://github.com/huseinzol05/Malaya-Dataset#iium-confession).
11. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [Wattpad](https://github.com/huseinzol05/Malaya-Dataset#wattpad).
12. [Next sentence prediction](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L1129) on [Academia PDF](https://github.com/huseinzol05/Malaya-Dataset#academia-pdf).
13. [Bahasa SNLI](https://github.com/huseinzol05/Malaya-Dataset#snli).
14. [Bahasa Question Quora](https://github.com/huseinzol05/Malaya-Dataset#quora).
15. [Bahasa Natural Questions](https://github.com/huseinzol05/Malaya-Dataset#natural-questions).
16. [News title summarization](https://github.com/huseinzol05/Malaya-Dataset#crawled-news).
17. [Stemming to original wikipedia](https://github.com/huseinzol05/Malaya/blob/master/pretrained-model/t5/generate-stemming.ipynb).
18. [Synonym to original wikipedia](https://github.com/huseinzol05/Malaya/blob/master/pretrained-model/t5/generate-synonym.ipynb).

Preprocessing steps can reproduce from here, [Malaya/pretrained-model/preprocess](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/preprocess).

## Pretraining details

- This model was trained using Google T5's github [repository](https://github.com/google-research/text-to-text-transfer-transformer) on v3-8 TPU.
- All steps can reproduce from here, [Malaya/pretrained-model/t5](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/t5).

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import T5Tokenizer, T5Model

model = T5Model.from_pretrained('huseinzol05/t5-base-bahasa-cased')
tokenizer = T5Tokenizer.from_pretrained('huseinzol05/t5-base-bahasa-cased')
```

## Example using T5ForConditionalGeneration

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('huseinzol05/t5-base-bahasa-cased')
model = T5ForConditionalGeneration.from_pretrained('huseinzol05/t5-base-bahasa-cased')
input_ids = tokenizer.encode('soalan: siapakah perdana menteri malaysia?', return_tensors = 'pt')
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

Output is,

```
'Mahathir Mohamad'
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train T5 for Bahasa. 
