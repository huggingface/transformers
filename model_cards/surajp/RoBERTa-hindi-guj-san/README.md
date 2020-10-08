---
language:
- hi
- sa
- gu
tags:
- Indic
license: mit
datasets:
- Wikipedia (Hindi, Sanskrit, Gujarati)
metrics:
- perplexity
---

# RoBERTa-hindi-guj-san

## Model description

Multillingual RoBERTa like model trained on Wikipedia articles of Hindi, Sanskrit, Gujarati languages. The tokenizer was trained on combined text. 
However, Hindi text was used to pre-train the model and then it was fine-tuned on Sanskrit and Gujarati Text combined hoping that pre-training with Hindi 
will help the model learn similar languages.

### Configuration

| Parameter | Value |
|---|---|
| `hidden_size` | 768 |
| `num_attention_heads` | 12 |
| `num_hidden_layers` | 6 |
| `vocab_size` | 30522 |
|`model_type`|`roberta`|

## Intended uses & limitations

#### How to use

```python
# Example usage
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline

tokenizer = AutoTokenizer.from_pretrained("surajp/RoBERTa-hindi-guj-san")
model = AutoModelWithLMHead.from_pretrained("surajp/RoBERTa-hindi-guj-san")

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

# Sanskrit: इयं भाषा न केवलं भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।
# Hindi:  अगर आप अब अभ्यास नहीं करते हो तो आप अपने परीक्षा में मूर्खतापूर्ण गलतियाँ करोगे।
# Gujarati: ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો <mask> હતો.
fill_mask("ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો <mask> હતો.")

'''
Output:
--------
[
{'score': 0.07849744707345963, 'sequence': '<s> ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો જ હતો.</s>', 'token': 390},
{'score': 0.06273336708545685, 'sequence': '<s> ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો ન હતો.</s>', 'token': 478},
{'score': 0.05160355195403099, 'sequence': '<s> ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો થઇ હતો.</s>', 'token': 2075},
{'score': 0.04751499369740486, 'sequence': '<s> ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો એક હતો.</s>', 'token': 600},
{'score': 0.03788900747895241, 'sequence': '<s> ગુજરાતમાં ૧૯મી માર્ચ સુધી કોઈ સકારાત્મક (પોઝીટીવ) રીપોર્ટ આવ્યો પણ હતો.</s>', 'token': 840}
]

```

## Training data

Cleaned wikipedia articles in Hindi, Sanskrit and Gujarati on Kaggle. It contains training as well as evaluation text. 
Used in [iNLTK](https://github.com/goru001/inltk)

- [Hindi](https://www.kaggle.com/disisbig/hindi-wikipedia-articles-172k)
- [Gujarati](https://www.kaggle.com/disisbig/gujarati-wikipedia-articles)
- [Sanskrit](https://www.kaggle.com/disisbig/sanskrit-wikipedia-articles)

## Training procedure

- On TPU (using `xla_spawn.py`)
- For language modelling
- Iteratively increasing `--block_size` from 128 to 256 over epochs
- Tokenizer trained on combined text
- Pre-training with Hindi and fine-tuning on Sanskrit and Gujarati texts

```
--model_type distillroberta-base \
--model_name_or_path "/content/SanHiGujBERTa" \
--mlm_probability 0.20 \
--line_by_line \
--save_total_limit 2 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--num_train_epochs 5 \
--block_size 256 \
--seed 108 \
--overwrite_output_dir \
```

## Eval results

perplexity = 2.920005983224673



> Created by [Suraj Parmar/@parmarsuraj99](https://twitter.com/parmarsuraj99) | [LinkedIn](https://www.linkedin.com/in/parmarsuraj99/)

> Made with <span style="color: #e25555;">&hearts;</span> in India
