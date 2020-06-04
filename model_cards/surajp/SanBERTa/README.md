---
language: sanskrit
---

# RoBERTa trained on Sanskrit (SanBERTa)

**Mode size** (after training): **340MB**

### Dataset:

[Wikipedia articles](https://www.kaggle.com/disisbig/sanskrit-wikipedia-articles) (used in [iNLTK](https://github.com/goru001/nlp-for-sanskrit)).
It contains evaluation set.

[Sanskrit scraps from CLTK](http://cltk.org/)

### Configuration

| Parameter | Value |
|---|---|
| `num_attention_heads` | 12 |
| `num_hidden_layers` | 6 |
| `hidden_size` | 768 |
| `vocab_size` | 29407 |

### Training :
- On TPU
- For language modelling
- Iteratively increasing `--block_size` from 128 to 256 over epochs

### Evaluation

|Metric| # Value |
|---|---|
|Perplexity (`block_size=256`)|4.04|

## Example of usage:

### For Embeddings

```

tokenizer = AutoTokenizer.from_pretrained("surajp/SanBERTa")
model = RobertaModel.from_pretrained("surajp/SanBERTa")

op = tokenizer.encode("इयं भाषा न केवलं भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।", return_tensors="pt")
ps = model(op)
ps[0].shape

```
```
'''
Output:
--------
torch.Size([1, 47, 768])

```


### For \<mask\> Prediction

```
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="surajp/SanBERTa",
    tokenizer="surajp/SanBERTa"
)

## इयं भाषा न केवलं भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।
fill_mask("इयं भाषा न केवल<mask> भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।")

ps = model(torch.tensor(enc).unsqueeze(1))
print(ps[0].shape)
```
```
'''
Output:
--------
[{'score': 0.7516744136810303,
  'sequence': '<s> इयं भाषा न केवलं भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।</s>',
  'token': 280,
  'token_str': 'à¤Ĥ'},
 {'score': 0.06230105459690094,
  'sequence': '<s> इयं भाषा न केवली भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।</s>',
  'token': 289,
  'token_str': 'à¥Ģ'},
 {'score': 0.055410224944353104,
  'sequence': '<s> इयं भाषा न केवला भारतस्य अपि तु विश्वस्य प्राचीनतमा भाषा इति मन्यते।</s>',
  'token': 265,
  'token_str': 'à¤¾'},
  ...]
```

### It works!! 🎉 🎉 🎉

> Created by [Suraj Parmar/@parmarsuraj99](https://twitter.com/parmarsuraj99) | [LinkedIn](https://www.linkedin.com/in/parmarsuraj99/)

> Made with <span style="color: #e25555;">&hearts;</span> in India
