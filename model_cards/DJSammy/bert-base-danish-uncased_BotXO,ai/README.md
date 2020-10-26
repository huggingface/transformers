---
language: da
tags:
- bert
- masked-lm
- lm-head
license: cc-by-4.0
datasets:
- common_crawl
- wikipedia
pipeline_tag: fill-mask
widget:
- text: "København er [MASK] i Danmark."
---

# Danish BERT (uncased) model 

[BotXO.ai](https://www.botxo.ai/) developed this model. For data and training details see their [GitHub repository](https://github.com/botxo/nordic_bert).  

The original model was trained in TensorFlow then I converted it to Pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html?highlight=cli).

For TensorFlow version download here: https://www.dropbox.com/s/19cjaoqvv2jicq9/danish_bert_uncased_v2.zip?dl=1


## Architecture

```python
from transformers import AutoModelForPreTraining

model = AutoModelForPreTraining.from_pretrained("DJSammy/bert-base-danish-uncased_BotXO,ai")

params = list(model.named_parameters())
print('danish_bert_uncased_v2 has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Last Transformer ====\n')
for p in params[181:197]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')
for p in params[197:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# danish_bert_uncased_v2 has 206 different named parameters.

# ==== Embedding Layer ====

# bert.embeddings.word_embeddings.weight                  (32000, 768)
# bert.embeddings.position_embeddings.weight                (512, 768)
# bert.embeddings.token_type_embeddings.weight                (2, 768)
# bert.embeddings.LayerNorm.weight                              (768,)
# bert.embeddings.LayerNorm.bias                                (768,)

# ==== First Transformer ====

# bert.encoder.layer.0.attention.self.query.weight          (768, 768)
# bert.encoder.layer.0.attention.self.query.bias                (768,)
# bert.encoder.layer.0.attention.self.key.weight            (768, 768)
# bert.encoder.layer.0.attention.self.key.bias                  (768,)
# bert.encoder.layer.0.attention.self.value.weight          (768, 768)
# bert.encoder.layer.0.attention.self.value.bias                (768,)
# bert.encoder.layer.0.attention.output.dense.weight        (768, 768)
# bert.encoder.layer.0.attention.output.dense.bias              (768,)
# bert.encoder.layer.0.attention.output.LayerNorm.weight        (768,)
# bert.encoder.layer.0.attention.output.LayerNorm.bias          (768,)
# bert.encoder.layer.0.intermediate.dense.weight           (3072, 768)
# bert.encoder.layer.0.intermediate.dense.bias                 (3072,)
# bert.encoder.layer.0.output.dense.weight                 (768, 3072)
# bert.encoder.layer.0.output.dense.bias                        (768,)
# bert.encoder.layer.0.output.LayerNorm.weight                  (768,)
# bert.encoder.layer.0.output.LayerNorm.bias                    (768,)

# ==== Last Transformer ====

# bert.encoder.layer.11.attention.self.query.weight         (768, 768)
# bert.encoder.layer.11.attention.self.query.bias               (768,)
# bert.encoder.layer.11.attention.self.key.weight           (768, 768)
# bert.encoder.layer.11.attention.self.key.bias                 (768,)
# bert.encoder.layer.11.attention.self.value.weight         (768, 768)
# bert.encoder.layer.11.attention.self.value.bias               (768,)
# bert.encoder.layer.11.attention.output.dense.weight       (768, 768)
# bert.encoder.layer.11.attention.output.dense.bias             (768,)
# bert.encoder.layer.11.attention.output.LayerNorm.weight       (768,)
# bert.encoder.layer.11.attention.output.LayerNorm.bias         (768,)
# bert.encoder.layer.11.intermediate.dense.weight          (3072, 768)
# bert.encoder.layer.11.intermediate.dense.bias                (3072,)
# bert.encoder.layer.11.output.dense.weight                (768, 3072)
# bert.encoder.layer.11.output.dense.bias                       (768,)
# bert.encoder.layer.11.output.LayerNorm.weight                 (768,)
# bert.encoder.layer.11.output.LayerNorm.bias                   (768,)

# ==== Output Layer ====

# bert.pooler.dense.weight                                  (768, 768)
# bert.pooler.dense.bias                                        (768,)
# cls.predictions.bias                                        (32000,)
# cls.predictions.transform.dense.weight                    (768, 768)
# cls.predictions.transform.dense.bias                          (768,)
# cls.predictions.transform.LayerNorm.weight                    (768,)
# cls.predictions.transform.LayerNorm.bias                      (768,)
# cls.seq_relationship.weight                                 (2, 768)
# cls.seq_relationship.bias                                       (2,)
```

## Example Pipeline

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='DJSammy/bert-base-danish-uncased_BotXO,ai')

unmasker('København er [MASK] i Danmark.')

# Copenhagen is the [MASK] of Denmark.
# =>

# [{'score': 0.788068950176239,
#  'sequence': '[CLS] københavn er hovedstad i danmark. [SEP]',
#  'token': 12610,
#  'token_str': 'hovedstad'},
# {'score': 0.07606703042984009,
#  'sequence': '[CLS] københavn er hovedstaden i danmark. [SEP]',
#  'token': 8108,
#  'token_str': 'hovedstaden'},
# {'score': 0.04299738258123398,
#  'sequence': '[CLS] københavn er metropol i danmark. [SEP]',
#  'token': 23305,
#  'token_str': 'metropol'},
# {'score': 0.008163209073245525,
#  'sequence': '[CLS] københavn er ikke i danmark. [SEP]',
#  'token': 89,
#  'token_str': 'ikke'},
# {'score': 0.006238455418497324,
#  'sequence': '[CLS] københavn er ogsa i danmark. [SEP]',
#  'token': 25253,
#  'token_str': 'ogsa'}]
```
