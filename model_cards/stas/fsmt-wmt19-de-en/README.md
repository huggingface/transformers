
---
language: de, en
thumbnail:
tags:
- translation
- wmt19
license: Apache 2.0
datasets:
- http://www.statmt.org/wmt19/ ([test-set](http://matrix.statmt.org/test_sets/newstest2019.tgz?1556572561))
metrics:
- http://www.statmt.org/wmt19/metrics-task.html
---

# Model name

## Model description

This is a ported version of [fairseq wmt19 transformer](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md) for de-en.

For more details, please see, [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616).

The abbreviation FSMT stands for FairSeqMachineTranslation

All four models are available:

* [fsmt-wmt19-en-ru](https://huggingface.co/stas/fsmt-wmt19-en-ru)
* [fsmt-wmt19-ru-en](https://huggingface.co/stas/fsmt-wmt19-ru-en)
* [fsmt-wmt19-en-de](https://huggingface.co/stas/fsmt-wmt19-en-de)
* [fsmt-wmt19-de-en](https://huggingface.co/stas/fsmt-wmt19-de-en)

## Intended uses & limitations

#### How to use

```python
from transformers.tokenization_fsmt import FSMTTokenizer
from transformers.modeling_fsmt import FSMTForConditionalGeneration
mname = "fsmt-wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

pair = ["de", "en"]
input = "Maschinelles Lernen ist großartig, oder?

input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # Machine learning is great, isn't it?

```

#### Limitations and bias

- The original (and this ported model) doesn't seem to handle well inputs with repeated sub-phrases, [content gets truncated](https://discuss.huggingface.co/t/issues-with-translating-inputs-containing-repeated-phrases/981)

## Training data

Pretrained weights were left identical to the original model released by fairseq. For more details, please, see the [paper](https://arxiv.org/abs/1907.06616)

## Eval results

Fairseq reported score is [42.3](http://matrix.statmt.org/matrix/output/1902?run_id=6750)

The porting of this model is still in progress, but so far we have the following BLEU score: 39.4278

The score was calculated using this code:

```python
git clone https://github.com/huggingface/transformers
cd transformers
cd examples/seq2seq
export PAIR=de-en
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=8
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target
echo $PAIR
PYTHONPATH="../../src" python run_eval.py stas/fsmt-wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation
```

## TODO

- port model ensemble (fairseq uses 4 model checkpoints)

