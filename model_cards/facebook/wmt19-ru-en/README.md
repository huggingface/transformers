---
language: 
- ru
- en
tags:
- translation
- wmt19
- facebook
license: apache-2.0
datasets:
- wmt19
metrics:
- bleu
thumbnail: https://huggingface.co/front/thumbnails/facebook.png
---

# FSMT

## Model description

This is a ported version of [fairseq wmt19 transformer](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md) for ru-en.

For more details, please see, [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616).

The abbreviation FSMT stands for FairSeqMachineTranslation

All four models are available:

* [wmt19-en-ru](https://huggingface.co/facebook/wmt19-en-ru)
* [wmt19-ru-en](https://huggingface.co/facebook/wmt19-ru-en)
* [wmt19-en-de](https://huggingface.co/facebook/wmt19-en-de)
* [wmt19-de-en](https://huggingface.co/facebook/wmt19-de-en)

## Intended uses & limitations

#### How to use

```python
from transformers.tokenization_fsmt import FSMTTokenizer
from transformers.modeling_fsmt import FSMTForConditionalGeneration
mname = "facebook/wmt19-ru-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "Машинное обучение - это здорово, не так ли?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # Machine learning is great, isn't it?

```

#### Limitations and bias

- The original (and this ported model) doesn't seem to handle well inputs with repeated sub-phrases, [content gets truncated](https://discuss.huggingface.co/t/issues-with-translating-inputs-containing-repeated-phrases/981)

## Training data

Pretrained weights were left identical to the original model released by fairseq. For more details, please, see the [paper](https://arxiv.org/abs/1907.06616).

## Eval results

pair   | fairseq | transformers
-------|---------|----------
ru-en  | [41.3](http://matrix.statmt.org/matrix/output/1907?run_id=6937) | 39.20

The score is slightly below the score reported by `fairseq`, since `transformers`` currently doesn't support:
- model ensemble, therefore the best performing checkpoint was ported (``model4.pt``).
- re-ranking

The score was calculated using this code:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
export PAIR=ru-en
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=8
export NUM_BEAMS=15
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target
echo $PAIR
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS
```
note: fairseq reports using a beam of 50, so you should get a slightly higher score if re-run with `--num_beams 50`.

## Data Sources

- [training, etc.](http://www.statmt.org/wmt19/)
- [test set](http://matrix.statmt.org/test_sets/newstest2019.tgz?1556572561)


### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020},
  title={Facebook FAIR's WMT19 News Translation Task Submission},
  author={Ng, Nathan and Yee, Kyra and Baevski, Alexei and Ott, Myle and Auli, Michael and Edunov, Sergey},
  booktitle={Proc. of WMT},
}
```


## TODO

- port model ensemble (fairseq uses 4 model checkpoints)

