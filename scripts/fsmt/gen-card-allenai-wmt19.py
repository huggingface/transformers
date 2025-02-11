#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
# ./gen-card-allenai-wmt19.py

import os
from pathlib import Path


def write_model_card(model_card_dir, src_lang, tgt_lang, model_name):

    texts = {
        "en": "Machine learning is great, isn't it?",
        "ru": "Машинное обучение - это здорово, не так ли?",
        "de": "Maschinelles Lernen ist großartig, nicht wahr?",
    }

    # BLUE scores as follows:
    # "pair": [fairseq, transformers]
    scores = {
        "wmt19-de-en-6-6-base": [0, 38.37],
        "wmt19-de-en-6-6-big": [0, 39.90],
    }
    pair = f"{src_lang}-{tgt_lang}"

    readme = f"""
---

language:
- {src_lang}
- {tgt_lang}
thumbnail:
tags:
- translation
- wmt19
- allenai
license: apache-2.0
datasets:
- wmt19
metrics:
- bleu
---

# FSMT

## Model description

This is a ported version of fairseq-based [wmt19 transformer](https://github.com/jungokasai/deep-shallow/) for {src_lang}-{tgt_lang}.

For more details, please, see [Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation](https://arxiv.org/abs/2006.10369).

2 models are available:

* [wmt19-de-en-6-6-big](https://huggingface.co/allenai/wmt19-de-en-6-6-big)
* [wmt19-de-en-6-6-base](https://huggingface.co/allenai/wmt19-de-en-6-6-base)


## Intended uses & limitations

#### How to use

```python
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "allenai/{model_name}"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "{texts[src_lang]}"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # {texts[tgt_lang]}

```

#### Limitations and bias


## Training data

Pretrained weights were left identical to the original model released by allenai. For more details, please, see the [paper](https://arxiv.org/abs/2006.10369).

## Eval results

Here are the BLEU scores:

model   |  transformers
-------|---------
{model_name}  |  {scores[model_name][1]}

The score was calculated using this code:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
export PAIR={pair}
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=8
export NUM_BEAMS=5
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target
echo $PAIR
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py allenai/{model_name} $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS
```

## Data Sources

- [training, etc.](http://www.statmt.org/wmt19/)
- [test set](http://matrix.statmt.org/test_sets/newstest2019.tgz?1556572561)


### BibTeX entry and citation info

```
@misc{{kasai2020deep,
    title={{Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation}},
    author={{Jungo Kasai and Nikolaos Pappas and Hao Peng and James Cross and Noah A. Smith}},
    year={{2020}},
    eprint={{2006.10369}},
    archivePrefix={{arXiv}},
    primaryClass={{cs.CL}}
}}
```

"""
    model_card_dir.mkdir(parents=True, exist_ok=True)
    path = os.path.join(model_card_dir, "README.md")
    print(f"Generating {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)

# make sure we are under the root of the project
repo_dir = Path(__file__).resolve().parent.parent.parent
model_cards_dir = repo_dir / "model_cards"

for model_name in ["wmt19-de-en-6-6-base", "wmt19-de-en-6-6-big"]:
    model_card_dir = model_cards_dir / "allenai" / model_name
    write_model_card(model_card_dir, src_lang="de", tgt_lang="en", model_name=model_name)
