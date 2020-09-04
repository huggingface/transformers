# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""

Convert fairseq transform wmt19 checkpoint.

To convert run:
assuming the fairseq data is under data/wmt19.ru-en.ensemble, data/wmt19.en-ru.ensemble, etc

export ROOT=/code/huggingface/transformers-fair-wmt
cd $ROOT
mkdir data

# get data (run once)
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz
tar -xvzf wmt19.en-de.joined-dict.ensemble.tar.gz
tar -xvzf wmt19.de-en.joined-dict.ensemble.tar.gz
tar -xvzf wmt19.en-ru.ensemble.tar.gz
tar -xvzf wmt19.ru-en.ensemble.tar.gz


# run conversions and uploads

export PAIR=ru-en
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=en-ru
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=de-en
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.joined-dict.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=en-de
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.joined-dict.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR


# upload
cd data
transformers-cli upload fsmt-wmt19-ru-en
transformers-cli upload fsmt-wmt19-en-ru
transformers-cli upload fsmt-wmt19-de-en
transformers-cli upload fsmt-wmt19-en-de
cd -

# force cache invalidation, which will now download the new models
PYTHONPATH="src" python -c 'from transformers import AutoModel; [AutoModel.from_pretrained("stas/fsmt-wmt19-"+p, use_cdn=False) for p in ["en-ru","ru-en","en-de","de-en"]]'

# happy translations

"""

import argparse
import json
import logging
import os
import re
from collections import OrderedDict
from os.path import basename, dirname

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import WEIGHTS_NAME
from transformers.configuration_fsmt import FSMTConfig
from transformers.modeling_fsmt import FSMTForConditionalGeneration, get_authorized_missing_keys
from transformers.tokenization_fsmt import VOCAB_FILES_NAMES


logging.basicConfig(level=logging.INFO)

DEBUG = 1

json_indent = 2 if DEBUG else None


def rewrite_dict_keys(d):
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up,
    # e.g.: d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    keep_keys = "<s> <pad> </s> <unk>".split()
    # restore the special tokens
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k]  # restore
    return d2


def write_model_card(model_card_dir, src_lang, tgt_lang):

    texts = {
        "en": "Machine learning is great, isn't it?",
        "ru": "Машинное обучение - это здорово, не так ли?",
        "de": "Maschinelles Lernen ist großartig, oder?",
    }

    # BLUE scores as follows:
    # "pair": [fairseq, transformers]
    scores = {
        "en-ru": ["[36.4](http://matrix.statmt.org/matrix/output/1914?run_id=6724)", "31.2695"],
        "ru-en": ["[41.3](http://matrix.statmt.org/matrix/output/1907?run_id=6937)", "38.8524"],
        "de-en": ["[42.3](http://matrix.statmt.org/matrix/output/1902?run_id=6750)", "39.4278"],
        "en-de": ["[43.1](http://matrix.statmt.org/matrix/output/1909?run_id=6862)", "41.0814"],
    }
    pair = f"{src_lang}-{tgt_lang}"

    readme = f"""
---
language: {src_lang}, {tgt_lang}
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

# FSMT

## Model description

This is a ported version of [fairseq wmt19 transformer](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md) for {src_lang}-{tgt_lang}.

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
mname = "fsmt-wmt19-{src_lang}-{tgt_lang}"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

pair = ["{src_lang}", "{tgt_lang}"]
input = "{texts[src_lang]}

input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # {texts[tgt_lang]}

```

#### Limitations and bias

- The original (and this ported model) doesn't seem to handle well inputs with repeated sub-phrases, [content gets truncated](https://discuss.huggingface.co/t/issues-with-translating-inputs-containing-repeated-phrases/981)

## Training data

Pretrained weights were left identical to the original model released by fairseq. For more details, please, see the [paper](https://arxiv.org/abs/1907.06616)

## Eval results

Fairseq reported score is { scores[pair][0] }

The porting of this model is still in progress, but so far we have the following BLEU score: { scores[pair][1] }

The score was calculated using this code:

```python
git clone https://github.com/huggingface/transformers
cd transformers
cd examples/seq2seq
export PAIR={pair}
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

"""
    os.makedirs(model_card_dir, exist_ok=True)
    path = os.path.join(model_card_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)


def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):

    # prep
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    # XXX: Need to work out the ensemble as fairseq does, for now using just one chkpt
    # checkpoint_file = 'model1.pt:model2.pt:model3.pt:model4.pt'
    checkpoint_file = "model1.pt"
    # model_name_or_path = 'transformer.wmt19.ru-en'
    data_name_or_path = "."
    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    models = cls.hub_models()
    kwargs = {"bpe": "fastbpe", "tokenizer": "moses"}

    # note: there is some magic happening here, so can't use torch.load() directly on the model file
    # see: load_state_dict() in fairseq_model.py
    chkpt = hub_utils.from_pretrained(
        fsmt_checkpoint_path, checkpoint_file, data_name_or_path, archive_map=models, **kwargs
    )

    args = dict(vars(chkpt["args"]))

    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]

    data_root = dirname(pytorch_dump_folder_path)
    model_dir = basename(pytorch_dump_folder_path)
    proj_root = dirname(dirname(dirname(os.path.realpath(__file__))))

    # dicts
    src_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{tgt_lang}.txt")

    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    pytorch_vocab_file_src = os.path.join(pytorch_dump_folder_path, f"vocab-{src_lang}.json")
    print(f"Generating {pytorch_vocab_file_src}")
    with open(pytorch_vocab_file_src, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    pytorch_vocab_file_tgt = os.path.join(pytorch_dump_folder_path, f"vocab-{tgt_lang}.json")
    print(f"Generating {pytorch_vocab_file_tgt}")
    with open(pytorch_vocab_file_tgt, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))

    # merge_file (bpecodes)
    merge_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    fairseq_merge_file = os.path.join(fsmt_checkpoint_path, "bpecodes")
    with open(fairseq_merge_file, encoding="utf-8") as fin:
        merges = fin.read()
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    print(f"Generating {merge_file}")
    with open(merge_file, "w", encoding="utf-8") as fout:
        fout.write(merges)

    # config
    fairseq_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    # XXX: need to compare with the other pre-trained models of this type and
    # only set here what's different between them - the common settings go into
    # config_fsmt
    conf = {
        "architectures": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "activation_dropout": 0.0,
        "activation_function": "relu",
        "attention_dropout": args["attention_dropout"],
        "d_model": args["decoder_embed_dim"],
        "dropout": args["dropout"],
        "init_std": 0.02,
        "max_position_embeddings": 1024,  # XXX: look up?
        "num_hidden_layers": 6,  # XXX: look up?
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "langs": [src_lang, tgt_lang],
        "encoder_attention_heads": args["encoder_attention_heads"],
        "encoder_ffn_dim": args["encoder_ffn_embed_dim"],
        "encoder_layerdrop": 0.0,
        "encoder_layers": args["encoder_layers"],
        "decoder_attention_heads": args["decoder_attention_heads"],
        "decoder_ffn_dim": args["decoder_ffn_embed_dim"],
        "decoder_layerdrop": 0.0,
        "decoder_layers": args["decoder_layers"],
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},  # not needed?
        "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},  # not needed?
        "add_bias_logits": False,
        "add_final_layer_norm": False,
        "is_encoder_decoder": True,
        "normalize_before": False,
        "normalize_embedding": False,
        "scale_embedding": True,
        "static_position_embeddings": True,
        "tie_word_embeddings": False,
    }

    print(f"Generating {fairseq_config_file}")
    with open(fairseq_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(conf, ensure_ascii=False, indent=json_indent))

    # model
    model = chkpt["models"][0]
    model_state_dict = model.state_dict()

    # rename keys to start with 'model.'
    model_state_dict = OrderedDict(("model." + k, v) for k, v in model_state_dict.items())

    # remove unneeded keys
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        "model.encoder_embed_tokens.weight",
        "model.decoder_embed_tokens.weight",
    ]
    # let's save a lot of space, by not saving unneeded keys - lots of them!
    ignore_keys.extend(get_authorized_missing_keys())
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = FSMTForConditionalGeneration(config)

    # check that it loads ok
    model_new.load_state_dict(model_state_dict, strict=False)

    # save
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    # test that it's the same
    test_state_dict = torch.load(pytorch_weights_dump_path)
    # print(test_state_dict)

    def compare_state_dicts(d1, d2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(d1.items(), d2.items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    print("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print("Models match perfectly! :)")

    compare_state_dicts(model_state_dict, test_state_dict)

    # model card
    model_card_dir = os.path.join(proj_root, "model_cards", "stas", model_dir)
    print(f"Generating model_card {src_lang}-{tgt_lang}")
    write_model_card(model_card_dir, src_lang, tgt_lang)

    print("Conversion is done!")
    print("\nLast step is to upload the files to s3")
    print(f"cd {data_root}")
    print(f"transformers-cli upload {model_dir}")
    print(f"Note: CDN caches files for up to 24h, so use `from_pretrained(mname, use_cdn=False)` to force redownload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fsmt_checkpoint_path", default=None, type=str, required=True, help="Path to the official PyTorch dump dir."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
