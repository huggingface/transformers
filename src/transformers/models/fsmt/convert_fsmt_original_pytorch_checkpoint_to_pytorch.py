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

# Note: if you intend to run this script make sure you look under scripts/fsmt/
# to locate the appropriate script to do the work correctly. There is a set of scripts to:
# - download and prepare data and run the conversion script
# - perform eval to get the best hparam into the config
# - generate model_cards - useful if you have multiple models from the same paper

import argparse
import json
import os
import re
from collections import OrderedDict
from os.path import basename, dirname

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import WEIGHTS_NAME, logging
from transformers.models.fsmt import VOCAB_FILES_NAMES, FSMTConfig, FSMTForConditionalGeneration
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE


logging.set_verbosity_warning()

json_indent = 2

# based on the results of a search on a range of `num_beams`, `length_penalty` and `early_stopping`
# values against wmt19 test data to obtain the best BLEU scores, we will use the following defaults:
#
# * `num_beams`: 5 (higher scores better, but requires more memory/is slower, can be adjusted by users)
# * `early_stopping`: `False` consistently scored better
# * `length_penalty` varied, so will assign the best one depending on the model
best_score_hparams = {
    # fairseq:
    "wmt19-ru-en": {"length_penalty": 1.1},
    "wmt19-en-ru": {"length_penalty": 1.15},
    "wmt19-en-de": {"length_penalty": 1.0},
    "wmt19-de-en": {"length_penalty": 1.1},
    # allenai:
    "wmt16-en-de-dist-12-1": {"length_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"length_penalty": 0.6},
    "wmt16-en-de-12-1": {"length_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"length_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"length_penalty": 0.6},
}

# this remaps the different models to their organization names
org_names = {}
for m in ["wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"]:
    org_names[m] = "facebook"
for m in [
    "wmt16-en-de-dist-12-1",
    "wmt16-en-de-dist-6-1",
    "wmt16-en-de-12-1",
    "wmt19-de-en-6-6-base",
    "wmt19-de-en-6-6-big",
]:
    org_names[m] = "allenai"


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


def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):

    # prep
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    # handle various types of models

    checkpoint_file = basename(fsmt_checkpoint_path)
    fsmt_folder_path = dirname(fsmt_checkpoint_path)

    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    models = cls.hub_models()
    kwargs = {"bpe": "fastbpe", "tokenizer": "moses"}
    data_name_or_path = "."
    # note: since the model dump is old, fairseq has upgraded its model some
    # time later, and it does a whole lot of rewrites and splits on the saved
    # weights, therefore we can't use torch.load() directly on the model file.
    # see: upgrade_state_dict(state_dict) in fairseq_model.py
    print(f"using checkpoint {checkpoint_file}")
    chkpt = hub_utils.from_pretrained(
        fsmt_folder_path, checkpoint_file, data_name_or_path, archive_map=models, **kwargs
    )

    args = vars(chkpt["args"]["model"])

    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]

    data_root = dirname(pytorch_dump_folder_path)
    model_dir = basename(pytorch_dump_folder_path)

    # dicts
    src_dict_file = os.path.join(fsmt_folder_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_folder_path, f"dict.{tgt_lang}.txt")

    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-src.json")
    print(f"Generating {src_vocab_file} of {src_vocab_size} of {src_lang} records")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # detect whether this is a do_lower_case situation, which can be derived by checking whether we
    # have at least one upcase letter in the source vocab
    do_lower_case = True
    for k in src_vocab.keys():
        if not k.islower():
            do_lower_case = False
            break

    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    tgt_vocab_file = os.path.join(pytorch_dump_folder_path, "vocab-tgt.json")
    print(f"Generating {tgt_vocab_file} of {tgt_vocab_size} of {tgt_lang} records")
    with open(tgt_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))

    # merges_file (bpecodes)
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    for fn in ["bpecodes", "code"]:  # older fairseq called the merges file "code"
        fsmt_merges_file = os.path.join(fsmt_folder_path, fn)
        if os.path.exists(fsmt_merges_file):
            break
    with open(fsmt_merges_file, encoding="utf-8") as fin:
        merges = fin.read()
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    print(f"Generating {merges_file}")
    with open(merges_file, "w", encoding="utf-8") as fout:
        fout.write(merges)

    # model config
    fsmt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    # validate bpe/tokenizer config, as currently it's hardcoded to moses+fastbpe -
    # may have to modify the tokenizer if a different type is used by a future model
    assert args["bpe"] == "fastbpe", f"need to extend tokenizer to support bpe={args['bpe']}"
    assert args["tokenizer"] == "moses", f"need to extend tokenizer to support bpe={args['tokenizer']}"

    model_conf = {
        "architectures": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "activation_dropout": args["activation_dropout"],
        "activation_function": "relu",
        "attention_dropout": args["attention_dropout"],
        "d_model": args["decoder_embed_dim"],
        "dropout": args["dropout"],
        "init_std": 0.02,
        "max_position_embeddings": args["max_source_positions"],
        "num_hidden_layers": args["encoder_layers"],
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "langs": [src_lang, tgt_lang],
        "encoder_attention_heads": args["encoder_attention_heads"],
        "encoder_ffn_dim": args["encoder_ffn_embed_dim"],
        "encoder_layerdrop": args["encoder_layerdrop"],
        "encoder_layers": args["encoder_layers"],
        "decoder_attention_heads": args["decoder_attention_heads"],
        "decoder_ffn_dim": args["decoder_ffn_embed_dim"],
        "decoder_layerdrop": args["decoder_layerdrop"],
        "decoder_layers": args["decoder_layers"],
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "is_encoder_decoder": True,
        "scale_embedding": not args["no_scale_embedding"],
        "tie_word_embeddings": args["share_all_embeddings"],
    }

    # good hparam defaults to start with
    model_conf["num_beams"] = 5
    model_conf["early_stopping"] = False
    if model_dir in best_score_hparams and "length_penalty" in best_score_hparams[model_dir]:
        model_conf["length_penalty"] = best_score_hparams[model_dir]["length_penalty"]
    else:
        model_conf["length_penalty"] = 1.0

    print(f"Generating {fsmt_model_config_file}")
    with open(fsmt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))

    # tokenizer config
    fsmt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)

    tokenizer_conf = {
        "langs": [src_lang, tgt_lang],
        "model_max_length": 1024,
        "do_lower_case": do_lower_case,
    }

    print(f"Generating {fsmt_tokenizer_config_file}")
    with open(fsmt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))

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
        "model.encoder.embed_positions._float_tensor",
        "model.decoder.embed_positions._float_tensor",
    ]
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

    print("Conversion is done!")
    print("\nLast step is to upload the files to s3")
    print(f"cd {data_root}")
    print(f"transformers-cli upload {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fsmt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts, bpecodes, etc.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
