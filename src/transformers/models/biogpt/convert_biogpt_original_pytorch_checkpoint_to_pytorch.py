# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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


import argparse
import json
import os
import re
import shutil

import torch

from transformers import BioGptConfig, BioGptForCausalLM
from transformers.models.biogpt.tokenization_biogpt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging


logging.set_verbosity_warning()

json_indent = 2


# modified from https://github.com/facebookresearch/fairseq/blob/dd74992d0d143155998e9ed4076826bcea80fb06/fairseq/data/dictionary.py#L18
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def _load_meta(self, lines):
        return 0

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please rebuild the dataset".format(f))
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")


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


def convert_biogpt_checkpoint_to_pytorch(biogpt_checkpoint_path, pytorch_dump_folder_path):
    # prep
    if not os.path.exists(biogpt_checkpoint_path):
        raise ValueError(f"path {biogpt_checkpoint_path} does not exist!")
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    # handle various types of models

    checkpoint_file = os.path.join(biogpt_checkpoint_path, "checkpoint.pt")
    if not os.path.isfile(checkpoint_file):
        raise ValueError(f"path to the file {checkpoint_file} does not exist!")
    chkpt = torch.load(checkpoint_file, map_location="cpu")

    args = chkpt["cfg"]["model"]

    # dicts
    dict_file = os.path.join(biogpt_checkpoint_path, "dict.txt")
    if not os.path.isfile(dict_file):
        raise ValueError(f"path to the file {dict_file} does not exist!")
    src_dict = Dictionary.load(dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["vocab_file"])
    print(f"Generating {src_vocab_file} of {src_vocab_size} records")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    # merges_file (bpecodes)
    bpecodes_file = os.path.join(biogpt_checkpoint_path, "bpecodes")
    if not os.path.isfile(bpecodes_file):
        raise ValueError(f"path to the file {bpecodes_file} does not exist!")

    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    shutil.copyfile(bpecodes_file, merges_file)

    # model config
    biogpt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    model_conf = {
        "activation_dropout": args["activation_dropout"],
        "architectures": ["BioGptForCausalLM"],
        "attention_probs_dropout_prob": args["attention_dropout"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": args["activation_fn"],
        "hidden_dropout_prob": args["dropout"],
        "hidden_size": args["decoder_embed_dim"],
        "initializer_range": 0.02,
        "intermediate_size": args["decoder_ffn_embed_dim"],
        "layer_norm_eps": 1e-12,
        "layerdrop": args["decoder_layerdrop"],
        "max_position_embeddings": args["max_target_positions"],
        "model_type": "biogpt",
        "num_attention_heads": args["decoder_attention_heads"],
        "num_hidden_layers": args["decoder_layers"],
        "pad_token_id": 1,
        "scale_embedding": not args["no_scale_embedding"],
        "tie_word_embeddings": args["share_decoder_input_output_embed"],
        "vocab_size": src_vocab_size,
    }

    # good hparam defaults to start with

    print(f"Generating {biogpt_model_config_file}")
    with open(biogpt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))

    # tokenizer config
    biogpt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path, TOKENIZER_CONFIG_FILE)

    tokenizer_conf = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1024,
        "pad_token": "<pad>",
        "special_tokens_map_file": None,
        "tokenizer_class": "BioGptTokenizer",
        "unk_token": "<unk>",
    }

    print(f"Generating {biogpt_tokenizer_config_file}")
    with open(biogpt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))

    # model
    model_state_dict = chkpt["model"]

    # remove unneeded keys
    ignore_keys = [
        "decoder.version",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    layer_names = list(model_state_dict.keys())
    for layer_name in layer_names:
        if layer_name.endswith("output_projection.weight"):
            model_state_dict[layer_name.replace("decoder.", "")] = model_state_dict.pop(layer_name)
        else:
            model_state_dict[layer_name.replace("decoder", "biogpt")] = model_state_dict.pop(layer_name)

    config = BioGptConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = BioGptForCausalLM(config)

    # check that it loads ok
    model_new.load_state_dict(model_state_dict)

    # save
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    print("Conversion is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--biogpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts,"
            " bpecodes, etc."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_biogpt_checkpoint_to_pytorch(args.biogpt_checkpoint_path, args.pytorch_dump_folder_path)
