# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert BART checkpoint."""


import argparse
import logging
import os
from pathlib import Path

import torch
from packaging import version

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.modeling_bart import _make_linear_from_emb


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_checkpoint_from_disk(checkpoint_path, **config_kwargs):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    mbart_config = BartConfig(vocab_size=vocab_size, **config_kwargs)
    model = BartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)
    return model


@torch.no_grad()
def convert_fairseq_rag_checkpoint_to_hf_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_checkpoint_name)

    state_dict = torch.load(checkpoint_path)
    state_dict = {s:v for s,v in state_dict['model'].items() if (s.startswith('encoder') or s.startswith('decoder'))}
    remove_ignore_keys_(state_dict)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]

    model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
    prev_state_dict = model.model.state_dict()
    for k, v in state_dict.items():
        assert k in prev_state_dict
    for k, v in prev_state_dict.items():
        assert k in state_dict
    assert set(state_dict.keys()) == set(prev_state_dict.keys())

    model.model.load_state_dict(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)

    # # Check results
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    # TODO: very hacky right now, also we dont handle the retriever params, !

    args.hf_config = 'facebook/bart-large-cnn'
    convert_fairseq_rag_checkpoint_to_hf_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
