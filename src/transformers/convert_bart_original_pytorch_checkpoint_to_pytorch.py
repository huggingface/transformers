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
from pathlib import Path

import fairseq
import torch
from packaging import version

from transformers import BartConfig, BartForSequenceClassification, BartModel, BartTokenizer


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"

rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]
IGNORE_KEYS = ["encoder.version", "decoder.version", "model.encoder.version", "model.decoder.version"]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    b2 = torch.hub.load("pytorch/fairseq", checkpoint_path)
    b2.eval()  # disable dropout
    b2.model.upgrade_state_dict(b2.model.state_dict())
    config = BartConfig()
    tokens = b2.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained("bart-large").encode(SAMPLE_TEXT).unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    # assert their_output.size() == (1, 11, 1024)

    if checkpoint_path == "bart.large":
        state_dict = b2.model.state_dict()
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        model = BartModel(config)
        their_output = b2.extract_features(tokens)

    else:  # MNLI Case
        state_dict = b2.state_dict()
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        state_dict.pop("_float_tensor", None)
        model = BartForSequenceClassification(config)
        their_output = b2.predict("mnli", tokens, return_logits=True)
    for k in IGNORE_KEYS:
        state_dict.pop(k, None)
    model.load_state_dict(state_dict)
    model.eval()
    our_outputs = model.forward(tokens)[0]

    assert their_output.shape == our_outputs.shape
    assert (their_output == our_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", choices=["bart.large", "bart.large.mnli"], type=str, help="")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_bart_checkpoint(
        args.fairseq_path, args.pytorch_dump_folder_path,
    )
