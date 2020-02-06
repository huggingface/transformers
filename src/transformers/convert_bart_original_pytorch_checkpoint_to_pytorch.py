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
"""Convert RoBERTa checkpoint."""


import argparse
import logging
from pathlib import Path

import fairseq
import torch
from fairseq.models.bart import BARTModel as FairseqBartModel
from packaging import version

from transformers.configuration_bart import BARTConfig
from transformers.modeling_bart import BARTModel
from transformers.modeling_bert import (
    BertConfig,
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak transformer's weights to our BERT structure.
    """
    b2 = FairseqBartModel.from_pretrained(checkpoint_path)
    b2.eval()  # disable dropout
    b2.model.upgrade_state_dict(b2.model.state_dict())
    upgraded = b2.model.state_dict()
    upgraded["shared.weight"] = upgraded["decoder.embed_tokens.weight"]
    config = BARTConfig()
    model = BARTModel(config)
    model.load_state_dict(upgraded)
    tokens = b2.encode(SAMPLE_TEXT)
    # TODO(SS): test BartTokenizer Equality
    model.eval()
    fairseq_features = b2.extract_features(tokens)
    all_outputs = model.forward(tokens)
    new_outputs = all_outputs[0]

    assert fairseq_features.shape == new_outputs.shape
    assert (fairseq_features == new_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", default=None, type=str, help="")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_bart_checkpoint(
        args.fairseq_path, args.pytorch_dump_folder_path,
    )
