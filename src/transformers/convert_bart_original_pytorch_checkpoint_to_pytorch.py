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

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn"]

if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]
IGNORE_KEYS = ["encoder.version", "decoder.version", "model.encoder.version", "model.decoder.version", "_float_tensor"]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    bart = torch.hub.load("pytorch/fairseq", checkpoint_path)
    bart.eval()  # disable dropout
    bart.model.upgrade_state_dict(bart.model.state_dict())
    hf_model_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_model_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_model_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    if checkpoint_path in ["bart.large", "bart.large.cnn"]:
        state_dict = bart.model.state_dict()
        for k in IGNORE_KEYS:
            state_dict.pop(k, None)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        model = BartModel(config)
        their_output = bart.extract_features(tokens)
    else:  # MNLI Case
        state_dict = bart.state_dict()
        for k in IGNORE_KEYS:
            state_dict.pop(k, None)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config)
        their_output = bart.predict("mnli", tokens, return_logits=True)

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    # Check results

    if checkpoint_path == "bart.large.cnn":
        model = BartForConditionalGeneration(config, base_model=model)
        assert "lm_head.weight" in model.state_dict()
        assert model.lm_head.out_features == config.max_position_embeddings
        model.eval()
        our_outputs = model.model(tokens)[0]
    else:
        our_outputs = model(tokens)[0]
    assert their_output.shape == our_outputs.shape
    assert (their_output == our_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", choices=FAIRSEQ_MODELS, type=str, help="")

    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_bart_checkpoint(
        args.fairseq_path, args.pytorch_dump_folder_path,
    )
