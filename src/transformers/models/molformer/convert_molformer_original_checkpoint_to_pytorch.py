# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Molformer checkpoint."""


import argparse
import re

import torch

from transformers import MolformerConfig, MolformerForMaskedLM
from transformers.utils import logging


logging.set_verbosity_info()

RULES = [
    (r"tok_emb", r"molformer.embeddings.word_embeddings"),
    (
        r"blocks\.layers\.(\d+)\.attention\.inner_attention\.feature_map\.omega",
        r"molformer.encoder.layer.\1.attention.self.feature_map.weight",
    ),
    (
        r"blocks\.layers\.(\d+)\.attention\.(query|key|value)_projection",
        r"molformer.encoder.layer.\1.attention.self.\2",
    ),
    (r"blocks\.layers\.(\d+)\.attention\.out_projection", r"molformer.encoder.layer.\1.attention.output.dense"),
    (r"blocks\.layers\.(\d+)\.norm1", r"molformer.encoder.layer.\1.attention.output.LayerNorm"),
    (r"blocks\.layers\.(\d+)\.linear1", r"molformer.encoder.layer.\1.intermediate.dense"),
    (r"blocks\.layers\.(\d+)\.linear2", r"molformer.encoder.layer.\1.output.dense"),
    (r"blocks\.layers\.(\d+)\.norm2", r"molformer.encoder.layer.\1.output.LayerNorm"),
    (r"blocks\.norm", r"molformer.LayerNorm"),
    (r"lang_model\.embed", r"lm_head.transform.dense"),
    (r"lang_model\.ln_f", r"lm_head.transform.LayerNorm"),
    (r"lang_model\.head", r"lm_head.decoder"),
]
for i, (find, replace) in enumerate(RULES):
    RULES[i] = (re.compile(find), replace)


def convert_lightning_checkpoint_to_pytorch(lightning_checkpoint_path, pytorch_dump_path, config=None):
    # Initialise PyTorch model
    config = MolformerConfig(tie_word_embeddings=False) if config is None else MolformerConfig.from_pretrained(config)
    print(f"Building PyTorch model from configuration: {config}")
    model = MolformerForMaskedLM(config)

    # Load weights from lightning checkpoint
    checkpoint = torch.load(lightning_checkpoint_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, val in state_dict.items():
        for find, replace in RULES:
            if find.search(key) is not None:
                new_state_dict[find.sub(replace, key)] = val
                break
    model.load_state_dict(new_state_dict)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--lightning_checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument("--config", default=None, type=str, help="Path to config.json")
    args = parser.parse_args()
    convert_lightning_checkpoint_to_pytorch(args.lightning_checkpoint_path, args.pytorch_dump_path, config=args.config)
