# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Convert a CLM head checkpoint and its encoder to the Transformers format."""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoModel, AutoTokenizer, ClmConfig, ClmForDecision


@torch.no_grad()
def convert_checkpoint(
    checkpoint: str,
    output_dir: str,
    encoder: str | None = None,
    revision: str | None = None,
    encoder_revision: str | None = None,
):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "CLM_v0.1-8B.pt"
    elif not path.is_file():
        path = Path(hf_hub_download(checkpoint, "CLM_v0.1-8B.pt", revision=revision))
    weights = torch.load(path, map_location="cpu", weights_only=True)
    settings = weights["cfg"]
    encoder = encoder or settings["model"]
    backbone = AutoModel.from_pretrained(encoder, revision=encoder_revision, dtype="auto")
    config = ClmConfig(
        text_config=backbone.config,
        hidden_size=settings["width"],
        num_hidden_layers=settings["depth"],
        projection_dim=weights.get("projection_dim", settings.get("projection_dim", 512)),
        hidden_act=settings.get("activation", "gelu"),
        layer_norm=settings.get("layernorm", False),
        residual=settings.get("residual", False),
    )
    model = ClmForDecision(config, model=backbone)
    names = {"inp": "input_projection", "hidden": "layers", "norms": "layer_norms", "out": "output_projection"}
    for head_name in ("state_head", "action_head"):
        state = {}
        for key, value in weights[head_name].items():
            module, parameter = key.split(".", 1)
            state[f"{names[module]}.{parameter}"] = value
        model.get_submodule(head_name).load_state_dict(state)
    model.logit_scale.copy_(weights["logit_scale"])
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(encoder, revision=encoder_revision).save_pretrained(output_dir)
    model.decision_config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Contrastive-LM/CLM-v0.1-8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder")
    parser.add_argument("--revision")
    parser.add_argument("--encoder_revision")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir, args.encoder, args.revision, args.encoder_revision)
