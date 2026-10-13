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
"""Convert the joint head of Cloudflare/clef or Cloudflare/clef-flash (`joint_head.safetensors`) to a `ClefDecisionHead`.

python src/transformers/models/clef/convert_clef_to_hf.py --repo_id Cloudflare/clef-flash --output_dir decision_head
"""

import argparse
import json
import re

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import ClefConfig, ClefDecisionHead


# nn.TransformerDecoderLayer and the evidence routing layers -> MBart decoder layers, nn.Sequential -> named Linears
KEY_MAPPING = {
    r"\.norm1\.": ".self_attn_layer_norm.",
    r"\.(norm2|query_norm)\.": ".encoder_attn_layer_norm.",
    r"\.(norm3|feedforward_norm)\.": ".final_layer_norm.",
    r"\.(multihead_attn|attention)\.": ".encoder_attn.",
    r"\.(linear1|feedforward\.0)\.": ".fc1.",
    r"\.(linear2|feedforward\.3)\.": ".fc2.",
    r"^residual_scorer\.0\.": "residual_scorer.layers.0.",
    r"^residual_scorer\.3\.": "residual_scorer.layers.1.",
}


def convert_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        for pattern, replacement in KEY_MAPPING.items():
            key = re.sub(pattern, replacement, key)
        if ".in_proj_" in key:  # nn.MultiheadAttention packs the query, key and value projections
            prefix, suffix = key.split(".in_proj_")
            for name, chunk in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3)):
                converted[f"{prefix}.{name}.{suffix}"] = chunk.contiguous()
        else:
            converted[key] = value
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="Cloudflare/clef-flash")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(hf_hub_download(args.repo_id, "joint_head_config.json", revision=args.revision), encoding="utf-8") as f:
        config = ClefConfig(**json.load(f))
    state_dict = load_file(hf_hub_download(args.repo_id, "joint_head.safetensors", revision=args.revision))
    head = ClefDecisionHead(config).to(next(iter(state_dict.values())).dtype)
    head.load_state_dict(convert_state_dict(state_dict), strict=True)
    head.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
