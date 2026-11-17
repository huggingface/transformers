# Copyright 2026 The KLM authors and The HuggingFace Team. All rights reserved.
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
"""Tensor-only conversion from original KLM checkpoints to the native layout."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from .configuration_klm import KlmConfig


# Applied in order to the name of every original weight
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^model\.embed\.": "model.embed_tokens.",
    r"^model\.memory\.(bigram|trigram)\.": r"model.embed_ngrams.\1_embeddings.",
    r"^model\.memory\.": "model.embed_ngrams.",
    r"^model\.blocks\.": "model.layers.",
    r"\.mixer_norm\.": ".input_layernorm.",
    r"\.ffn_norm\.": ".post_attention_layernorm.",
    r"\.ffn\.(gate|up|down)\.": r".mlp.\1_proj.",
    r"\.(f|g)_proj\.0\.": r".\1_a_proj.",
    r"\.(f|g)_proj\.1\.": r".\1_b_proj.",
}


def convert_state_dict(original_state_dict: dict[str, torch.Tensor], config: KlmConfig) -> dict[str, torch.Tensor]:
    """Convert without mutating the caller's mapping; absent parameters stay absent.

    All three original convolution tensors are required to construct a fused convolution.
    """
    state_dict = {}
    convolutions = {}
    for original_key, tensor in original_state_dict.items():
        key = original_key
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            key = re.sub(pattern, replacement, key)
        if match := re.match(r"model\.layers\.(\d+)\.mixer\.(.+)", key):
            layer_idx, name = int(match.group(1)), match.group(2)
            if layer_idx >= len(config.layer_types):
                raise ValueError(f"Original checkpoint layer {layer_idx} is outside the configured layer layout.")
            if match := re.fullmatch(r"(q|k|v)_conv1d\.weight", name):
                # Gathered below: the three depthwise convolutions become one
                convolutions[layer_idx, match.group(1)] = tensor
                continue
            attention = "self_attn" if config.layer_types[layer_idx] == "full_attention" else "linear_attn"
            key = f"model.layers.{layer_idx}.{attention}.{name}"
        state_dict[key] = tensor

    for layer_idx, layer_type in enumerate(config.layer_types):
        present = [(layer_idx, name) in convolutions for name in "qkv"]
        if any(present):
            if layer_type != "linear_attention" or not all(present):
                raise ValueError(
                    f"Original convolution at layer {layer_idx} needs q, k and v tensors on a linear attention layer."
                )
            state_dict[f"model.layers.{layer_idx}.linear_attn.conv1d.weight"] = torch.cat(
                [convolutions[layer_idx, name] for name in "qkv"], dim=0
            )

    # The original keeps the parameters of all block residuals in two matrices, with one row for each of them: the
    # attention and the MLP of every layer in turn, then the final readout
    names = [
        f"model.layers.{layer_idx}.{name}"
        for layer_idx in range(config.num_hidden_layers)
        for name in ("input_residual", "post_attention_residual")
    ]
    names.append("model.output_residual")
    for original, suffix in (("model.attn_res_query", "query"), ("model.attn_res_norm", "weight")):
        if original not in state_dict:
            continue
        rows = state_dict.pop(original)
        if rows.shape != (len(names), config.hidden_size):
            raise ValueError(
                f"Original residual bank {original} has shape {tuple(rows.shape)}, expected {(len(names), config.hidden_size)}."
            )
        for row, name in enumerate(names):
            state_dict[f"{name}.{suffix}"] = rows[row].clone()

    for name in ("bigram_keys", "trigram_keys"):
        key = f"model.embed_ngrams.{name}"
        if key in state_dict:
            state_dict[key] = state_dict[key].long()
    return state_dict
