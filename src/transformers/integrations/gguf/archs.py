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
"""Per-architecture GGUF -> transformers weight mappings.

This is the only file that needs an entry when adding an architecture. Everything is declarative:

- `WeightRenaming`s map GGUF names into the transformers namespace. They **chain** (every matching
  renaming fires, in order), so a shared skeleton plus a few per-arch leaf renames is enough.
- `WeightConverter`s then undo llama.cpp's value/layout transforms. They match on the **renamed**
  (transformers) name, and at most one fires per key.
"""

import torch

from ...core_model_loading import WeightConverter, WeightRenaming, WeightTransform
from .ops import LogNegate, PermuteInputFeatures, PermuteRows, SubtractOne, Unsqueeze


# Shared skeleton for decoder-only models: llama, mistral, qwen2/3, phi3, ... all use these names.
DENSE_DECODER_RENAMINGS = [
    WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
    WeightRenaming(r"^output_norm\.", "model.norm."),
    WeightRenaming(r"^output\.", "lm_head."),
    WeightRenaming(r"^blk\.", "model.layers."),
    WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
    WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
    WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
    WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
    WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
    WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
    WeightRenaming(r"\.attn_q_norm\.", ".self_attn.q_norm."),
    WeightRenaming(r"\.attn_k_norm\.", ".self_attn.k_norm."),
    WeightRenaming(r"\.ffn_gate\.", ".mlp.gate_proj."),
    WeightRenaming(r"\.ffn_up\.", ".mlp.up_proj."),
    WeightRenaming(r"\.ffn_down\.", ".mlp.down_proj."),
]

# Norms that llama.cpp stores as `w + 1` (zero-centred RMSNorm). Per-arch, because it depends on how
# the model defines its norms.
_QWEN35_OFFSET_NORMS = [
    "model.norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]


def _tiled_to_grouped(num_k_heads: int, heads_per_k: int, head_dim: int) -> torch.Tensor:
    """Inverse of llama.cpp's v-head reorder.

    llama.cpp stores value heads *tiled* (`v0k0 v0k1 ... v0k15 v1k0 ...`) while transformers groups
    them by key head (`k0v0 k0v1 k1v0 ...`). Indexing with this permutation converts the former to
    the latter. `head_dim=1` gives the permutation over head indices alone (for `A_log`, `dt_bias`).
    """
    total = num_k_heads * heads_per_k * head_dim
    tiled_from_grouped = torch.arange(total).reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
    return torch.argsort(tiled_from_grouped)


def _qwen35(config) -> list[WeightTransform]:
    """Qwen3.5: hybrid GatedDeltaNet + full attention.

    llama.cpp applies four transforms we undo here: zero-centred norms stored as `w + 1`,
    `ssm_a = -exp(A_log)`, `conv1d` squeezed to 2D, and the value heads reordered from grouped to
    tiled on every v-indexed tensor.
    """
    text_config = config.get_text_config()
    num_k_heads = text_config.linear_num_key_heads
    heads_per_k = text_config.linear_num_value_heads // num_k_heads
    head_v_dim = text_config.linear_value_head_dim
    # rows of the fused in_proj_qkv/conv1d before the value block: q and k, which are not reordered
    qk_rows = 2 * text_config.linear_key_head_dim * num_k_heads

    v_perm = _tiled_to_grouped(num_k_heads, heads_per_k, head_v_dim)  # over value_dim (e.g. 4096)
    head_perm = _tiled_to_grouped(num_k_heads, heads_per_k, 1)  # over value heads (e.g. 32)

    renamings = DENSE_DECODER_RENAMINGS + [
        WeightRenaming(r"\.post_attention_norm\.", ".post_attention_layernorm."),
        WeightRenaming(r"\.attn_qkv\.", ".linear_attn.in_proj_qkv."),
        WeightRenaming(r"\.attn_gate\.", ".linear_attn.in_proj_z."),
        WeightRenaming(r"\.ssm_alpha\.", ".linear_attn.in_proj_a."),
        WeightRenaming(r"\.ssm_beta\.", ".linear_attn.in_proj_b."),
        WeightRenaming(r"\.ssm_conv1d\.", ".linear_attn.conv1d."),
        WeightRenaming(r"\.ssm_norm\.", ".linear_attn.norm."),
        WeightRenaming(r"\.ssm_out\.", ".linear_attn.out_proj."),
        WeightRenaming(r"\.ssm_a$", ".linear_attn.A_log"),
        WeightRenaming(r"\.ssm_dt\.bias$", ".linear_attn.dt_bias"),
    ]

    converters = [
        # norms stored as w + 1
        *(
            WeightConverter(source_patterns=name, target_patterns=name, operations=[SubtractOne()])
            for name in _QWEN35_OFFSET_NORMS
        ),
        # ssm_a = -exp(A_log), and A_log is indexed by value head
        WeightConverter(
            source_patterns="linear_attn.A_log",
            target_patterns="linear_attn.A_log",
            operations=[LogNegate(), PermuteRows(head_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.dt_bias",
            target_patterns="linear_attn.dt_bias",
            operations=[PermuteRows(head_perm)],
        ),
        # value-head reorder: rows for everything that *produces* the value axis...
        *(
            WeightConverter(
                source_patterns=f"linear_attn.{name}.weight",
                target_patterns=f"linear_attn.{name}.weight",
                operations=[PermuteRows(head_perm)],
            )
            for name in ("in_proj_a", "in_proj_b")
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_z.weight",
            target_patterns="linear_attn.in_proj_z.weight",
            operations=[PermuteRows(v_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_qkv.weight",
            target_patterns="linear_attn.in_proj_qkv.weight",
            operations=[PermuteRows(v_perm, offset=qk_rows)],
        ),
        WeightConverter(
            source_patterns="linear_attn.conv1d.weight",
            target_patterns="linear_attn.conv1d.weight",
            operations=[PermuteRows(v_perm, offset=qk_rows), Unsqueeze(1)],
        ),
        # ...and columns for the one that *consumes* it
        WeightConverter(
            source_patterns="linear_attn.out_proj.weight",
            target_patterns="linear_attn.out_proj.weight",
            operations=[PermuteInputFeatures(v_perm)],
        ),
    ]
    return renamings + converters


# gguf `general.architecture` -> builder taking the model config
GGUF_ARCHS = {
    "qwen35": _qwen35,
}


def get_gguf_conversion_mapping(gguf_arch: str, config) -> list[WeightTransform]:
    """Weight transforms turning a GGUF checkpoint of `gguf_arch` into transformers weights."""
    if gguf_arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {gguf_arch!r} is not supported yet. Supported: {sorted(GGUF_ARCHS)}.")
    return GGUF_ARCHS[gguf_arch](config)
