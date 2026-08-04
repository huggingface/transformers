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
"""GGUF -> transformers weight conversion mappings, one entry per architecture.

Mirrors the model-side `transformers.conversion_mapping`: a declarative description of how a
checkpoint's tensors map onto a model's parameters. This is the only file that needs an entry when
adding an architecture. Everything is declarative:

- `WeightRenaming`s map GGUF names into the transformers namespace. They **chain** (every matching
  renaming fires, in order), so a shared skeleton plus a few per-arch leaf renames is enough.
- `WeightConverter`s then undo llama.cpp's value/layout transforms. They match on the **renamed**
  (transformers) name, and at most one fires per key. The `ConversionOps` they are built from are at
  the bottom of this file: one per transform llama.cpp applies when it writes a file, each taking a
  dense tensor and returning a dense one, run by the loading pipeline like any other conversion.
"""

import torch

from ...core_model_loading import ConversionOps, WeightConverter, WeightRenaming, WeightTransform


# Shared skeleton for decoder-only models: llama, mistral, qwen2/3, phi3, ... all use these names.
# Norms are deliberately absent: whether llama.cpp stores them offset by one is per-architecture, so
# each arch declares its own — as a `WeightConverter` when offset, a `WeightRenaming` when not.
DENSE_DECODER_RENAMINGS = [
    WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
    WeightRenaming(r"^output\.", "lm_head."),
    WeightRenaming(r"^blk\.", "model.layers."),
    WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
    WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
    WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
    WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
    WeightRenaming(r"\.ffn_gate\.", ".mlp.gate_proj."),
    WeightRenaming(r"\.ffn_up\.", ".mlp.up_proj."),
    WeightRenaming(r"\.ffn_down\.", ".mlp.down_proj."),
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
    """Qwen3.5: hybrid GatedDeltaNet linear attention + full attention every fourth layer.

    llama.cpp's converter differs from transformers in five ways, each undone below:

    1. **Names.** `blk.N.*` with ggml leaf names (`attn_qkv`, `ssm_out`, ...).
    2. **Zero-centred norms are stored as `w + 1`.** Every norm except `ssm_norm`, which is not
       offset. Undone in fp32 (`SubtractOne`), since the file holds them as F32 and subtracting after
       a bf16 cast would lose ~1 ULP near 1.0.
    3. **`ssm_a` holds `-exp(A_log)`** rather than `A_log` itself (`LogNegate`).
    4. **`conv1d` is squeezed** from `(channels, 1, kernel)` to 2D (`Unsqueeze`).
    5. **Value heads are reordered** from transformers' grouped layout (`k0v0 k0v1 k1v0 ...`) to a
       tiled one (`v0k0 v0k1 ... v1k0 ...`), on *every* v-indexed tensor. It lands on **rows**
       (`PermuteRows`) for tensors that produce the value dimension and on **columns**
       (`PermuteInputFeatures`) for the one that consumes it.

    For a quantized one, a row is stored as blocks of 32 or 256 elements that share their scales, so a
    transform is only valid on packed data if it keeps every value in its block:

    - (2), (3), (4): F32 in the file, so never packed and never affected.
    - (5) row permutes: exact on blocks, so those tensors stay packed.
    - (5) on `out_proj`: a column permute crosses blocks, so this one tensor is always dequantized.
    """
    text_config = config.get_text_config()
    num_key_heads = text_config.linear_num_key_heads
    value_heads_per_key_head = text_config.linear_num_value_heads // num_key_heads
    value_head_dim = text_config.linear_value_head_dim

    # `in_proj_qkv` and `conv1d` are fused: a q block, a k block, then the value block. Only the
    # value block is reordered, so the permutations start after the q and k rows.
    query_key_rows = 2 * text_config.linear_key_head_dim * num_key_heads

    # The same reorder at two granularities: over the flattened value dimension for tensors with a
    # row (or column) per value element, and over head indices alone for the per-head vectors.
    value_perm = _tiled_to_grouped(num_key_heads, value_heads_per_key_head, value_head_dim)
    value_head_perm = _tiled_to_grouped(num_key_heads, value_heads_per_key_head, 1)

    renamings = DENSE_DECODER_RENAMINGS + [
        WeightRenaming(r"\.attn_qkv\.", ".linear_attn.in_proj_qkv."),
        WeightRenaming(r"\.attn_gate\.", ".linear_attn.in_proj_z."),
        WeightRenaming(r"\.ssm_alpha\.", ".linear_attn.in_proj_a."),
        WeightRenaming(r"\.ssm_beta\.", ".linear_attn.in_proj_b."),
        WeightRenaming(r"\.ssm_conv1d\.", ".linear_attn.conv1d."),
        WeightRenaming(r"\.ssm_norm\.", ".linear_attn.norm."),  # the one norm with no offset
        WeightRenaming(r"\.ssm_out\.", ".linear_attn.out_proj."),
        WeightRenaming(r"\.ssm_a$", ".linear_attn.A_log"),
        WeightRenaming(r"\.ssm_dt\.bias$", ".linear_attn.dt_bias"),
    ]

    # (2) norms stored as `w + 1`. Each entry renames the leaf and un-offsets in one place; the
    # leading `blk.` -> `model.layers.` renaming has already fired by the time these match.
    offset_norms = [
        WeightConverter(
            source_patterns=r"^output_norm\.weight",
            target_patterns="model.norm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_norm\.weight",
            target_patterns=".input_layernorm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.post_attention_norm\.weight",
            target_patterns=".post_attention_layernorm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_q_norm\.weight",
            target_patterns=".self_attn.q_norm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_k_norm\.weight",
            target_patterns=".self_attn.k_norm.weight",
            operations=[SubtractOne()],
        ),
    ]

    # (3) and (4): value-scoped scalars, and the conv1d reshape.
    per_value_head = [
        WeightConverter(
            source_patterns="linear_attn.A_log",
            target_patterns="linear_attn.A_log",
            operations=[LogNegate(), PermuteRows(value_head_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.dt_bias",
            target_patterns="linear_attn.dt_bias",
            operations=[PermuteRows(value_head_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_a.weight",
            target_patterns="linear_attn.in_proj_a.weight",
            operations=[PermuteRows(value_head_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_b.weight",
            target_patterns="linear_attn.in_proj_b.weight",
            operations=[PermuteRows(value_head_perm)],
        ),
    ]

    # (5) the value-head reorder on tensors with a row or column per value element.
    value_reorder = [
        WeightConverter(
            source_patterns="linear_attn.in_proj_z.weight",
            target_patterns="linear_attn.in_proj_z.weight",
            operations=[PermuteRows(value_perm)],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_qkv.weight",
            target_patterns="linear_attn.in_proj_qkv.weight",
            operations=[PermuteRows(value_perm, offset=query_key_rows)],
        ),
        WeightConverter(
            source_patterns="linear_attn.conv1d.weight",
            target_patterns="linear_attn.conv1d.weight",
            operations=[PermuteRows(value_perm, offset=query_key_rows), Unsqueeze(1)],
        ),
        # The only tensor that *consumes* the value dimension, so the reorder is on its columns —
        # which cannot be done on blocks, so this weight is always dequantized at load.
        WeightConverter(
            source_patterns="linear_attn.out_proj.weight",
            target_patterns="linear_attn.out_proj.weight",
            operations=[PermuteInputFeatures(value_perm)],
        ),
    ]

    return renamings + offset_norms + per_value_head + value_reorder


# gguf `general.architecture` -> builder taking the model config
GGUF_ARCHS = {
    "qwen35": _qwen35,
}


class SubtractOne(ConversionOps):
    """Undo llama.cpp storing zero-centred RMSNorm weights as `w + 1`.

    Runs in fp32 on purpose. The file holds these norms as F32 and llama.cpp computed `w + 1` in
    fp32, so subtracting in fp32 recovers `w` exactly; the cast to the model dtype then reproduces
    the original value. Subtracting after a bf16 cast loses ~1 ULP near 1.0.
    """

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: (tensor.float() - self.offset).to(tensor.dtype)}

    @property
    def reverse_op(self) -> ConversionOps:
        return SubtractOne(offset=-self.offset)


class LogNegate(ConversionOps):
    """`A_log = log(-a)`, undoing llama.cpp storing `ssm_a = -exp(A_log)`.

    Not bit-exact: `exp` then `log` is a 1-2 ULP round trip.
    """

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: torch.log(-tensor.float()).to(tensor.dtype)}


class Unsqueeze(ConversionOps):
    """Add a size-1 dim, undoing llama.cpp squeezing `conv1d` from `(C, 1, K)` to `(C, K)`."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.unsqueeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Squeeze(dim=self.dim)


class Squeeze(ConversionOps):
    """Reverse of `Unsqueeze`."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.squeeze(self.dim)}


class PermuteRows(ConversionOps):
    """Reorder rows (dim 0), optionally only those from `offset` onwards.

    llama.cpp can store head-indexed tensors in a different head order than transformers. Where
    that axis is the tensor's *output* axis, undoing it is a row permutation.

    `offset` covers tensors whose leading rows must stay put: e.g. Qwen3.5's fused
    `in_proj_qkv`, where only the v-block is reordered.

    Can run on packed GGUF blocks: a block never spans two rows, so reordering rows moves whole
    groups of blocks along with their scales.
    """

    supports_packed = True

    def __init__(self, permutation: torch.Tensor, offset: int = 0):
        self.permutation = permutation
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        perm = self.permutation.to(tensor.device)
        if self.offset:
            head, tail = tensor[: self.offset], tensor[self.offset :]
            tensor = torch.cat([head, tail[perm]], dim=0)
        else:
            tensor = tensor[perm]
        return {target_patterns[0]: tensor.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteRows(torch.argsort(self.permutation), offset=self.offset)


class PermuteInputFeatures(ConversionOps):
    """Reorder columns (dim 1), for a tensor that *consumes* an axis llama.cpp reordered.

    Same logical reordering as `PermuteRows`, but it lands on `in_features` because this tensor is
    on the consuming side (Qwen3.5's `linear_attn.out_proj`).

    Cannot be applied to packed GGUF blocks: it moves values between blocks, which would mean
    re-picking scales and re-rounding, i.e. a requantization. Tensors needing it are therefore always
    dequantized at load.
    """

    supports_packed = False

    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        perm = self.permutation.to(tensor.device)
        return {target_patterns[0]: tensor[:, perm].contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteInputFeatures(torch.argsort(self.permutation))


def _single_tensor(input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """These ops are all one-to-one; unwrap the single (possibly listed) tensor."""
    if len(input_dict) != 1:
        raise ValueError(f"expected a single source tensor, got {list(input_dict)}")
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors
