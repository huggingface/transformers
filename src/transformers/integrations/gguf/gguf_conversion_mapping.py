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
from .dequant import GGML_BLOCK
from .kernels import dequantize_blocks


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


def tiled_to_grouped(num_k_heads: int, heads_per_k: int, head_dim: int = 1) -> torch.Tensor:
    """Inverse of llama.cpp's head reorder, as a permutation.

    llama.cpp stores head-indexed axes *tiled* (`v0k0 v0k1 ... v0k15 v1k0 ...`) while transformers
    groups them by key head (`k0v0 k0v1 k1v0 ...`). Indexing with this permutation converts the
    former to the latter. `head_dim` defaults to 1, giving the permutation over head indices alone
    (what a per-head vector like `A_log` or `dt_bias` needs).

    A convention of llama.cpp's converter rather than of any one architecture, so every model whose
    heads it tiles reorders them with this -- see `TiledToGroupedRows`/`TiledToGroupedInputs`.
    """
    total = num_k_heads * heads_per_k * head_dim
    # On the CPU explicitly: a mapping may be built inside the model's init context, where the default
    # device is meta, and a meta index tensor silently permutes nothing once moved to the weight.
    indices = torch.arange(total, device="cpu")
    tiled_from_grouped = indices.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
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
    heads = (num_key_heads, value_heads_per_key_head)
    per_value = TiledToGroupedRows(*heads, value_head_dim)
    per_head = TiledToGroupedRows(*heads)

    # llama.cpp counts the multi-token-prediction block in `block_count` and writes it as
    # `blk.{num_hidden_layers}.*`, one past the decoder stack and under ordinary leaf names -- so
    # nothing about an individual tensor marks it as MTP. Transformers has no MTP module, and
    # `Qwen3_5ForCausalLM` already drops these by name (`_keys_to_ignore_on_load_unexpected =
    # [r"^mtp.*"]`), so hand them that prefix. Ahead of the blanket `blk.` rule below, which would
    # otherwise make them `model.layers.{N}.*` -- a layer the model does not have, which every load
    # then reports as unexpected. The leaf renamings still rewrite what follows the prefix, which is
    # harmless: every one of them is relative, so what they produce is still under `mtp.`.
    mtp_block = [WeightRenaming(rf"^blk\.{text_config.num_hidden_layers}\.", "mtp.")]

    renamings = (
        mtp_block
        + DENSE_DECODER_RENAMINGS
        + [
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
    )

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
            operations=[LogNegate(), per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.dt_bias",
            target_patterns="linear_attn.dt_bias",
            operations=[per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_a.weight",
            target_patterns="linear_attn.in_proj_a.weight",
            operations=[per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_b.weight",
            target_patterns="linear_attn.in_proj_b.weight",
            operations=[per_head],
        ),
    ]

    # (5) the value-head reorder on tensors with a row or column per value element.
    value_reorder = [
        WeightConverter(
            source_patterns="linear_attn.in_proj_z.weight",
            target_patterns="linear_attn.in_proj_z.weight",
            operations=[per_value],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_qkv.weight",
            target_patterns="linear_attn.in_proj_qkv.weight",
            operations=[TiledToGroupedRows(*heads, value_head_dim, offset=query_key_rows)],
        ),
        WeightConverter(
            source_patterns="linear_attn.conv1d.weight",
            target_patterns="linear_attn.conv1d.weight",
            operations=[TiledToGroupedRows(*heads, value_head_dim, offset=query_key_rows), Unsqueeze(1)],
        ),
        # The only tensor that *consumes* the value dimension, so the reorder is on its columns. Columns
        # cross quantization blocks, so when it stays packed the reorder is applied to its input instead.
        WeightConverter(
            source_patterns="linear_attn.out_proj.weight",
            target_patterns="linear_attn.out_proj.weight",
            operations=[TiledToGroupedInputs(*heads, value_head_dim)],
        ),
    ]

    return renamings + offset_norms + per_value_head + value_reorder


# gguf `general.architecture` -> builder taking the model config
GGUF_ARCHS = {
    "qwen35": _qwen35,
}


class SubtractOne(ConversionOps):
    """Undo llama.cpp storing zero-centred RMSNorm weights as `w + 1`."""

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: (tensor.float() - self.offset).to(tensor.dtype)}


class LogNegate(ConversionOps):
    """`A_log = log(-a)`, undoing llama.cpp storing `ssm_a = -exp(A_log)`."""

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


class PermuteInputFeatures(ConversionOps):
    """Reorder columns (dim 1), for a tensor that *consumes* an axis llama.cpp reordered.

    Same logical reordering as `PermuteRows`, but on `in_features`, because this tensor reads the axis
    the others produce (Qwen3.5's `linear_attn.out_proj`).

    Columns cross quantization blocks, so reordering them in packed bytes would mean re-picking scales
    and re-rounding -- a requantization. It does not have to happen on the weight at all:

        x @ W[:, p].T  ==  x[:, argsort(p)] @ W.T

    both sides pairing `x[j]` with `W[p[j]]`. So a packed weight is left exactly as the file stores it
    and the module gathers its *input* instead -- one row of activations, against tens of megabytes of
    weight. `GgufLinear` reads `input_permutation` for that.

    A tensor that arrives dense still has its columns permuted here, so the dequantized path is unchanged.
    """

    supports_packed = True

    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    @property
    def input_permutation(self) -> torch.Tensor:
        """The reordering to apply to the input of the module holding this weight, when it stays packed.

        The inverse of the column permutation, not the permutation itself: `x @ W[:, p].T` sums
        `x[j] * W[p[j]]`, which is `x[argsort(p)] @ W.T`. Getting it backwards is not an error, it is
        fluent nonsense.
        """
        return torch.argsort(self.permutation)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        if tensor.dtype == torch.uint8:
            # Packed: the reorder rides on the input instead, so the blocks pass through untouched.
            return {target_patterns[0]: tensor}
        perm = self.permutation.to(tensor.device)
        return {target_patterns[0]: tensor[:, perm].contiguous()}


class TiledToGroupedRows(PermuteRows):
    """`PermuteRows` with llama.cpp's head reorder, for a tensor that *produces* the head axis."""

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1, offset: int = 0):
        super().__init__(tiled_to_grouped(num_k_heads, heads_per_k, head_dim), offset=offset)


class TiledToGroupedInputs(PermuteInputFeatures):
    """`PermuteInputFeatures` with llama.cpp's head reorder, for a tensor that *consumes* it."""

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1):
        super().__init__(tiled_to_grouped(num_k_heads, heads_per_k, head_dim))


class Cast(ConversionOps):
    """Cast to the dtype the model is being loaded in, after every other transform has run.

    Last, not first, because llama.cpp stores values this path has to do arithmetic on: a zero-centred
    norm is written as `w + 1`, and rounding *that* to bf16 spends the precision available near 1.0
    (a step of 7.8e-03) on a weight that is usually much smaller, where the step would have been
    9.8e-04. Subtracting first and rounding after is what a safetensors checkpoint of the same model
    holds. `A_log` is the same story through `LogNegate`.

    The loader will not do this itself: it skips the cast for a pre-quantized checkpoint under renamed
    keys, and a GGUF renames every key.

    Blocks pass through untouched -- they are `uint8`, and a module that holds them unpacks into this
    dtype when it computes.
    """

    def __init__(self, dtype: "torch.dtype"):
        self.dtype = dtype

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        name = full_layer_name if full_layer_name is not None else target_patterns[0]
        if tensor.dtype in (torch.uint8, self.dtype):
            return {name: tensor}
        return {name: tensor.to(self.dtype)}


class Dequantize(ConversionOps):
    """Unpack GGUF blocks into values.

    Weights arrive as blocks whatever their destination: the ones with a packed module keep them, and
    these are unpacked here — by which point the loading pipeline has put the bytes on the parameter's
    own device, so the unpacking happens there rather than on the host.

    First in its chain, because every transform after it is defined on dense values — a column permute
    moves values between blocks, which on packed data would mean requantizing.

    One instance serves the whole file: llama.cpp mixes quantization types across a checkpoint, so the
    type is looked up per parameter. A parameter that is not listed keeps its blocks, which is how a
    chain shared with packed weights leaves those alone.
    """

    def __init__(self, ggml_types: dict[str, int], dtype: "torch.dtype"):
        self.ggml_types = ggml_types
        self.dtype = dtype

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        ggml_type = self.ggml_types.get(full_layer_name)
        if ggml_type is None:
            return input_dict
        blocks = _single_tensor(input_dict)
        block_elements, block_bytes = GGML_BLOCK[ggml_type]
        rows, cols = blocks.shape[0], blocks.shape[1] // block_bytes * block_elements
        return {full_layer_name: dequantize_blocks(blocks, ggml_type, rows, cols, self.dtype)}


def _single_tensor(input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """These ops are all one-to-one; unwrap the single (possibly listed) tensor."""
    if len(input_dict) != 1:
        raise ValueError(f"expected a single source tensor, got {list(input_dict)}")
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors
