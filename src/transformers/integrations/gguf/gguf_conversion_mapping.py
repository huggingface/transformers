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

`WeightRenaming`s map GGUF names into the transformers namespace and chain; `WeightConverter`s then
undo llama.cpp's value/layout transforms, matching on the renamed key, at most one per key.
"""

import torch

from ...core_model_loading import (
    Concatenate,
    ConversionOps,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from .dequant import GGML_BLOCK
from .kernels import dequantize_blocks


# Shared skeleton for decoder-only models. Norms are absent: whether llama.cpp offsets them by one is
# per-architecture, so each arch declares its own.
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


def _qwen35(config) -> list[WeightTransform]:
    """Qwen3.5: hybrid GatedDeltaNet linear attention + full attention every fourth layer.

    llama.cpp's converter differs in five ways, each undone below:

    1. Names: `blk.N.*` with ggml leaf names.
    2. Zero-centred norms stored as `w + 1`, except `ssm_norm` (`SubtractOne`, in fp32).
    3. `ssm_a` holds `-exp(A_log)` (`LogNegate`).
    4. `conv1d` is squeezed to 2D (`Unsqueeze`).
    5. Value heads are tiled rather than grouped, on every v-indexed tensor: `PermuteRows` where the
       tensor produces the value dimension, `PermuteInputFeatures` for the one that consumes it.

    Only (5)'s column permute crosses quantization blocks, so `out_proj` is the one tensor that has
    to be dequantized; everything else stays packed or is F32 in the file.
    """
    text_config = config.get_text_config()
    num_key_heads = text_config.linear_num_key_heads
    value_heads_per_key_head = text_config.linear_num_value_heads // num_key_heads
    value_head_dim = text_config.linear_value_head_dim

    # `in_proj_qkv` and `conv1d` are fused q/k/v; only the value block is reordered.
    query_key_rows = 2 * text_config.linear_key_head_dim * num_key_heads

    # The same reorder over the flattened value dimension, and over head indices alone.
    heads = (num_key_heads, value_heads_per_key_head)
    per_value = TiledToGroupedRows(*heads, value_head_dim)
    per_head = TiledToGroupedRows(*heads)

    # llama.cpp writes the multi-token-prediction block as `blk.{num_hidden_layers}.*`. Transformers
    # has no MTP module and drops `^mtp.*` by name, so give it that prefix before the blanket `blk.`
    # rule turns it into a layer the model does not have.
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

    # (2) norms stored as `w + 1`: rename the leaf and un-offset in one place.
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


def _qwen35moe(config) -> list[WeightTransform]:
    """Everything outside the FFN is `_qwen35`'s: llama.cpp converts both through the same base class."""
    routed = [
        WeightRenaming(r"\.ffn_gate_inp\.", ".mlp.gate."),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        WeightRenaming(r"\.ffn_gate_inp_shexp\.", ".mlp.shared_expert_gate."),
        WeightRenaming(r"\.ffn_gate_shexp\.", ".mlp.shared_expert.gate_proj."),
        WeightRenaming(r"\.ffn_up_shexp\.", ".mlp.shared_expert.up_proj."),
        WeightRenaming(r"\.ffn_down_shexp\.", ".mlp.shared_expert.down_proj."),
    ]
    # `gate_up_proj` is chunked in two on dim 1, so gate comes first and up second.
    fuse_gate_up = WeightConverter(
        source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
        target_patterns=".mlp.experts.gate_up_proj",
        operations=[ConcatenateRows(dim=1)],
    )
    restore_gate = WeightConverter(
        source_patterns="mlp.shared_expert_gate.weight",
        target_patterns="mlp.shared_expert_gate.weight",
        operations=[Unsqueeze(0, expected_ndim=2)],
    )
    return _qwen35(config) + routed + [fuse_gate_up, restore_gate]


# gguf `general.architecture` -> builder taking the model config
GGUF_ARCHS = {
    "qwen35": _qwen35,
    "qwen35moe": _qwen35moe,
}


class SubtractOne(ConversionOps):
    """Subtract `offset` from the tensor, in float32.

    Example: `[1.02, 0.98] -> [0.02, -0.02]`
    """

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: (tensor.float() - self.offset).to(tensor.dtype)}


class LogNegate(ConversionOps):
    """Take the log of the negated tensor, `log(-x)`.

    Example: `[-2.0, -1.0] -> [0.69, 0.0]`
    """

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: torch.log(-tensor.float()).to(tensor.dtype)}


class Unsqueeze(ConversionOps):
    """Add a size-1 dim at `dim`, unless the tensor already has `expected_ndim` dims.

    Pass `expected_ndim` when a file may store the tensor either way, so that both load through the
    same mapping. Safe on packed blocks: a new axis does not move any bytes.

    Example: `[[1, 2, 3]] -> [[[1, 2, 3]]]`
    """

    supports_packed = True

    def __init__(self, dim: int, expected_ndim: int | None = None):
        self.dim = dim
        self.expected_ndim = expected_ndim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        if self.expected_ndim is not None and tensor.dim() >= self.expected_ndim:
            return {target_patterns[0]: tensor}
        return {target_patterns[0]: tensor.unsqueeze(self.dim)}


class PermuteRows(ConversionOps):
    """Reorder rows (dim 0), optionally only those from `offset` on.

    Example: `[[1, 2], [3, 4], [5, 6]] -> [[5, 6], [1, 2], [3, 4]]` for `[2, 0, 1]`
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
    """Reorder columns (dim 1). Columns cross blocks, so a packed tensor is left as stored and `GgufLinear`
    permutes its input instead.

    Example: `[[1, 2, 3]] -> [[3, 1, 2]]` for `[2, 0, 1]`
    """

    supports_packed = True

    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    @property
    def input_permutation(self) -> torch.Tensor:
        """Reordering for the input of the module holding this weight, when it stays packed.

        The inverse of the column permutation, not the permutation itself.
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
    """Reorder rows from tiled to grouped by key head, for a tensor that *produces* the head axis.

    Tiled is `v0k0 v0k1 v1k0 v1k1`, grouped is `v0k0 v1k0 v0k1 v1k1`. `head_dim` defaults to 1, for
    tensors with one row per head rather than a block of rows.

    Example: `[a, b, c, d] -> [a, c, b, d]` for 2 key heads and 2 heads each
    """

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1, offset: int = 0):
        total = num_k_heads * heads_per_k * head_dim
        # On the CPU explicitly: a mapping may be built under a meta default device, and a meta index
        # tensor silently permutes nothing.
        indices = torch.arange(total, device="cpu")
        tiled_from_grouped = indices.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
        permutation = torch.argsort(tiled_from_grouped)
        super().__init__(permutation, offset=offset)


class TiledToGroupedInputs(PermuteInputFeatures):
    """The same reorder as `TiledToGroupedRows`, on columns, for a tensor that *consumes* the head axis.

    Example: `[[a, b, c, d]] -> [[a, c, b, d]]` for 2 key heads and 2 heads each
    """

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1):
        total = num_k_heads * heads_per_k * head_dim
        # On the CPU explicitly: a mapping may be built under a meta default device, and a meta index
        # tensor silently permutes nothing.
        indices = torch.arange(total, device="cpu")
        tiled_from_grouped = indices.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
        permutation = torch.argsort(tiled_from_grouped)
        super().__init__(permutation)


class Cast(ConversionOps):
    """Cast to the model's dtype, last in the chain so that the transforms before it keep their precision. Blocks
    pass through untouched.

    Example: `[0.02] f32 -> [0.02] bf16`
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


class ConcatenateRows(Concatenate):
    """`Concatenate` on a row axis, which packed blocks survive. A block never spans two rows; joining on the last
    axis would cut through them.

    Example: `[[1, 2]] + [[3, 4]] -> [[1, 2], [3, 4]]`
    """

    supports_packed = True


class Dequantize(ConversionOps):
    """Unpack GGUF blocks into values. First in its chain, since later transforms need dense values, and the ggml
    type is looked up per parameter.

    Example: `[[210, 17, ...]] u8 -> [[0.31, -1.20, ...]] f32`
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
        block_elements, block_bytes = GGML_BLOCK[ggml_type]
        values = {}
        for key, tensors in input_dict.items():
            blocks = tensors[0] if isinstance(tensors, list) else tensors
            if blocks.dtype != torch.uint8:
                raise ValueError(f"{key} is {blocks.dtype}, not torch.uint8")
            cols = blocks.shape[-1] // block_bytes * block_elements
            flat = blocks.reshape(-1, blocks.shape[-1])
            unpacked = dequantize_blocks(flat, ggml_type, flat.shape[0], cols, self.dtype)
            values[key] = unpacked.reshape(*blocks.shape[:-1], cols)
        if len(values) == 1:
            return {full_layer_name: next(iter(values.values()))}
        return values


def _single_tensor(input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """These ops are all one-to-one; unwrap the single (possibly listed) tensor."""
    if len(input_dict) != 1:
        raise ValueError(f"expected a single source tensor, got {list(input_dict)}")
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors
