# Copyright 2026 Google LLC
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

"""Quantized layers for Gemma: INT2/4/8 packed-weight Linear and Embedding,
plus SRQ (Static Range Quantization) activation rounding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_srq(x: torch.Tensor, scale: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Apply Static Range Quantization rounding and clipping (in x's dtype).

    A `scale` of 0 means the layer is uncalibrated, in which case this is a no-op. The guard uses
    `torch.where` rather than `scale.item()` so it stays on-device and `torch.compile`-friendly (an
    `.item()` would force a host-device sync and break `fullgraph=True`).
    """
    scale = scale.to(x.dtype)
    max_value = 2 ** (bits - 1) - 1
    min_value = -max_value - 1
    calibrated = scale != 0
    safe_scale = torch.where(calibrated, scale, torch.ones_like(scale))
    x_q = torch.clamp(torch.round(x / safe_scale), float(min_value), float(max_value)) * safe_scale
    return torch.where(calibrated, x_q, x)


def _unpack_int4(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack int4 values from uint8 storage. Two values per byte.

    Each byte: low nibble = first value, high nibble = second value.
    Values are stored unsigned in [0, 15] and shifted to signed [-8, 7].
    Cast to uint8 first so the right shift is logical, not arithmetic.
    """
    packed = packed.to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    interleaved = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :original_width]


def _unpack_int2(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack int2 values from uint8 storage. Four values per byte.

    Bits [1:0]/[3:2]/[5:4]/[7:6] hold values 0..3 each, shifted to signed [-2, 1].
    """
    packed = packed.to(torch.uint8)
    v0 = (packed & 0x03).to(torch.int8) - 2
    v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
    v3 = (packed >> 6).to(torch.int8) - 2
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :original_width]


class QuantizedLinear(nn.Linear):
    """Linear layer with INT2/4/8 packed weights and SRQ activation rounding."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_bits = num_bits

        # int2/int4 packed in uint8 (4 / 2 values per byte); int8 stored directly.
        # Replace the inherited fp32 weight with packed-int storage.
        if num_bits == 2:
            packed_in = (in_features + 3) // 4
            weight_storage = torch.empty(out_features, packed_in, dtype=torch.uint8)
        elif num_bits == 4:
            packed_in = (in_features + 1) // 2
            weight_storage = torch.empty(out_features, packed_in, dtype=torch.uint8)
        else:
            weight_storage = torch.empty(out_features, in_features, dtype=torch.int8)
        self.weight = nn.Parameter(weight_storage, requires_grad=False)
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1, dtype=torch.float32))
        # SRQ activation scales — optional, loaded from checkpoint. 0 means uncalibrated, in which
        # case `apply_srq` is a no-op, so `forward` can apply it unconditionally.
        self.input_activation_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_activation_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _dequantize_weights(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize weights (handles int2/int4/int8 storage). If `dtype` is given,
        the math runs in that dtype; otherwise int×fp32 promotion gives fp32."""
        if self.num_bits == 2:
            int_weights = _unpack_int2(self.weight, self.in_features)
        elif self.num_bits == 4:
            int_weights = _unpack_int4(self.weight, self.in_features)
        else:
            int_weights = self.weight
        if dtype is None:
            return int_weights * self.weight_scale
        return int_weights.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = apply_srq(x, self.input_activation_scale)
        out = F.linear(x, self._dequantize_weights(x.dtype), self.bias)
        return apply_srq(out, self.output_activation_scale)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, num_bits={self.num_bits}"
        )


class QuantizedEmbedding(nn.Module):
    """Embedding with INT2/4/8 packed table, per-row dequant scale, and architectural embed_scale.

    Does NOT subclass `nn.Embedding` because the packed-int storage isn't a usable
    embedding table on its own: indexing `.embedding_quantized[idx]` returns packed
    bytes, not a row of size `embedding_dim`. Callers expect `embed_tokens.weight[idx, :]`
    to return the *dequantized* row, so we expose `weight` as a property (below)
    that returns the dequantized table on demand.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        output_dtype: torch.dtype,
        embed_scale: float = 1.0,
        num_bits: int = 8,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scalar_embed_scale = embed_scale
        self.num_bits = num_bits
        self.output_dtype = output_dtype

        # int2/int4 packed in uint8 (4 / 2 values per byte); int8 stored directly.
        if num_bits == 2:
            packed_dim = (embedding_dim + 3) // 4
            embed_storage = torch.empty(num_embeddings, packed_dim, dtype=torch.uint8)
        elif num_bits == 4:
            packed_dim = (embedding_dim + 1) // 2
            embed_storage = torch.empty(num_embeddings, packed_dim, dtype=torch.uint8)
        else:
            embed_storage = torch.empty(num_embeddings, embedding_dim, dtype=torch.int8)
        self.embedding_quantized = nn.Parameter(embed_storage, requires_grad=False)
        self.embedding_scale = nn.Parameter(torch.ones(num_embeddings, 1, dtype=torch.float32))

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized embedding table (no architectural `embed_scale` applied).

        Mirrors `nn.Embedding.weight` so callers can do `weight[idx, :]` and get
        the same unscaled row they'd get from a non-quantized embedding.
        """
        return self._dequantize_weights(self.embedding_quantized, self.embedding_scale)

    def _dequantize_weights(self, quant_rows: torch.Tensor, scale_rows: torch.Tensor) -> torch.Tensor:
        """Unpack int2/int4/int8 + apply per-row block-wise dequantization scale."""
        if self.num_bits == 4:
            int_rows = _unpack_int4(quant_rows, self.embedding_dim)
        elif self.num_bits == 2:
            int_rows = _unpack_int2(quant_rows, self.embedding_dim)
        else:
            int_rows = quant_rows

        block_size = self.embedding_dim // scale_rows.shape[-1]
        scale = scale_rows.repeat_interleave(block_size, dim=-1)
        return int_rows.to(self.output_dtype) * scale.to(self.output_dtype)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        result = self._dequantize_weights(self.embedding_quantized[input_ids], self.embedding_scale[input_ids])
        return (result * self.scalar_embed_scale).to(self.output_dtype)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"num_bits={self.num_bits}, embed_scale={self.scalar_embed_scale}"
        )


def replace_with_quant_layers(
    model: nn.Module,
    quantization_config=None,
    modules_to_not_convert: list[str] | None = None,
) -> None:
    """Replace `nn.Linear` / `nn.Embedding` modules with `QuantizedLinear` / `QuantizedEmbedding`.

    Per-module bit widths come from `quantization_config.module_quant_configs`.
    `nn.Embedding` modules are only replaced when `quantize_embeddings` is True.
    Modules whose name matches an entry in `modules_to_not_convert` are skipped.
    """
    import re

    from ..quantizers.quantizers_utils import should_convert_module

    quantize_embeddings = quantization_config.quantize_embeddings
    num_bits = quantization_config.num_bits
    module_quant_configs = quantization_config.module_quant_configs or {}

    # Join all the per-module patterns into one regex, compiled once, so each module name needs a
    # single search instead of a loop over patterns. Each pattern is a named group `g0`, `g1`, ...;
    # whichever group matches identifies its override.
    overrides_by_group = {f"g{i}": override for i, override in enumerate(module_quant_configs.values())}
    matcher = (
        re.compile("|".join(f"(?P<g{i}>{pattern})" for i, pattern in enumerate(module_quant_configs)))
        if module_quant_configs
        else None
    )

    for name, module in list(model.named_modules()):
        if not should_convert_module(name, modules_to_not_convert):
            continue
        opts = {"num_bits": num_bits}
        if matcher is not None and (match := matcher.search(name)) is not None:
            override = next(overrides_by_group[g] for g, v in match.groupdict().items() if v is not None)
            opts = {"num_bits": num_bits, **override}
        if isinstance(module, nn.Embedding):
            if not quantize_embeddings:
                continue
            new_module = QuantizedEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                embed_scale=getattr(module, "scalar_embed_scale", 1.0),
                output_dtype=module.weight.dtype,
                **opts,
            )
        elif isinstance(module, nn.Linear):
            new_module = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                **opts,
            )
        else:
            continue
        new_module.requires_grad_(False)
        model.set_submodule(name, new_module)
    return model
