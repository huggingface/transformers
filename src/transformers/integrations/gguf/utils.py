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

import re
from copy import deepcopy

import torch
from torch import nn

from ...core_model_loading import WeightConverter, WeightRenaming, WeightTransform
from ...utils import logging
from .dequant import GGML_BLOCK
from .gguf_conversion_mapping import GGUF_ARCHS, Dequantize
from .kernels import DEQUANT_CHUNK_ELEMS, MAX_GEMV_ROWS, dequantize_blocks, mul_mat_vec
from .reader import GgufHeader


logger = logging.get_logger(__name__)


def is_gguf_arch_supported(gguf_arch: str) -> bool:
    """Whether this path handles the architecture, or the legacy loader has to."""
    return gguf_arch in GGUF_ARCHS


def get_gguf_conversion_mapping(gguf_arch: str, config) -> list[WeightTransform]:
    """Weight transforms turning a GGUF checkpoint of `gguf_arch` into transformers weights."""
    if gguf_arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {gguf_arch!r} is not supported yet. Supported: {sorted(GGUF_ARCHS)}.")
    return GGUF_ARCHS[gguf_arch](config)


def get_gguf_plan(header: GgufHeader, mapping: list[WeightTransform]) -> tuple[dict[str, int], dict[str, int]]:
    """`{param_name: ggml_type}` for the file's quantized tensors, and the subset that can stay packed.

    Reads `mapping` as it is before `add_gguf_dequantize_ops` has touched it: what a converter does to a
    tensor decides whether it can stay packed, so the answer has to be taken before the unpacking op is
    inserted at the head of those same operations.
    """
    # On a copy: asking a transform whether it matches a name marks it as used and arms the stateful
    # renamings, and the mapping handed back to the loader has to be untouched by that.
    mapping = deepcopy(mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]

    quantized, packable = {}, {}
    for gguf_name, ggml_type in header.ggml_types.items():
        if ggml_type not in GGML_BLOCK:
            continue
        param_name = gguf_name
        for renaming in renamings:
            param_name, _ = renaming.rename_source_key(param_name)
        quantized[param_name] = ggml_type
        # every conversion applied to this tensor must be safe on packed bytes
        converter = next((entry for entry in converters if entry.rename_source_key(param_name)[1]), None)
        if converter is None or all(getattr(op, "supports_packed", False) for op in converter.operations):
            packable[param_name] = ggml_type
    return quantized, packable


def add_gguf_dequantize_ops(mapping: list[WeightTransform], to_unpack: dict[str, int], dtype) -> list:
    """Give every quantized tensor that has to land dense a `Dequantize` as its first conversion.

    `to_unpack` maps a parameter name to its ggml type. It holds only the tensors that need unpacking:
    the ones the file stores quantized whose module cannot compute on blocks.
    """
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    if not to_unpack:
        return mapping
    dequantize_op = Dequantize(to_unpack, dtype)
    for converter in converters:
        converter.operations.insert(0, dequantize_op)
    unconverted = [name for name in to_unpack if not any(c.rename_source_key(name)[1] for c in converters)]
    if unconverted:
        return mapping + [
            WeightConverter(
                source_patterns=[f"({re.escape(name)})" for name in unconverted],
                target_patterns=[r"\1"],
                operations=[dequantize_op],
            )
        ]
    return mapping


class GgufLinear(nn.Module):
    """`nn.Linear` whose weight stays as GGUF blocks: `(out_features, bytes_per_row)` uint8."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ggml_type: int,
        bias: bool = False,
        gemv: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type = ggml_type
        self.gemv = gemv
        block_elems, block_bytes = GGML_BLOCK[ggml_type]
        bytes_per_row = in_features // block_elems * block_bytes
        self.weight = nn.Parameter(torch.empty((out_features, bytes_per_row), dtype=torch.uint8), requires_grad=False)
        # A GGUF stores a bias as its own f32 tensor, never quantized, so it stays an ordinary
        # parameter and the loader fills it like any other.
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        # `gemv` says the kernel implements one for this quantization. Where the weight sits is not checked:
        # running the gemv on a weight the kernel cannot read is a misconfiguration, and it faults there.
        if self.gemv and flat.shape[0] <= MAX_GEMV_ROWS:
            out = mul_mat_vec(self.weight, flat, self.ggml_type, self.out_features)
        else:
            out = self._unpack_matmul(flat)
        if self.bias is not None:
            out = out + self.bias
        out = out.reshape(*x.shape[:-1], self.out_features)
        # The gemv returns f32 whatever `x` is, so this is usually a real cast -- but when the model
        # already runs in f32 it is not, and skipping it saves a dispatch on every linear.
        return out if out.dtype == x.dtype else out.to(x.dtype)

    def _unpack_matmul(self, flat: torch.Tensor) -> torch.Tensor:
        """Matmul against the weight unpacked a row chunk at a time, so it is never fully materialized."""
        rows_per_chunk = max(1, DEQUANT_CHUNK_ELEMS // self.in_features)
        out = torch.empty(flat.shape[0], self.out_features, dtype=flat.dtype, device=flat.device)
        for start in range(0, self.out_features, rows_per_chunk):
            rows = self.weight[start : start + rows_per_chunk]
            chunk = dequantize_blocks(rows, self.ggml_type, rows.shape[0], self.in_features, flat.dtype)
            out[:, start : start + rows_per_chunk] = flat @ chunk.T
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ggml_type={self.ggml_type}, gemv={self.gemv}"
        )


class GgufEmbedding(nn.Module):
    """`nn.Embedding` whose table stays as GGUF blocks."""

    def __init__(self, num_embeddings: int, embedding_dim: int, ggml_type: int, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ggml_type = ggml_type
        self.compute_dtype = dtype or torch.get_default_dtype()
        block_elems, block_bytes = GGML_BLOCK[ggml_type]
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim // block_elems * block_bytes), dtype=torch.uint8),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather these ids' rows out of the packed bytes, then dequantize only those."""
        flat = input_ids.reshape(-1)
        rows = self.weight.index_select(0, flat).contiguous()
        out = dequantize_blocks(rows, self.ggml_type, flat.numel(), self.embedding_dim, self.compute_dtype)
        return out.reshape(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, ggml_type={self.ggml_type}"


def replace_with_gguf_modules(model, plan: dict[str, int], kernel) -> dict[str, nn.Module]:
    """Replace every module named in `plan` with one that holds GGUF blocks; return `{param_name: module}`.

    Runs under the model's init context, so the modules built here get meta weights and the requested
    dtype from `torch.get_default_dtype()`, exactly like the ones they replace.
    """
    # update plan for tied weights
    for target, source in (model._tied_weights_keys or {}).items():
        if source in plan and target not in plan:
            plan[target] = plan[source]

    replaced = {}
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        ggml_type = plan.get(param_name)
        if ggml_type is None:
            continue
        if type(module) is nn.Linear:
            # the fused gemv is used only where one exists; without it the forward unpacks instead
            gemv = bool(kernel) and kernel.supports(ggml_type)
            new_module = GgufLinear(module.in_features, module.out_features, ggml_type, module.bias is not None, gemv)
        elif type(module) is nn.Embedding:
            new_module = GgufEmbedding(module.num_embeddings, module.embedding_dim, ggml_type)
        else:
            continue
        model.set_submodule(module_name, new_module)
        replaced[param_name] = new_module

    # An empty plan is not worth a warning: either the file holds nothing quantized, or `get_gguf_plan`
    # already said why. Weights that could have stayed packed and found no module to hold them are.
    if plan and not replaced:
        logger.warning(
            "You are loading your model from a GGUF file but no module could keep its weights packed."
            " Every quantized tensor will be dequantized at load time."
        )
    return replaced
