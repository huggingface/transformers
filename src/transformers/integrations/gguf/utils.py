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
from .gguf_conversion_mapping import GGUF_ARCHS, Cast, Dequantize
from .kernels import DEQUANT_CHUNK_ELEMS, MAX_GEMV_ROWS, dequantize_blocks, mul_mat_id, mul_mat_vec
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


def get_gguf_plan(
    header: GgufHeader, mapping: list[WeightTransform]
) -> tuple[dict[str, int], dict[str, int], dict[str, torch.Tensor], list[str]]:
    """The file's quantized tensors, the subset that can stay packed, their input permutations, and
    every tensor's renamed name.

    Read before `add_gguf_load_ops` inserts the unpacking op: what a converter does to a tensor
    decides whether it can stay packed.
    """
    # On a copy: asking a transform whether it matches a name marks it as used and arms the stateful
    # renamings, and the mapping handed back to the loader has to be untouched by that.
    mapping = deepcopy(mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]

    quantized, packable, permutations, names = {}, {}, {}, []
    for gguf_name, ggml_type in header.ggml_types.items():
        param_name = gguf_name
        for renaming in renamings:
            param_name, _ = renaming.rename_source_key(param_name)
        names.append(param_name)
        if ggml_type not in GGML_BLOCK:
            continue
        # A converter may rename as well as transform, so a parameter's name is the converter's target.
        converter = None
        for entry in converters:
            renamed, matched = entry.rename_source_key(param_name)
            if matched:
                converter, param_name = entry, renamed
                break
        quantized[param_name] = ggml_type
        operations = getattr(converter, "operations", ())
        # every conversion applied to this tensor must be safe on packed bytes
        if converter is None or all(getattr(op, "supports_packed", False) for op in operations):
            packable[param_name] = ggml_type
            # an op that cannot reorder packed columns asks for its input to be reordered instead
            for operation in operations:
                if (permutation := getattr(operation, "input_permutation", None)) is not None:
                    permutations[param_name] = permutation
    return quantized, packable, permutations, names


def add_gguf_load_ops(mapping: list[WeightTransform], to_unpack: dict[str, int], names: list[str], dtype) -> list:
    """Bracket every conversion chain: unpack blocks first where needed, cast to `dtype` last."""
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    dequantize_op = Dequantize(to_unpack, dtype) if to_unpack else None
    cast_op = Cast(dtype)
    for converter in converters:
        if dequantize_op is not None:
            converter.operations.insert(0, dequantize_op)
        converter.operations.append(cast_op)
    # `Dequantize` passes through a name it was not given, so one converter serves both kinds here
    operations = [dequantize_op, cast_op] if dequantize_op is not None else [cast_op]
    unconverted = [name for name in names if not any(c.rename_source_key(name)[1] for c in converters)]
    if unconverted:
        return mapping + [
            WeightConverter(
                source_patterns=[f"({re.escape(name)})" for name in unconverted],
                target_patterns=[r"\1"],
                operations=operations,
            )
        ]
    return mapping


# The two weights of a stacked expert bank, in the order `GgufExperts` takes their types.
_EXPERT_PARAMS = ("gate_up_proj", "down_proj")


def _row_bytes(ggml_type: int, in_features: int) -> int:
    """How many bytes one row of `in_features` values occupies, packed as `ggml_type`."""
    block_elems, block_bytes = GGML_BLOCK[ggml_type]
    return in_features // block_elems * block_bytes


class GgufLinear(nn.Module):
    """`nn.Linear` whose weight stays as GGUF blocks: `(out_features, bytes_per_row)` uint8."""

    def __init__(self, in_features: int, out_features: int, ggml_type: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type = ggml_type
        self.weight = nn.Parameter(
            torch.empty((out_features, _row_bytes(ggml_type, in_features)), dtype=torch.uint8), requires_grad=False
        )
        # A GGUF stores a bias as its own f32 tensor, never quantized, so it stays an ordinary
        # parameter and the loader fills it like any other.
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.register_buffer("input_permutation", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        if self.input_permutation is not None:
            flat = flat.index_select(1, self.input_permutation)
        if flat.shape[0] <= MAX_GEMV_ROWS:
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
            f"bias={self.bias is not None}, ggml_type={self.ggml_type}, "
            f"permuted_input={self.input_permutation is not None}"
        )


class GgufExperts(nn.Module):
    """A MoE expert bank whose weights stay as GGUF blocks: `(n_experts, rows, bytes_per_row)`.

    Worth keeping packed more than anything else in the model: a router hands each token a handful of
    experts, so the bank is never needed at once, and it is most of the weights -- Qwen3.5-35B-A3B's
    routed experts are 32B parameters, 60 GB unpacked against 20 GB of blocks. Only the experts a
    token hits are read, each through the same fused dequant-gemv a `GgufLinear` uses.

    The loop is the model's own: one pass per expert that was hit, over the tokens that chose it.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, gate_up_type, down_type, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_type = gate_up_type
        self.down_type = down_type
        self.act_fn = act_fn
        self.gate_up_proj = nn.Parameter(
            torch.empty((num_experts, 2 * intermediate_dim, _row_bytes(gate_up_type, hidden_dim)), dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty((num_experts, hidden_dim, _row_bytes(down_type, intermediate_dim)), dtype=torch.uint8),
            requires_grad=False,
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """The whole bank in two dispatches, rather than two per expert the router picked.

        `mul_mat_id` is ggml's own MoE matmul: it takes the bank as one `(n_experts, rows, bytes)`
        tensor and the router's choices, and computes every (token, expert) pair in a single grid.
        The loop this replaces spent its time launching kernels, not running them.
        """
        tokens, used = top_k_index.shape
        ids = top_k_index if top_k_index.dtype == torch.int32 else top_k_index.to(torch.int32)

        # Every slot of a token reads that token's hidden state, so the ids index the weights only.
        fused = mul_mat_id(self.gate_up_proj, hidden_states, ids, self.gate_up_type, 2 * self.intermediate_dim)
        gate, up = fused.chunk(2, dim=-1)
        current = (self.act_fn(gate) * up).reshape(tokens * used, self.intermediate_dim)
        # Each (token, slot) carries its own vector, so the pair flattens into the token axis.
        out = mul_mat_id(self.down_proj, current, ids.reshape(-1, 1), self.down_type, self.hidden_dim)

        # The same weighted sum the model's own experts do.
        out = out.reshape(tokens, used, self.hidden_dim) * top_k_weights.unsqueeze(-1)
        return out.sum(dim=1).to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, ggml_types=({self.gate_up_type}, {self.down_type})"
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
        """Unpack these ids' rows out of the packed table, gathering as it goes."""
        flat = input_ids.reshape(-1)
        out = dequantize_blocks(
            self.weight, self.ggml_type, flat.numel(), self.embedding_dim, self.compute_dtype, indices=flat
        )
        return out.reshape(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, ggml_type={self.ggml_type}"


def replace_with_gguf_modules(model, plan: dict[str, int], kernel, dtype=None) -> dict[str, nn.Module]:
    """Replace every module named in `plan` with one that holds GGUF blocks; return `{param_name: module}`."""
    # update plan for tied weights
    for target, source in (model._tied_weights_keys or {}).items():
        if source in plan and target not in plan:
            plan[target] = plan[source]

    replaced, unsupported = {}, set()
    for module_name, module in model.named_modules():
        # An expert bank holds bare parameters, not child `Linear`s, so match by name. Both stay packed
        # or neither: they are read in one forward.
        expert_types = [plan.get(f"{module_name}.{name}") for name in _EXPERT_PARAMS]
        if all(t is not None for t in expert_types):
            if not all(kernel.supports(t) for t in expert_types):
                unsupported.update(expert_types)
                continue
            new_module = GgufExperts(
                module.num_experts, module.hidden_dim, module.intermediate_dim, *expert_types, module.act_fn
            )
            model.set_submodule(module_name, new_module)
            for name in _EXPERT_PARAMS:
                replaced[f"{module_name}.{name}"] = new_module
            continue

        param_name = f"{module_name}.weight"
        ggml_type = plan.get(param_name)
        if ggml_type is None:
            continue
        if not kernel.supports(ggml_type):
            unsupported.add(ggml_type)
            continue
        if type(module) is nn.Linear:
            new_module = GgufLinear(module.in_features, module.out_features, ggml_type, module.bias is not None)
        elif type(module) is nn.Embedding:
            new_module = GgufEmbedding(module.num_embeddings, module.embedding_dim, ggml_type, dtype)
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
    elif unsupported:
        # Partial otherwise passes unremarked, and the only symptom is memory the caller cannot account for.
        logger.warning(
            f"No GGUF kernel for ggml types {sorted(unsupported)}, so those tensors are dequantized at "
            f"load while the rest stay packed."
        )
    return replaced
