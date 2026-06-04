# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import re

import torch
import torch.nn as nn

from .gguf_kernels import _KERNEL_FMT, ensure_metal_kernels
from .moe import ExpertsInterface


# Per quant type: (block bytes, block elems). The format string side of the
# table (kernel names like mul_mat_q4_K_f32) lives in gguf_kernels._KERNEL_FMT.
_QUANT_INFO: dict[str, tuple[int, int]] = {
    "Q4_0": (18, 32),
    "Q5_0": (22, 32),
    "Q5_1": (24, 32),
    "Q8_0": (34, 32),
    "Q4_K": (144, 256),
    "Q5_K": (176, 256),
    "Q6_K": (210, 256),
    "IQ4_NL": (18, 32),
    "IQ4_XS": (136, 256),
}


# Q4_K_M averages ~4.5 bits/weight. Used to size the meta-time placeholder
# buffers when the exact quant type isn't known yet (uniform swap): the loader
# replaces them with the real-shaped bytes at load, and this keeps the
# device_map memory estimate (numel × 1 byte for a uint8 buffer) in the right
# ballpark without needing per-tensor quant types up front.
_AVG_BYTES_PER_ELEM = 4.5 / 8


def _quant_type_name(quant_type) -> str | None:
    if quant_type is None:
        return None
    return quant_type.name if hasattr(quant_type, "name") else str(quant_type)


def gguf_linear_supports(quant_type) -> bool:
    """Return True if quant_type has a matching mul_mat_<fmt>_f32 kernel."""
    return _quant_type_name(quant_type) in _QUANT_INFO


def _bytes_per_row(in_features: int, quant_type: str | None) -> int:
    """Row byte-width for a GGUF-quantized weight. Exact when the quant type is
    known; otherwise the Q4_K_M average (placeholder for the meta-time uniform
    swap, corrected to the real width when the bytes load)."""
    if quant_type in _QUANT_INFO:
        block_bytes, block_elems = _QUANT_INFO[quant_type]
        return (in_features // block_elems) * block_bytes
    return max(1, round(in_features * _AVG_BYTES_PER_ELEM))


class GgufLinear(nn.Module):
    """Linear layer with GGUF-quantized weights stored as raw block bytes (uint8).

    Forward dispatches to mul_mat_vec_<fmt>_f32 for batch 1 and mul_mat_<fmt>_f32
    for batched inputs. Inference-only — no dequant fallback. Callers that want
    a CPU/CUDA dequant path should let the loader's GGUFDequantize op produce
    fp32 weights into a standard nn.Linear instead.
    """

    def __init__(self, in_features: int, out_features: int, quant_type: str | None = None, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        # 2D `(out_features, bytes_per_row)` — matches the source byte shape coming
        # out of the GGUF reader, so the standard loader assigns it in without any
        # reshape. `quant_type` may be `None` here: the uniform meta-time swap
        # creates these placeholders without knowing the per-tensor type, sizes the
        # buffer from the Q4_K_M average, and the loader overwrites it with the
        # real-shaped bytes. Kernels + exact quant type are bound post-load via
        # `bind_after_load`.
        self.register_buffer(
            "weight", torch.empty(out_features, _bytes_per_row(in_features, quant_type), dtype=torch.uint8)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._mv_op = self._mat_op = None
        if quant_type is not None:
            self._bind_kernels(quant_type)

    def _bind_kernels(self, quant_type: str) -> None:
        """Resolve the metal matvec/matmul ops for `quant_type` and cache them so
        the forward is branch-free (compile-friendly). `ensure_metal_kernels()`
        raises a clear RuntimeError if the kernels can't load — no fallback."""
        if quant_type not in _QUANT_INFO:
            raise NotImplementedError(f"GgufLinear only supports {sorted(_QUANT_INFO)} today, got {quant_type}")
        block_elems = _QUANT_INFO[quant_type][1]
        if self.in_features % block_elems != 0:
            raise ValueError(
                f"in_features must be a multiple of {block_elems} for {quant_type}, got {self.in_features}"
            )
        self.quant_type = quant_type
        ops = ensure_metal_kernels()._ops
        fmt = _KERNEL_FMT[quant_type]
        self._mv_op = getattr(ops, f"mul_mat_vec_{fmt}_f32").default
        self._mat_op = getattr(ops, f"mul_mat_{fmt}_f32").default

    def bind_after_load(self, quant_type: str | None = None) -> None:
        """Post-load: bind kernels once the real bytes are in `weight`. `quant_type`
        comes from the saved plan on a safetensors reload; otherwise it's read off
        the loaded buffer (a `GGUFQuantizedTensor` carries it). The buffer is
        unwrapped to a plain uint8 tensor so the subclass stays out of the
        compiled forward."""
        if quant_type is None:
            quant_type = _quant_type_name(getattr(self.weight, "quant_type", None))
        if type(self.weight) is not torch.Tensor:
            self.weight = self.weight.as_subclass(torch.Tensor)
        self._bind_kernels(quant_type)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_type}, bias={self.bias is not None}"
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Last dim of x must be {self.in_features}, got {x.shape[-1]}")
        if x.device.type != "mps":
            raise RuntimeError(
                "GgufLinear runs only on MPS with the metal kernels loaded. Re-load with "
                "`dtype=torch.bfloat16` (or any explicit dtype) to dequantize to a normal nn.Linear."
            )

        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).contiguous().to(torch.float32)
        N = x_flat.shape[0]
        # Buffer is 2D `(out, bytes_per_row)` to match the GGUF source layout;
        # kernels want a flat byte array, so view here at the call boundary.
        qw = self.weight.view(-1)
        # Pick the kernel by batch size: matvec when N==1, matmul otherwise. The
        # matmul kernel requires N % 32 == 0; non-aligned batches loop per-row
        # over matvec (rare in practice — most call sites land aligned).
        if N == 1:
            y = torch.empty(self.out_features, dtype=torch.float32, device=x.device)
            self._mv_op(qw, x_flat.view(-1), y)
        elif N % 32 == 0:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            self._mat_op(qw, x_flat, y)
        else:
            y = torch.empty(N * self.out_features, dtype=torch.float32, device=x.device)
            for i in range(N):
                self._mv_op(qw, x_flat[i].reshape(-1), y[i * self.out_features : (i + 1) * self.out_features])
        y = y.view(N, self.out_features).reshape(*batch_shape, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.device).to(y.dtype)
        # Metal matvec / matmul emit fp32. Cast back to the caller's dtype so the
        # KV cache stays in the model's chosen precision instead of silently widening.
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y


class GgufExperts(nn.Module):
    """Drop-in for fused-expert MoE modules with the Mixtral / Qwen2MoE /
    DeepSeek-V3 layout (`gate_up_proj` shape `(num_experts, 2*intermediate,
    hidden)`, `down_proj` shape `(num_experts, hidden, intermediate)`,
    no per-expert biases), but storing raw uint8 GGUF bytes instead of fp
    weights. Mixed quant types per layer are fine — Q4_K_M ships gate/up =
    Q4_K, down = Q8_0.

    Archs with a different layout (transposed params, biases — gpt_oss) are
    not swapped to this class; `GGUFDequantize` lands bf16 into their
    original fused-expert module and `moe.py`'s `batched_mm_experts_forward`
    handles them via its existing `is_transposed` / `has_bias` flags."""

    # ExpertsInterface contract flags.
    has_gate = True
    has_bias = False
    is_transposed = False
    is_concatenated = True

    def __init__(self, config, gate_up_quant: str | None = None, down_quant: str | None = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant

        # uint8 byte buffers, sized from the per-tensor quant type when known and
        # from the Q4_K_M average otherwise (uniform meta-time swap); the loader
        # overwrites them with the real-shaped bytes. Kernels bind post-load.
        self.register_buffer(
            "gate_up_proj",
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                _bytes_per_row(self.hidden_dim, gate_up_quant),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "down_proj",
            torch.empty(
                self.num_experts,
                self.hidden_dim,
                _bytes_per_row(self.intermediate_dim, down_quant),
                dtype=torch.uint8,
            ),
        )

        from ..activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]
        self._id_op_gate_up = self._id_op_down = None
        if gate_up_quant is not None and down_quant is not None:
            self._bind_kernels(gate_up_quant, down_quant)

        decode_size = getattr(config, "num_experts_per_tok", None) or 4
        # Pre-allocated scratch for the decode hot path. At decode there is a single token, so the
        # kernels always see the same `(num_experts_per_tok, ...)` shapes; allocating the gate/up/out
        # buffers once at init (instead of `torch.empty` inside `forward`) keeps the forward graph
        # static so torch.compile / dynamo does not re-trace on every step. Prefill (variable token
        # counts) falls back to fresh allocations — see `gguf_bmm_experts_forward`.
        self.register_buffer(
            "_gate_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_up_buf", torch.empty(decode_size, self.intermediate_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_pair_out", torch.empty(decode_size, self.hidden_dim, dtype=torch.float32), persistent=False
        )
        self._decode_size = decode_size

    def _bind_kernels(self, gate_up_quant: str, down_quant: str) -> None:
        """Resolve and cache the `mul_mat_id` ops for the gate/up and down quant
        types so the forward is branch-free."""
        for label, qt in (("gate_up", gate_up_quant), ("down", down_quant)):
            if qt not in _QUANT_INFO:
                raise NotImplementedError(f"GgufExperts {label}_quant {qt} not in {sorted(_QUANT_INFO)}")
        self.gate_up_quant, self.down_quant = gate_up_quant, down_quant
        ops = ensure_metal_kernels()._ops
        self._id_op_gate_up = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[gate_up_quant]}_f32").default
        self._id_op_down = getattr(ops, f"mul_mat_id_{_KERNEL_FMT[down_quant]}_f32").default

    def bind_after_load(self, gate_up_quant: str | None = None, down_quant: str | None = None) -> None:
        """Post-load: bind kernels from the saved plan (reload) or the loaded
        buffers' carried quant types, and unwrap the byte buffers to plain uint8."""
        if gate_up_quant is None:
            gate_up_quant = _quant_type_name(getattr(self.gate_up_proj, "quant_type", None))
        if down_quant is None:
            down_quant = _quant_type_name(getattr(self.down_proj, "quant_type", None))
        if type(self.gate_up_proj) is not torch.Tensor:
            self.gate_up_proj = self.gate_up_proj.as_subclass(torch.Tensor)
        if type(self.down_proj) is not torch.Tensor:
            self.down_proj = self.down_proj.as_subclass(torch.Tensor)
        self._bind_kernels(gate_up_quant, down_quant)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"gate_up={self.gate_up_quant}, down={self.down_quant}"
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch through ALL_GGUF_EXPERTS_FUNCTIONS (GGUF-specific sibling
        of ALL_EXPERTS_FUNCTIONS). Default is the bmm path."""
        impl = getattr(self, "_experts_implementation", None) or "bmm"
        return ALL_GGUF_EXPERTS_FUNCTIONS.get_interface(impl, gguf_bmm_experts_forward)(
            self, hidden_states, top_k_index, top_k_weights
        )


MODEL_TYPE_TO_GGUF_EXPERTS: dict[str, type[nn.Module]] = {
    # Same layout as MixtralExperts: `gate_up_proj` shape
    # `(num_experts, 2*intermediate, hidden)`.
    "qwen2_moe": GgufExperts,
    "qwen3_moe": GgufExperts,
    "minimax_m2": GgufExperts,
    "mixtral": GgufExperts,
    "deepseek_v3": GgufExperts,
    # gpt_oss is intentionally NOT here. Its experts store `gate_up_proj`
    # transposed — `(num_experts, hidden, 2*intermediate)` — and carry
    # per-expert biases. The Metal `mul_mat_id` kernels we ship assume rows =
    # output axis, so a byte-passthrough GgufExperts subclass would have a
    # broken forward. The FP8 pattern handles archs without specialised kernels
    # the same way: skip the swap, let the dequant op land bf16 into the
    # original `GptOssExperts`, and reuse `moe.py`'s
    # `batched_mm_experts_forward` (which already handles
    # `is_transposed=True` + `has_bias=True`). Re-add gpt_oss here once a
    # transposed-aware `mul_mat_id` kernel exists.
}


@torch.no_grad()
def gguf_bmm_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = self.hidden_dim
    intermediate_dim = self.intermediate_dim
    device = hidden_states.device
    # S = total token-expert pairs across the batch (each routed token contributes
    # num_top_k work items). The mul_mat_id kernel writes one output row per pair.
    S = num_tokens * num_top_k

    selected = hidden_states.repeat_interleave(num_top_k, dim=0).to(torch.float32).contiguous()
    ids32 = top_k_index.reshape(-1).to(torch.int32)
    sample_weights = top_k_weights.reshape(-1).to(torch.float32)

    # Avoids `torch.empty` inside the compiled graph — without this,
    # dynamo would re-trace on the second call
    if num_tokens == 1 and self._decode_size == S:
        # this was alloceted at init time
        gate_buf, up_buf, out = self._gate_buf, self._up_buf, self._pair_out
    else:
        # this one is for prefill
        gate_buf = torch.empty(S, intermediate_dim, dtype=torch.float32, device=device)
        up_buf = torch.empty(S, intermediate_dim, dtype=torch.float32, device=device)
        out = torch.empty(S, hidden_dim, dtype=torch.float32, device=device)

    # TODO: gate_up should be a single call passing gate_up (no slice, contiguous, view here)
    gate_up_bytes = self.gate_up_proj.view(self.num_experts, -1)
    half = gate_up_bytes.shape[1] // 2
    gate_qw = gate_up_bytes[:, :half].contiguous().view(-1)
    up_qw = gate_up_bytes[:, half:].contiguous().view(-1)
    down_qw = self.down_proj.view(-1)
    self._id_op_gate_up(gate_qw, selected, ids32, gate_buf)
    self._id_op_gate_up(up_qw, selected, ids32, up_buf)
    inter = (self.act_fn(gate_buf) * up_buf).contiguous()
    self._id_op_down(down_qw, inter, ids32, out)

    weighted = out * sample_weights.unsqueeze(-1)
    return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


class GgufExpertsInterface(ExpertsInterface):
    _global_mapping = {
        "bmm": gguf_bmm_experts_forward,
    }


ALL_GGUF_EXPERTS_FUNCTIONS = GgufExpertsInterface()


def replace_with_gguf_linear(
    model: nn.Module,
    modules_to_not_convert: set[str] | None = None,
    modules_to_convert: list[str] | None = None,
    quant_type: str | None = None,
) -> int:
    """Uniformly swap every `nn.Linear` (and matching fused-expert module) to a
    `GgufLinear` / `GgufExperts` placeholder at meta time — no source→target
    rename.

    gguf_file / reload (`quant_type=None`): the per-module quant type is unknown
    here; the buffer is sized from the Q4_K_M average and the exact type + metal
    kernels are bound post-load from the bytes that arrive (`bind_after_load`).
    Linears whose GGUF source turns out to be float (e.g. MoE routers) are
    reverted to plain `nn.Linear` then.

    on-the-fly (`quant_type` set): a single config quant type applies to every
    converted module, so it's bound at construction; `modules_to_convert` (glob
    includes) restricts which Linears are quantized.

    `modules_to_not_convert` is a user/skip list; the GGUF file itself decides
    what is quantized, so no default skip-list is applied."""
    import fnmatch

    skip_re = re.compile("|".join(re.escape(p) for p in modules_to_not_convert)) if modules_to_not_convert else None
    experts_cls = MODEL_TYPE_TO_GGUF_EXPERTS.get(getattr(model.config, "model_type", None))
    swapped = 0
    for name, mod in list(model.named_modules()):
        if skip_re is not None and skip_re.search(name):
            continue
        if modules_to_convert is not None and not any(fnmatch.fnmatchcase(name, pat) for pat in modules_to_convert):
            continue
        new_module: nn.Module | None = None
        with torch.device("meta"):
            if isinstance(mod, nn.Linear):
                new_module = GgufLinear(mod.in_features, mod.out_features, quant_type=quant_type, bias=mod.bias is not None)
            elif experts_cls is not None and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj"):
                new_module = experts_cls(config=getattr(mod, "config", model.config), gate_up_quant=quant_type, down_quant=quant_type)
        if new_module is None:
            continue
        model.set_submodule(name, new_module)
        swapped += 1
    return swapped


__all__: list[str] = [
    "ALL_GGUF_EXPERTS_FUNCTIONS",
    "GgufLinear",
    "GgufExperts",
    "MODEL_TYPE_TO_GGUF_EXPERTS",
    "gguf_bmm_experts_forward",
    "gguf_linear_supports",
    "replace_with_gguf_linear",
]
