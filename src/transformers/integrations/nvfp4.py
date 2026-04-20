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
"""NVFP4 integration — NVIDIA Blackwell 4-bit float quantization support.

Handles pre-quantized checkpoints produced by NVIDIA's ModelOpt (`weight_packed`,
`weight_scale`, `weight_global_scale` key layout). Forward pass dequantizes on
the fly via a Triton kernel (with a pure-torch fallback)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# E2M1 lookup table: 3-bit magnitude → float value.
E2M1_LOOKUP = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK_SIZE = 16  # NVFP4 groups 16 values per shared FP8 scale.


# ─── Triton kernel (optional) ──────────────────────────────────────────────

_USE_TRITON = False
try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except ImportError:
    pass

if _USE_TRITON:
    @triton.jit
    def _nvfp4_dequant_kernel(
        packed_ptr, scale_ptr, output_ptr,
        global_scale, N_packed, in_features, in_features_packed,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_packed
        packed = tl.load(packed_ptr + offsets, mask=mask, other=0).to(tl.int32)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        low_sign = (low >> 3) & 1
        low_mag = low & 0x07
        low_val = tl.where(low_mag == 0, 0.0,
                  tl.where(low_mag == 1, 0.5,
                  tl.where(low_mag == 2, 1.0,
                  tl.where(low_mag == 3, 1.5,
                  tl.where(low_mag == 4, 2.0,
                  tl.where(low_mag == 5, 3.0,
                  tl.where(low_mag == 6, 4.0, 6.0)))))))
        low_val = low_val * (1.0 - 2.0 * low_sign.to(tl.float32))
        high_sign = (high >> 3) & 1
        high_mag = high & 0x07
        high_val = tl.where(high_mag == 0, 0.0,
                   tl.where(high_mag == 1, 0.5,
                   tl.where(high_mag == 2, 1.0,
                   tl.where(high_mag == 3, 1.5,
                   tl.where(high_mag == 4, 2.0,
                   tl.where(high_mag == 5, 3.0,
                   tl.where(high_mag == 6, 4.0, 6.0)))))))
        high_val = high_val * (1.0 - 2.0 * high_sign.to(tl.float32))
        row = offsets // in_features_packed
        col_packed = offsets % in_features_packed
        col_low = col_packed * 2
        col_high = col_packed * 2 + 1
        scale_cols = in_features // 16
        scale_idx_low = row * scale_cols + col_low // 16
        scale_idx_high = row * scale_cols + col_high // 16
        scale_low = tl.load(scale_ptr + scale_idx_low, mask=mask, other=1.0).to(tl.float32)
        scale_high = tl.load(scale_ptr + scale_idx_high, mask=mask, other=1.0).to(tl.float32)
        low_val = low_val * scale_low * global_scale
        high_val = high_val * scale_high * global_scale
        out_idx_low = row * in_features + col_low
        out_idx_high = row * in_features + col_high
        tl.store(output_ptr + out_idx_low, low_val.to(tl.bfloat16), mask=mask)
        tl.store(output_ptr + out_idx_high, high_val.to(tl.bfloat16), mask=mask)


# ─── Dequant dispatchers ───────────────────────────────────────────────────

def unpack_nvfp4_triton(weight_packed, weight_scale, weight_global_scale, dtype=torch.bfloat16):
    """Triton-accelerated NVFP4 dequant. Requires Blackwell + Triton."""
    out_features = weight_packed.shape[0]
    in_features = weight_packed.shape[1] * 2
    in_features_packed = weight_packed.shape[1]
    N_packed = weight_packed.numel()
    stored_global = weight_global_scale.to(torch.float32).item()
    actual_scale = 1.0 / stored_global if stored_global != 0 else 1.0
    output = torch.empty(out_features, in_features, dtype=dtype, device=weight_packed.device)
    grid = lambda meta: (triton.cdiv(N_packed, meta['BLOCK_SIZE']),)
    _nvfp4_dequant_kernel[grid](
        weight_packed, weight_scale, output,
        actual_scale, N_packed, in_features, in_features_packed,
        BLOCK_SIZE=1024,
    )
    return output


def unpack_nvfp4_python(weight_packed, weight_scale, weight_global_scale, dtype=torch.bfloat16):
    """Pure-torch NVFP4 dequant fallback. Works on any device with torch."""
    device = weight_packed.device
    lookup = E2M1_LOOKUP.to(device)
    packed_flat = weight_packed.flatten().to(torch.int32)
    low_nibble = packed_flat & 0x0F
    high_nibble = (packed_flat >> 4) & 0x0F
    combined = torch.stack([low_nibble, high_nibble], dim=1).flatten()
    signs = (combined >> 3) & 1
    magnitude_idx = combined & 0x07
    values = lookup[magnitude_idx.long()]
    values = values * (1.0 - 2.0 * signs.float())
    out_features = weight_packed.shape[0]
    in_features = weight_packed.shape[1] * 2
    values = values.reshape(out_features, in_features)
    scale = weight_scale.to(torch.float32)
    values_blocked = values.reshape(out_features, -1, BLOCK_SIZE)
    scale_expanded = scale.unsqueeze(-1)
    values_blocked = values_blocked * scale_expanded
    values = values_blocked.reshape(out_features, in_features)
    stored_global = weight_global_scale.to(torch.float32).item()
    actual_scale = 1.0 / stored_global if stored_global != 0 else 1.0
    values = values * actual_scale
    return values.to(dtype)


def unpack_nvfp4(weight_packed, weight_scale, weight_global_scale, dtype=torch.bfloat16, use_triton=True):
    """Dispatcher: prefer Triton when available and enabled."""
    if _USE_TRITON and use_triton:
        return unpack_nvfp4_triton(weight_packed, weight_scale, weight_global_scale, dtype)
    return unpack_nvfp4_python(weight_packed, weight_scale, weight_global_scale, dtype)


# ─── Module ────────────────────────────────────────────────────────────────

class NVFP4Linear(nn.Module):
    """NVFP4-quantized Linear layer. Frozen base weights; no LoRA here.

    Buffer layout matches NVIDIA ModelOpt checkpoint keys:
      - weight_packed: (out_features, in_features // 2) uint8
      - weight_scale:  (out_features, in_features // 16) float8_e4m3fn
      - weight_global_scale: () float32

    Forward dequantizes on the fly (or uses cached bf16 weight if
    cache_dequant() was called).
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        in_features_packed = in_features // 2
        scale_groups = in_features // 16
        # Empty buffers; populated by HF state_dict loader from checkpoint.
        self.register_buffer(
            "weight_packed",
            torch.empty(out_features, in_features_packed, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, scale_groups, dtype=torch.float8_e4m3fn, device=device),
        )
        self.register_buffer(
            "weight_global_scale",
            torch.empty((), dtype=torch.float32, device=device),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None
        self._cached_weight = None
        self._use_triton = True

    @property
    def qweight(self):
        """Alias for weight_packed — lets PEFT introspect the packed tensor
        the same way it handles bitsandbytes 4bit quantized weights."""
        return self.weight_packed

    def cache_dequant(self):
        """Pre-compute and cache the dequantized bf16 weight for faster forward."""
        self._cached_weight = unpack_nvfp4(
            self.weight_packed, self.weight_scale, self.weight_global_scale,
            dtype=torch.bfloat16, use_triton=self._use_triton,
        ).detach()

    def _init_weights(self, module):
        pass

    def reset_parameters(self):
        pass

    def forward(self, x):
        if self._cached_weight is not None:
            W_deq = self._cached_weight
        else:
            W_deq = unpack_nvfp4(
                self.weight_packed, self.weight_scale, self.weight_global_scale,
                dtype=x.dtype, use_triton=self._use_triton,
            )
        return F.linear(x, W_deq, self.bias)


# ─── Model traversal: replace nn.Linear with NVFP4Linear ───────────────────

def replace_with_nvfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """Recursively walk `model`, swap every nn.Linear for NVFP4Linear.

    Skips modules whose dotted name matches any pattern in
    `modules_to_not_convert` (e.g. `['lm_head']`).

    Returns:
        (model, has_been_replaced): model with replacements applied; bool
        indicating whether any replacement happened.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)

        if isinstance(module, nn.Linear) and not any(
            pattern in current_key_name_str for pattern in modules_to_not_convert
        ):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            use_triton = (
                quantization_config.use_triton
                if quantization_config is not None and hasattr(quantization_config, "use_triton")
                else True
            )
            new_module = NVFP4Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
                device=module.weight.device if module.weight.device.type != "meta" else None,
            )
            new_module._use_triton = use_triton
            model._modules[name] = new_module
            has_been_replaced = True

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_nvfp4_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced,
            )
        current_key_name.pop(-1)

    return model, has_been_replaced


# ─── MoE expert replacement (Qwen 3.5 fused-parameter pattern) ─────────────

class _NVFP4ExpertTriple(nn.Module):
    """SwiGLU triple: gate_proj, up_proj, down_proj — each as NVFP4Linear.
    Used as one child of NVFP4MoeExperts per expert."""

    def __init__(self, hidden_dim, intermediate_dim, device=None):
        super().__init__()
        self.gate_proj = NVFP4Linear(hidden_dim, intermediate_dim, bias=False, device=device)
        self.up_proj = NVFP4Linear(hidden_dim, intermediate_dim, bias=False, device=device)
        self.down_proj = NVFP4Linear(intermediate_dim, hidden_dim, bias=False, device=device)


class NVFP4MoeExperts(nn.Module):
    """Replacement for Qwen3_5MoeExperts (fused bf16) using per-expert NVFP4Linear.

    Child modules are registered as "0", "1", ... "num_experts-1" so state_dict
    keys like `<prefix>.experts.0.gate_proj.weight_packed` map directly to the
    corresponding buffer in expert 0's gate_proj — no custom Conversion Op
    needed, HF's default loader handles population by name.

    Forward routes each token through its top-K experts; dequant happens
    on-the-fly per expert per forward. A future optimization can add an LRU
    cache for hot experts (see train_security_nvfp4.py for reference).
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim,
                 act_fn_name="silu", device=None):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = F.silu if act_fn_name == "silu" else F.gelu
        for i in range(num_experts):
            self.add_module(
                str(i),
                _NVFP4ExpertTriple(hidden_dim, intermediate_dim, device=device),
            )

    def _init_weights(self, module):
        pass

    def reset_parameters(self):
        pass

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]
            top_k_index:   [num_tokens, top_k] expert indices
            top_k_weights: [num_tokens, top_k] routing weights
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0].item()
            if expert_idx >= self.num_experts:
                continue
            idx = str(expert_idx)
            if idx not in self._modules:
                continue
            expert = self._modules[idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_out = expert.gate_proj(current_state)
            up_out = expert.up_proj(current_state)
            current_hidden = self.act_fn(gate_out) * up_out
            current_hidden = expert.down_proj(current_hidden)

            current_hidden = current_hidden * top_k_weights[
                token_idx, top_k_pos
            ].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, current_hidden)

        return final_hidden_states


def replace_fused_moe_experts_with_nvfp4(
    model, modules_to_not_convert=None, quantization_config=None
):
    """Replace Qwen3_5MoeExperts (fused bf16) with NVFP4MoeExperts (per-expert NVFP4).

    Walks the model, finds every Qwen3_5MoeExperts module, and substitutes it
    with an NVFP4MoeExperts of matching dimensions. After this runs, HF's
    state_dict loader will populate the per-expert NVFP4 buffers by name.

    Returns (model, has_been_replaced).
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeExperts,
        )
    except ImportError:
        return model, False

    has_been_replaced = False
    for name, child in list(model.named_modules()):
        if not isinstance(child, Qwen3_5MoeExperts):
            continue
        if any(pattern in name for pattern in modules_to_not_convert):
            continue
        nvfp4_experts = NVFP4MoeExperts(
            num_experts=child.num_experts,
            hidden_dim=child.hidden_dim,
            intermediate_dim=child.intermediate_dim,
            act_fn_name="silu",
        )
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nvfp4_experts)
        has_been_replaced = True

    return model, has_been_replaced
import torch as _torch_for_nvfp4_place
from ..core_model_loading import ConversionOps as _ConversionOps_nvfp4


class NVFP4PlaceOp(_ConversionOps_nvfp4):
    """Stream NVFP4 tensors into pre-allocated GPU buffers."""

    @_torch_for_nvfp4_place.no_grad()
    def convert(self, input_dict, source_patterns=None, target_patterns=None, **kwargs):
        import sys as _sys
        full_layer_name = kwargs.get("full_layer_name")
        model = kwargs.get("model")
        _cls = type(self)
        _cnt = getattr(_cls, "_dbg_cnt", 0)
        _cls._dbg_cnt = _cnt + 1

        if _cnt < 20:
            print(f"[PLACEOP#{_cnt}] flname={full_layer_name!r} keys={list(input_dict.keys()) if input_dict else []}", file=_sys.stderr, flush=True)

        if model is None or full_layer_name is None or not input_dict:
            if _cnt < 20: print(f"[PLACEOP#{_cnt}] GUARD_A", file=_sys.stderr, flush=True)
            return input_dict

        suffix = next(iter(input_dict.keys()))
        module_path = full_layer_name[:-len(suffix)] if full_layer_name.endswith(suffix) else full_layer_name
        module_path = module_path.strip(".")
        buf_name = suffix.lstrip(".")

        try:
            module = model.get_submodule(module_path)
        except Exception as _e:
            if _cnt < 20: print(f"[PLACEOP#{_cnt}] GUARD_B path={module_path!r} err={type(_e).__name__}:{_e}", file=_sys.stderr, flush=True)
            return input_dict

        target_buffer = getattr(module, buf_name, None)
        if target_buffer is None:
            if _cnt < 20: print(f"[PLACEOP#{_cnt}] GUARD_C module={type(module).__name__} buf={buf_name!r} attrs={[a for a in dir(module) if not a.startswith('_')][:10]}", file=_sys.stderr, flush=True)
            return input_dict

        # FIX: if buffer is on meta device, materialize it on cuda before copy
        import torch as _torch_nvfp4_fix
        if target_buffer.device.type == "meta":
            _real_dev = _torch_nvfp4_fix.device("cuda:0")
            new_buf = _torch_nvfp4_fix.empty(
                target_buffer.shape, dtype=target_buffer.dtype, device=_real_dev
            )
            module._buffers[buf_name] = new_buf
            target_buffer = new_buf

        tensor_or_list = input_dict[suffix]
        tensor = tensor_or_list[0] if isinstance(tensor_or_list, list) else tensor_or_list

        if _cnt < 20:
            print(f"[PLACEOP#{_cnt}] REACHED_COPY tensor_type={type(tensor).__name__} has_shape={hasattr(tensor,'shape')}", file=_sys.stderr, flush=True)

        try:
            _src = tensor.to(target_buffer.device) if hasattr(tensor, 'to') else tensor
            # Reshape scalar-like mismatches (e.g. checkpoint [1] vs buffer ())
            if _src.shape != target_buffer.shape and _src.numel() == target_buffer.numel():
                _src = _src.reshape(target_buffer.shape)
            if _src.shape == target_buffer.shape and _src.dtype == target_buffer.dtype:
                target_buffer.copy_(_src)
                _copy_done = True
                if _cnt < 20: print(f"[PLACEOP#{_cnt}] COPY_OK shape={tuple(target_buffer.shape)}", file=_sys.stderr, flush=True)
            else:
                # True mismatch — replace the buffer entry, do NOT free source storage afterward (aliasing risk)
                module._buffers[buf_name] = _src.clone()
                _copy_done = False
                if _cnt < 20: print(f"[PLACEOP#{_cnt}] REPLACED (no storage kill) tgt={tuple(target_buffer.shape)}/{target_buffer.dtype} src={tuple(_src.shape)}/{_src.dtype}", file=_sys.stderr, flush=True)
        except Exception as _e:
            if _cnt < 20: print(f"[PLACEOP#{_cnt}] COPY_FAIL {type(_e).__name__}:{_e}", file=_sys.stderr, flush=True)
            return input_dict

        # Only free source storage when we did an in-place copy (no aliasing).
        if _copy_done:
            try:
                tensor.untyped_storage().resize_(0)
            except Exception:
                pass

        return {}
