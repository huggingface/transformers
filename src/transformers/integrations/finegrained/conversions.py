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
"""Loader/saver conversion ops for the block-scaled formats.

Each op moves ONE checkpoint-side layout to the one the `finegrained` modules hold, and carries a
`reverse_op` so `save_pretrained` writes the checkpoint's layout back. Nothing here runs in a
forward.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from ...core_model_loading import ConversionOps, Interleave
from ...quantizers.quantizers_utils import FUSED_STACKS, get_module_from_name
from .layers import (
    _FP8_DTYPE,
    _FP8_MAX,
    _FP8_MIN,
    FineGrainedExperts,
    WeightFormat,
    _cdiv,
    _FineGrainedModule,
    _get_ue8m0_dtype,
    load_finegrained_kernel,
    weight_formats,
)


def single_tensor(value):
    """A converter's tensor, which the loader hands in a list: one source's is a one-element list. Several are
    sources this op does not merge, so they are refused rather than cut down to the first."""
    if not isinstance(value, list):
        return value
    if len(value) != 1:
        raise ValueError(f"expected a single tensor, got a list of {len(value)}")
    return value[0]


def keyed_by_target(input_dict: dict, target_patterns) -> dict:
    """A converter's ops see their single tensor under the SOURCE pattern until an op re-keys it to the
    target (the core ops do; the loader then expands the target into the full name) — so a
    layout op that is first in its converter re-keys the same way. Multi-tensor dicts (a
    deserializer's fully named outputs) pass through."""
    if target_patterns and len(input_dict) == 1:
        value = next(iter(input_dict.values()))
        return {target_patterns[0]: single_tensor(value)}
    return input_dict


def held_scale(model, full_layer_name, key):
    """The finegrained module and its ``*_scale_inv`` Parameter a converter output fills, else
    ``(None, None)``. A converter emitting several tensors names them fully (key); a single target's
    name arrives as the pattern, its full name (with the ``_scale_inv`` suffix) in ``full_layer_name``."""
    param_name = (key if key.endswith("_scale_inv") else full_layer_name).rpartition(".")[-1]
    if not param_name.endswith("_scale_inv") or model is None:
        return None, None
    module = model.get_submodule(full_layer_name.rpartition(".")[0])
    if not isinstance(module, _FineGrainedModule):
        return None, None
    return module, getattr(module, param_name, None)


def finegrained_module(model, full_layer_name):
    """The finegrained module the tensor at ``full_layer_name`` fills, else ``None``. On save the name
    runs past the module (``experts.*.gate_proj.weight``)."""
    name = full_layer_name or ""
    while model is not None and name:
        name = name.rpartition(".")[0]
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        return module if isinstance(module, _FineGrainedModule) else None
    return None


def finegrained_experts(model, full_layer_name):
    """The ``FineGrainedExperts`` the tensor at ``full_layer_name`` fills, else ``None``."""
    module = finegrained_module(model, full_layer_name)
    return module if isinstance(module, FineGrainedExperts) else None


def as_container(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The same bytes under a same-width dtype (uint8 <-> e8m0), an exact numeric cast otherwise."""
    return scale.view(dtype) if scale.element_size() == dtype.itemsize else scale.to(dtype)


class _FineGrainedOp(ConversionOps):
    """Base for the ops below. Each carries the quantizer that built it, which is how an op reaches
    the config (the activation format, the block size) while it runs.

    The default reverse is the op ITSELF with `inverse` flipped: it applies the layout the modules
    hold on load and puts the checkpoint's back on save. The pairs that reverse into a *different*
    class — merge/split, quantize/dequantize — override `reverse_op` and ignore the flag. A layout op
    built for one module carries it, since a save may name the tensor as the checkpoint does.
    """

    def __init__(self, hf_quantizer=None, inverse: bool = False, module=None):
        self.hf_quantizer = hf_quantizer
        self.inverse = inverse
        self.module = module

    @property
    def reverse_op(self) -> ConversionOps:
        return type(self)(self.hf_quantizer, inverse=not self.inverse, module=self.module)


class FineGrainedDeserialize(_FineGrainedOp):
    """Apply a module's reversible layouts after its checkpoint tensors have been merged.

    On-the-fly quantization applies these layouts to its outputs instead, after quantizing the
    full-precision rows. Saving reverses the layouts before splitting or renaming the tensors.
    """

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        if not self.inverse:
            input_dict = keyed_by_target(input_dict, target_patterns)
            if not self.hf_quantizer.pre_quantized:
                return input_dict
        module = finegrained_module(model, full_layer_name)
        if module is None:
            return input_dict
        module_name = next(name for name, candidate in model.named_modules() if candidate is module)
        result = {}
        for key, value in input_dict.items():
            name = key.rpartition(".")[2]
            member = self.hf_quantizer.member_of(module)
            ops = member.get_param_ops(model, f"{module_name}.{name}") if member is not None else []
            if self.inverse:
                ops = [op.reverse_op for op in ops[::-1]]
            tensors = {key: value}
            for op in ops:
                tensors = op.convert(
                    tensors, model=model, full_layer_name=f"{module_name}.{name}", target_patterns=[key], **kwargs
                )
            result.update(tensors)
        return result


class FineGrainedInterleaveGateUp(_FineGrainedOp):
    """Expert gate|up rows between the checkpoint's order and the one the experts hold: stacked
    ``[gate; up]`` or the kernels' ``[g0, u0, g1, u1, ...]`` — the core ``Interleave`` along dim 1,
    over the weight, scale grid and bias alike. Runs on load and save wherever the two differ."""

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        input_dict = keyed_by_target(input_dict, target_patterns)
        experts = finegrained_experts(model, full_layer_name)
        # `is_concatenated` is the checkpoint's stacked order
        if experts is None or not experts.has_gate or experts.is_concatenated != experts.holds_interleaved_gate_up:
            return input_dict
        # inverse=True is stacked -> interleaved
        interleave = Interleave(dim=1, inverse=experts.holds_interleaved_gate_up != self.inverse)
        out = {}
        for key, value in input_dict.items():
            value = single_tensor(value)
            # a per-tensor scale is ONE value covering the whole gate|up stack — no row axis to
            # reorder, and reshaping its length-1 axis into pairs is what would fail
            if getattr(value, "ndim", 0) < 2 or value.shape[1] < 2:
                out[key] = value
            else:
                out[key] = interleave.convert({key: value}, None, [key])[key]
        return out


class FineGrainedScaleContainer(_FineGrainedOp):
    """A UE8M0 block scale into the ``float8_e8m0fnu`` its module holds: it ships either as the
    exponent byte under ``uint8`` (reinterpreted) or as its power-of-two value in float32 (cast
    exactly). Exponent-byte checkpoints save back as uint8; float32 containers save as native UE8M0.
    Every other scale
    stays in the dtype the checkpoint ships (Qwen3 and Mistral BF16, DeepSeek-V3 fp32)."""

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        out = {}
        for key, value in keyed_by_target(input_dict, target_patterns).items():
            value = single_tensor(value)
            if self.inverse:
                # the container its module recorded at load; e8m0 is only ever a scale, so the dtype
                # alone picks the tensors to restore
                module = self.module or finegrained_module(model, full_layer_name)
                container = getattr(module, "scale_container_dtype", None)
                if container is not None and value.dtype == _get_ue8m0_dtype():
                    value = as_container(value, container)
            else:
                module, held = held_scale(model, full_layer_name, key)
                # the one container there is: e8m0 held, shipped as its bytes or its fp32 powers of two
                if held is not None and held.dtype == _get_ue8m0_dtype() and value.dtype != held.dtype:
                    if value.dtype == torch.uint8:
                        module.scale_container_dtype = torch.uint8
                    value = as_container(value, held.dtype)
            out[key] = value
        return out


class FineGrainedSwizzleScales(_FineGrainedOp):
    """Expert block scales into the ``SWIZZLE_32_4_4`` layout their module holds (a 5-D Parameter):
    one triton launch per rank-local shard, values unchanged. The swizzle covers whole 128-row
    blocks and 4-column groups, so the reverse reads the affine grid back off the 5-D shape."""

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        out = {}
        for key, value in keyed_by_target(input_dict, target_patterns).items():
            value = single_tensor(value)
            out[key] = self._unswizzle(value) if self.inverse else self._swizzle(key, value, model, full_layer_name)
        return out

    @staticmethod
    def _unswizzle(value):
        if value.ndim != 5:
            return value
        experts, row_blocks, col_groups = value.shape[:3]
        kernel = load_finegrained_kernel()
        return kernel.unswizzle_mx_scales(value, row_blocks * 128, col_groups * 4, num_experts=experts)

    @staticmethod
    def _swizzle(key, value, model, full_layer_name):
        _, held = held_scale(model, full_layer_name, key)
        if held is None or held.ndim != 5 or value.ndim == 5:
            return value
        return load_finegrained_kernel().swizzle_mx_scales(value)


class FineGrainedPackedBlocks(_FineGrainedOp):
    """GPT-OSS's ``{proj}_blocks`` ``(E, N, K/32, 16)`` uint8 as the packed ``(E, N, K/2)`` int8
    the kernels read: the same bytes regrouped per row, folded back into 16-byte groups on the way
    out. Two low-nibble-first E2M1 values per byte."""

    def convert(self, input_dict, target_patterns=None, **kwargs):
        out = {}
        for key, value in keyed_by_target(input_dict, target_patterns).items():
            value = single_tensor(value)
            if "_scales" in key:  # a converter that carries the scale alongside (the dequant chain)
                out[key] = value
            elif self.inverse:
                out[key] = value.view(torch.uint8).reshape(*value.shape[:-1], value.shape[-1] // 16, 16)
            else:
                out[key] = value.reshape(*value.shape[:-2], -1).view(torch.int8)
        return out


class FineGrainedViewPackedInt8(_FineGrainedOp):
    """Bitcast packed-FP4 uint8 checkpoint bytes to the int8 view the finegrained modules store:
    ``copy_`` into an int8 param would numerically CONVERT and corrupt values >= 128. Non-uint8
    tensors pass through, so this can ride converters that also match unquantized modules."""

    def convert(self, input_dict, target_patterns=None, **kwargs):
        held, want = (torch.int8, torch.uint8) if self.inverse else (torch.uint8, torch.int8)
        out = {}
        for key, value in keyed_by_target(input_dict, target_patterns).items():
            value = single_tensor(value)
            out[key] = value.view(want) if torch.is_tensor(value) and value.dtype == held else value
        return out


class FineGrainedReciprocal(_FineGrainedOp):
    """compressed-tensors stores NVFP4's globals as ``448 * 6 / amax``, the reciprocal of the multiplier the kernels
    (and modelopt) take. Its own inverse, so a save writes them back as shipped."""

    def convert(self, input_dict, target_patterns=None, **kwargs):
        return {
            key: [1 / tensor for tensor in value] if isinstance(value, list) else 1 / value
            for key, value in keyed_by_target(input_dict, target_patterns).items()
        }


class FineGrainedWeightGlobals(_FineGrainedOp):
    """The second-level NVFP4 globals and the calibrated ``input_scale``, as the kernels index them:
    one fp32 global per expert per projection, from either the per-expert checkpoint keys
    (``experts.*.gate_proj.weight_scale_2``) or one stacked ``(E, 2)`` tensor per layer.

    A gate|up stack has two per expert and merging them is not local, so one op owns a layer's
    whole set: the stack keeps the gate's and the up half's leaves as a ratio. ``inter =
    silu(gate) * up`` is linear in the up half, so that only scales the expert output by
    ``1 / ratio`` — which the down projection's weight global takes back, and its calibrated input
    global keeps the requant on the range the e4m3 block scales were chosen for. Rescaling the up
    half's block scales instead would re-round them against codes chosen for the old global."""

    @staticmethod
    def role(key: str) -> tuple[str, str]:
        """What a global-scale key holds, for a checkpoint key and a module target alike: which
        projection it belongs to, and which of modelopt's two levels it is — the weight's
        second-level global (``weight_scale_2``) or the calibrated activation one (``input_scale``).

        This runs while the converters are being BUILT, against raw checkpoint keys, so the
        vLLM-style `w1`/`w2`/`w3` names are still in play and have not been renamed to
        `gate/down/up_proj` yet. Matched on path SEGMENTS: the default is gate_up, so a loose
        `"w2" in key` would silently bucket a down global as one."""
        down = any(seg == "w2" or seg.startswith("down_proj") for seg in key.split("."))
        level = "input" if "input_scale" in key or "activation_scale" in key else "weight"
        return ("down" if down else "gate_up"), level

    def convert(self, input_dict, target_patterns=None, model=None, **kwargs):
        sources = defaultdict(dict)
        for key, value in input_dict.items():
            # the loader hands a converter's tensors in a list; unwrapped, a fused `(E, 2)` pair
            # reads as `(1, 2E)` and the per-half fold below silently does not fire
            sources[self.role(key)][key] = value[0] if isinstance(value, list) and len(value) == 1 else value
        globals_ = {}
        for target in target_patterns:
            if role_sources := sources.get(self.role(target)):
                # one (E, calibrated projections) fp32 tensor, gate column first: two sources are
                # the stack's halves, one is the fused (E, 2) pair or a single projection's (E,)
                columns = []
                for key, value in sorted(role_sources.items(), key=lambda kv: 0 if "gate_proj" in kv[0] else 1):
                    value = torch.stack(value, dim=0) if isinstance(value, list) else value
                    columns.append(value.float().reshape(value.shape[0], -1))
                globals_[target] = torch.cat(columns, dim=1)
        targets = {self.role(target): target for target in globals_}
        stack = targets.get(("gate_up", "weight"))

        # a stack calibrated per half keeps the gate's global; the up half's leaves as a ratio,
        # which the down's weight global takes on (scaling the expert output back up) and its
        # input global gives back (keeping the requantized intermediate on the calibrated range)
        if stack is not None and globals_[stack].shape[1] == 2:
            gate, up = globals_[stack].chunk(2, dim=1)
            gate, up = gate.squeeze(1), up.squeeze(1)
            if not torch.equal(gate, up):
                self._assert_halves_foldable(model)
            globals_[stack] = gate
            ratio = (up / gate)[:, None]
            for role, factor in ((("down", "weight"), ratio), (("down", "input"), 1 / ratio)):
                if target := targets.get(role):
                    globals_[target] = globals_[target] * factor
        return {target: value.reshape(-1).contiguous() for target, value in globals_.items()}

    @staticmethod
    def _assert_halves_foldable(model) -> None:
        """Merging the halves' globals moves the up half's past the GLU. A gate|up bias is added AFTER the global, so
        its up rows would have to be divided by the same ratio, in the bias converter the layout ops own; no
        checkpoint pairs the two. And the GLU must be linear in `up`: a clamped SwiGLU (`swiglu_limit`) clips the
        rescaled values, and SwiGLU-OAI (`swiglu_alpha`) multiplies by `up + 1`."""
        experts = [module for module in model.modules() if isinstance(module, FineGrainedExperts)] if model else []
        if any(module.has_bias for module in experts):
            raise NotImplementedError(
                "this checkpoint calibrates the gate|up halves separately AND carries an expert "
                "bias; the up half's bias would have to be folded with the globals."
            )
        if any(module.swiglu_limit is not None or module.swiglu_alpha is not None for module in experts):
            raise NotImplementedError(
                "this checkpoint calibrates the gate|up halves separately, and this model's GLU "
                "(`swiglu_limit` / `swiglu_alpha`) is not linear in the up half, so the halves' globals "
                "cannot be merged into one per expert."
            )


class FineGrainedInputScales(_FineGrainedOp):
    """A calibrated checkpoint's ``input_scale`` in the layout the module holds: one value per
    quantized module, so a MoE brings one per expert, the gate|up pair reducing to their max since
    both halves read the same rows."""

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        values = []
        for value in input_dict.values():
            # unwrapped, a fused `(E,)` scale in the loader's list reads as `(1, E)` and reduces across experts
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else torch.stack(value, dim=0)
            values.append(value.float())
        stacked = torch.stack([v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(-1, 1) for v in values], dim=-1)
        # a scale is a magnitude, so the reduce is over absolute values
        per_expert = stacked.reshape(stacked.shape[0], -1).abs().amax(dim=1)
        return {full_layer_name: per_expert.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedInputScalesSplit(self.hf_quantizer)


class FineGrainedInputScalesSplit(_FineGrainedOp):
    """Save reverse of :class:`FineGrainedInputScales`: the module's scale back onto every
    projection key the checkpoint calibrated one for. The merge took a max, so the folded halves
    are gone and each projection is written the value that covers it. A collapsed global
    re-expands per expert, the shape the checkpoint holds."""

    def convert(self, input_dict, model=None, full_layer_name=None, target_patterns=None, **kwargs):
        value = single_tensor(next(iter(input_dict.values())))
        if value.numel() == 1:
            experts = finegrained_experts(model, full_layer_name)
            if experts is not None:
                value = value.reshape(1).expand(experts.num_experts).contiguous()
        # one tensor per key: a save refuses keys that share storage
        return {key: value.clone() for key in target_patterns or list(input_dict)}

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedInputScales(self.hf_quantizer)


class FineGrainedQuantize(_FineGrainedOp):
    """Quantize a full-precision weight on load into the format its module holds, emitting the
    scale (and the NVFP4 global) in its held dtype. Each full stack is quantized before applying
    the module's layouts (gate|up interleave, scale swizzle), so a block covers one projection.

    Block-FP8 is computed in torch; the group formats run the kernels' row-wise quantizers, NVFP4
    after normalizing by the canonical global ``amax / (6 * 448)``. A tensor that is not a
    finegrained module's weight passes through — `_weight_holder` decides, since rank alone does
    not (an expert bias is 2-D)."""

    def convert(self, input_dict: dict[str, torch.Tensor], model=None, **kwargs) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, value in input_dict.items():
            tensor = single_tensor(value)
            holder = self._weight_holder(model, key)
            if holder is None and model is not None:
                # not a weight, whatever its rank: an expert bias is 2-D and would otherwise be
                # quantized into a `<proj>_bias_scale_inv` no module holds. Only a direct
                # invocation, with no model to ask, quantizes without a holder.
                result[key] = tensor
                continue
            quantized = self._quantize_one(key, tensor, holder)
            for name, value in quantized.items():
                tensors = {name: value}
                if holder is not None and hasattr(self.hf_quantizer, "member_of"):
                    member = self.hf_quantizer.member_of(holder[0])
                    for op in member.get_param_ops(model, name) if member is not None else []:
                        tensors = op.convert(tensors, model=model, full_layer_name=name, target_patterns=[name])
                result.update(tensors)
        return result

    @staticmethod
    def _as_expert_rows(module, name: str, value: torch.Tensor) -> torch.Tensor:
        """An expert stack in the ``(E, rows, K)`` the module's slot and the kernels take. A model may store its
        experts transposed (GPT-OSS: ``(E, H, 2I)``), which the module it replaced declares: quantizing before the
        swap packs the wrong axis, and a square stack (GPT-OSS's down projection) cannot tell from its shape."""
        if module is None or value.ndim != 3 or name not in FUSED_STACKS:
            return value
        return value.transpose(1, 2).contiguous() if getattr(module, "ships_transposed", False) else value

    @staticmethod
    def _weight_holder(model, key: str):
        """``(module, scale_param_name)`` when ``key`` is a finegrained module's weight, else ``None``."""
        if model is None:
            return None
        module, name = get_module_from_name(model, key)
        if not isinstance(module, _FineGrainedModule) or getattr(module, name, None) is None:
            return None
        if name == "weight":
            return module, "weight_scale_inv"
        if name in FUSED_STACKS:
            return module, f"{name}_scale_inv"
        return None

    def _quantize_one(self, key: str, value: torch.Tensor, holder) -> dict[str, torch.Tensor]:
        if value.ndim < 2:
            return {key: value}
        prefix = key.rsplit(".", 1)[0] + ".weight" if key.endswith(".weight") else key
        module, scale_name = holder if holder is not None else (None, None)
        value = self._as_expert_rows(module, key.rsplit(".", 1)[1], value)
        held = getattr(module, scale_name) if module is not None else None
        format_spec = weight_formats()[module.weight_format] if module is not None else None
        if format_spec is None or format_spec.scale_group is None:
            if module is not None and module.block_size:
                block = tuple(module.block_size)
            else:
                # no module (direct invocation): take the block from the config, and failing that
                # put one scale over the whole matrix
                config = self.hf_quantizer.quantization_config if self.hf_quantizer is not None else None
                config_block = (
                    config.get("weight_block_size")
                    if isinstance(config, dict)
                    else getattr(config, "weight_block_size", None)
                )
                block = tuple(config_block) if config_block else (value.shape[-2], value.shape[-1])
            ue8m0 = (
                held.dtype == _get_ue8m0_dtype()
                if held is not None
                else self.hf_quantizer.quantization_config.scale_fmt == "ue8m0"
            )
            weight, scale = self._quantize_block_fp8(value, block, ue8m0)
            return {
                key: weight,
                f"{prefix}_scale_inv": scale,
                **self._identity_activation_scales(module, prefix, scale_name, value.device),
            }

        if value.device.type not in ("cuda", "xpu"):
            # rocm counts: torch reports it as "cuda"
            raise ValueError(
                f"on-the-fly {module.weight_format} quantization runs the kernels' quantizers on an "
                f"accelerator, but the weight is on {value.device.type}"
            )
        weight, scale, global_scale = self._quantize_group(value, format_spec)
        scale = as_container(scale, held.dtype)
        out = {key: weight, f"{prefix}_scale_inv": scale}
        if global_scale is not None:
            # the experts name their global `<proj>_weight_global_scale`, a dense linear just
            # `weight_global_scale` — both sit beside a `<stem>_scale_inv`, so ask for the slot
            stem = scale_name.removesuffix("_scale_inv")
            suffix = "_global_scale" if hasattr(module, f"{stem}_global_scale") else "_weight_global_scale"
            out[f"{prefix}{suffix}"] = global_scale
        out.update(self._identity_activation_scales(module, prefix, scale_name, value.device))
        return out

    @staticmethod
    def _identity_activation_scales(module, prefix: str, scale_name: str | None, device) -> dict:
        """This module's activation-side scales, as the identity, because quantizing on the fly
        is not a calibration pass. They have to be WRITTEN: the loader materializes a key no
        checkpoint supplies with `torch.empty_like` and `_init_weights` has no branch for a
        scale, so a slot left out here reaches the kernels as uninitialized memory."""
        if module is None:
            return {}
        stem = scale_name.removesuffix("_scale_inv")
        # the experts prefix each slot with the projection; a dense linear, whose stem IS
        # `weight`, names them bare
        proj = "" if stem == "weight" else f"{stem}_"
        out = {}
        slot = f"{proj}activation_scale"
        held = getattr(module, slot, None)
        if held is not None:
            # on the weight's device, not the slot's: the module is still on meta here, and
            # `ones_like` would inherit that and leave the parameter unmaterialized
            out[prefix.removesuffix(stem) + slot] = torch.ones(held.shape, dtype=held.dtype, device=device)
        return out

    @staticmethod
    def _quantize_block_fp8(value: torch.Tensor, block: tuple[int, int], ue8m0: bool):
        """``(E4M3 weight, inverse scale grid)`` for a ``(..., rows, cols)`` tensor at ``block``.

        A trailing PARTIAL block is padded, not refused — DeepSeek-V3 ships `kv_a_proj_with_mqa`
        as `(576, 7168)` against a 128x128 block with a `(5, 56)` grid, so the format's own
        producers round up and the module allocates to match. The zeros cannot move a block's
        amax, so the short block is scaled by its real values. UE8M0 rounds the inverse scale up
        to a power of two before quantizing, so dequant multiplies by exactly what it divided by.
        """
        block_m, block_n = block
        rows, cols = value.shape[-2], value.shape[-1]
        padded = torch.nn.functional.pad(value.float(), (0, -cols % block_n, 0, -rows % block_m))
        grid_m, grid_n = padded.shape[-2] // block_m, padded.shape[-1] // block_n
        tiles = padded.reshape(*padded.shape[:-2], grid_m, block_m, grid_n, block_n)
        max_abs = tiles.abs().amax(dim=(-3, -1))
        inv_scale = torch.where(max_abs > 0, max_abs / _FP8_MAX, torch.ones_like(max_abs))
        if ue8m0:
            inv_scale = torch.pow(2.0, torch.ceil(torch.log2(inv_scale.clamp(min=torch.finfo(torch.float32).tiny))))
        scaled = tiles / inv_scale.unsqueeze(-1).unsqueeze(-3)
        quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE).reshape(padded.shape)
        # a no-op when the shape tiled
        return quantized[..., :rows, :cols].contiguous(), inv_scale.to(_get_ue8m0_dtype() if ue8m0 else torch.float32)

    @staticmethod
    def _quantize_group(value: torch.Tensor, format_spec: WeightFormat):
        """``(weight, scale, global)`` for a ``(..., rows, K)`` tensor in a group-scaled format through the
        kernels' row-wise quantizers, one launch per tensor; ``global`` is ``None`` unless the format has
        one. NVFP4's global is per matrix (per expert for a stack, a scalar for a dense weight): the
        smallest that keeps every E4M3 block scale in range, divided out before the block quant."""
        kernel = load_finegrained_kernel()
        rows = value.reshape(-1, value.shape[-1])
        global_scale = None
        if format_spec.global_scale_dtype is not None:
            per_matrix = value.float().reshape(-1, value.shape[-2] * value.shape[-1]).abs().amax(-1) / (6.0 * 448.0)
            global_scale = per_matrix.clamp(min=torch.finfo(torch.float32).tiny)
            rows = (value.float() / global_scale.reshape(-1, *[1] * (value.ndim - 1))).reshape_as(rows)
            packed, scale = kernel.nvfp4_act_quant(rows.contiguous())
            global_scale = global_scale.reshape(()) if value.ndim == 2 else global_scale
        else:
            quantize = kernel.mxfp4_act_quant if format_spec.values_per_byte == 2 else kernel.mxfp8_act_quant
            packed, scale = quantize(rows.to(torch.bfloat16).contiguous())
        return (
            packed.view(format_spec.weight_dtype).reshape(*value.shape[:-1], -1),
            scale.reshape(*value.shape[:-1], -1),
            global_scale,
        )

    @property
    def reverse_op(self) -> ConversionOps:
        return FineGrainedDequantize(self.hf_quantizer)


class FineGrainedDequantize(_FineGrainedOp):
    """A quantized weight folded back to full precision against its per-block scale grid.

    Runs FIRST in its converter under ``dequantize=True``, which is why
    :meth:`update_weight_conversions` attaches it to each of the model's own converters: the
    merge / concat ops after it collapse the per-expert structure the (weight, scale) pairing
    needs. It pairs each weight pattern with its sibling scale pattern by index and emits the
    result under the weight key, dropping the scales so the rest of the chain sees weights only.
    """

    def _scale_pattern_for(self, weight_pattern: str) -> str:
        anchor = "(?=$)" if weight_pattern.endswith("(?=$)") else "$" if weight_pattern.endswith("$") else ""
        base = weight_pattern.removesuffix(anchor) if anchor else weight_pattern
        if base.endswith("_blocks"):
            # GPT-OSS packs its experts as `{proj}_blocks` + `{proj}_scales`
            scale = base[: -len("_blocks")] + "_scales"
        elif base.endswith(".weight"):
            scale = base[: -len(".weight")] + ".weight_scale_inv"
        elif base == "weight":
            scale = "weight_scale_inv"
        else:
            scale = base + "_scale_inv"
        return scale + anchor

    @property
    def _reciprocal_globals(self) -> bool:
        """compressed-tensors ships NVFP4's globals as their reciprocal (see `FineGrainedReciprocal`)."""
        plan = self.hf_quantizer.plan() if hasattr(self.hf_quantizer, "plan") else {}
        return any(config.checkpoint_format == "compressed-tensors" for config in plan.values())

    # the weights a converter hands one at a time: a dense `weight`, or an experts module's fused stack
    _SINGLE_WEIGHTS = ("weight", *FUSED_STACKS)

    @staticmethod
    def _name(pattern: str) -> str:
        """The name a source pattern matches: `(?<!\\w)gate_up_proj$` -> `gate_up_proj`."""
        return pattern.removeprefix(r"(?<!\w)").removesuffix("(?=$)").removesuffix("$")

    def _companion(self, input_dict, pattern: str) -> str | None:
        """The source pattern in `input_dict` matching the same name as `pattern`, if the converter collected one."""
        return next((key for key in input_dict if self._name(key) == self._name(pattern)), None)

    @staticmethod
    def _global_pattern_for(weight_pattern: str) -> str:
        """NVFP4's second-level global beside `weight_pattern`: `x.weight` -> `x.weight_global_scale`, and a fused
        stack's `gate_up_proj` -> `gate_up_proj_weight_global_scale`."""
        anchor = "(?=$)" if weight_pattern.endswith("(?=$)") else "$" if weight_pattern.endswith("$") else ""
        base = weight_pattern.removesuffix(anchor) if anchor else weight_pattern
        suffix = "_global_scale" if base.endswith("weight") else "_weight_global_scale"
        return base + suffix + anchor

    # packed-FP4 experts arrive as int8 / float4_e2m1fn_x2, two e2m1 nibbles per byte
    _FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

    def _unpack_fp4(self, packed: torch.Tensor) -> torch.Tensor:
        """Two ``e2m1`` FP4 values per byte → float32 tensor twice as wide on the last dim."""
        lut = torch.tensor(self._FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
        u8 = packed.contiguous().view(torch.uint8)
        low = (u8 & 0xF).long()
        high = ((u8 >> 4) & 0xF).long()
        unpacked = torch.stack([lut[low], lut[high]], dim=-1)
        return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])

    def _dequantize_one(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        output_dtype: torch.dtype | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # unpacked first, so the rest of the routine sees a normal (rows, cols) float matrix
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        # modelopt ships its packed bytes as uint8
        if quantized.dtype in (torch.int8, torch.uint8) or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        rows, cols = quantized_fp32.shape[-2:]
        # the block comes from the scale grid, not the config: one checkpoint holds both a
        # ``[1, 32]`` MXFP4 expert block and a ``[128, 128]`` FP8 dense one
        # a per-tensor scale has no grid: one block over the whole matrix
        scale_rows, scale_cols = scales.shape[-2:] if scales.ndim >= 2 else (1, 1)
        # the grid ROUNDS UP, so it does not invert to the block (DSv3's `(576, 7168)` ships a
        # `(5, 56)` grid against a 128x128 block); only guess when no block the plan names reproduces it
        plan = self.hf_quantizer.plan() if hasattr(self.hf_quantizer, "plan") else {}
        configured = next(
            (
                block
                for block in {tuple(config.weight_block_size) for config in plan.values() if config.weight_block_size}
                if (_cdiv(rows, block[0]), _cdiv(cols, block[1])) == (scale_rows, scale_cols)
            ),
            None,
        )
        if configured:
            block_m, block_n = configured
        elif rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
            )
        else:
            block_m = rows // scale_rows
            block_n = cols // scale_cols
        # the math runs in fp32 either way (``float8_e8m0fnu`` has no ``mul`` kernel); the
        # destination parameter's dtype wins when known, so an eager module keeps the model's
        if output_dtype is None:
            output_dtype = (
                scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
            )
        # an MXFP8 checkpoint's uint8 scale is a biased E8M0 exponent, not a multiplier
        if scales.dtype == torch.uint8:
            s_fp32 = (scales.to(torch.float32) - 127.0).exp2()
        else:
            s_fp32 = scales.to(torch.float32)
        original_shape = quantized_fp32.shape
        # pad out to whole blocks so a ceil-rounded grid lines up, then crop back
        pad_m, pad_n = -rows % block_m, -cols % block_n
        padded = torch.nn.functional.pad(quantized_fp32, (0, pad_n, 0, pad_m))
        blocked = padded.reshape(-1, scale_rows, block_m, scale_cols, block_n)
        per_block = s_fp32.reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
        out = (blocked * per_block).reshape(*original_shape[:-2], rows + pad_m, cols + pad_n)[..., :rows, :cols]
        if global_scale is not None:
            global_scale = global_scale.to(torch.float32)
            if out.ndim > 2 and global_scale.ndim:
                # a stack's globals, one per expert (E,) or one per stacked gate|up half (E, 2), over its rows
                global_scale = global_scale.reshape(global_scale.shape[0], -1)
                global_scale = global_scale.repeat_interleave(rows // global_scale.shape[1], dim=1)[..., None]
            out = out / global_scale if self._reciprocal_globals else out * global_scale
        return out.to(output_dtype)

    def _get_target_dtype(self, model: torch.nn.Module | None, full_layer_name: str | None) -> torch.dtype | None:
        if model is None or full_layer_name is None:
            return None
        module, tensor_name = get_module_from_name(model, full_layer_name)
        param = getattr(module, tensor_name, None)
        return getattr(param, "dtype", None)

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        output_dtype = self._get_target_dtype(model, full_layer_name)
        # The dense converter (``["weight$", companions...] -> weight``) and a fused stack's (``["gate_up_proj$", its
        # companions...] -> gate_up_proj``) each hand one weight, with or without a scale (RMSNorm weights match
        # ``weight$`` too).
        weight_pattern = next((p for p in input_dict if self._name(p) in self._SINGLE_WEIGHTS), None)
        if weight_pattern is not None:
            # the loader derives prefix/suffix from the output KEY; without a `full_layer_name`
            # (direct invocation) key it by the weight's own name
            target_key = full_layer_name if full_layer_name is not None else self._name(weight_pattern)
            quantized = single_tensor(input_dict[weight_pattern])
            scale, global_scale = (
                single_tensor(input_dict.get(self._companion(input_dict, companion(weight_pattern))))
                for companion in (self._scale_pattern_for, self._global_pattern_for)
            )
            if scale is None:
                return {target_key: quantized}
            return {target_key: self._dequantize_one(quantized, scale, output_dtype, global_scale)}

        # Generic chain path: dequantize every weight pattern that has a sibling scale.
        consumed = {
            pattern(key) for key in input_dict for pattern in (self._scale_pattern_for, self._global_pattern_for)
        }
        result: dict[str, list[torch.Tensor] | torch.Tensor] = {}
        for key, value in input_dict.items():
            if "activation_scale" in key or key in consumed:
                continue  # consumed by the dequant; drop from the chain
            scale_key = self._scale_pattern_for(key)
            if scale_key not in input_dict:
                result[key] = value
                continue
            weights = value if isinstance(value, list) else [value]
            scales = input_dict[scale_key]
            scales = scales if isinstance(scales, list) else [scales]
            if len(weights) != len(scales):
                raise ValueError(
                    f"FineGrainedDequantize: weight/scale count mismatch for {key} "
                    f"({len(weights)} weights vs {len(scales)} scales)."
                )
            global_scales = input_dict.get(self._global_pattern_for(key), [None] * len(weights))
            global_scales = global_scales if isinstance(global_scales, list) else [global_scales]
            result[key] = [
                self._dequantize_one(w, s, output_dtype, g) for w, s, g in zip(weights, scales, global_scales)
            ]
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        # a dequantized model saves in full precision: none of its weights fills a finegrained module,
        # so the quantize passes each through
        return FineGrainedQuantize(self.hf_quantizer)
