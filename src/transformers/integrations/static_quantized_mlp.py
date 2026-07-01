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
from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from ..activations import ACT2FN
from ..utils import logging
from .finegrained_fp8 import _FP8_DTYPE, FP8Linear
from .hub_kernels import get_kernel
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)

_GATED_MLP_PATTERN = "gated_mlp"
_DENSE_GELU_MLP_PATTERN = "dense_gelu_mlp"
_INPUT_QUANT_KEY = "input_quant"
_SUPPORTED_PATTERN_KEYS = {_GATED_MLP_PATTERN, _DENSE_GELU_MLP_PATTERN, _INPUT_QUANT_KEY}


@dataclass(frozen=True)
class HubKernelEndpoint:
    repo_id: str
    version: int | str | None = 1
    revision: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class StaticQuantizedMLPInputQuantSpec:
    kernel: HubKernelEndpoint
    func: str = "channel_scale_quantize_fp8_static_bf16"


@dataclass(frozen=True)
class StaticQuantizedGatedMLPSpec:
    kernel: HubKernelEndpoint
    swiglu_func: str = "fp8_swiglu_mlp_bf16"
    geglu_func: str = "fp8_geglu_mlp_bf16"

    @property
    def required_functions(self) -> tuple[str, str]:
        return (self.swiglu_func, self.geglu_func)


@dataclass(frozen=True)
class StaticQuantizedDenseGELUMLPSpec:
    kernel: HubKernelEndpoint
    gelu_func: str = "fp8_gelu_mlp_bf16"

    @property
    def required_functions(self) -> tuple[str]:
        return (self.gelu_func,)


@dataclass(frozen=True)
class StaticQuantizedMLPFusionSpec:
    input_quant: StaticQuantizedMLPInputQuantSpec
    gated_mlp: StaticQuantizedGatedMLPSpec | None = None
    dense_gelu_mlp: StaticQuantizedDenseGELUMLPSpec | None = None


def _endpoint_from_options(
    options: str | Mapping[str, Any],
    *,
    pattern_name: str,
) -> HubKernelEndpoint:
    if isinstance(options, str):
        return HubKernelEndpoint(repo_id=options)
    if not isinstance(options, Mapping):
        raise ValueError(f"Static quantized MLP kernel options for {pattern_name!r} must be a repo string or mapping.")

    repo_id = options.get("repo_id")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError(
            f"Expected a valid Hub repo id for static quantized MLP fusion option {pattern_name!r}, got {repo_id!r}"
        )

    trust_remote_code = bool(options.get("trust_remote_code", False))
    return HubKernelEndpoint(
        repo_id=repo_id,
        version=options.get("version", 1),
        revision=options.get("revision", None),
        trust_remote_code=trust_remote_code,
    )


def _input_quant_spec_from_options(options: Mapping[str, Any]) -> StaticQuantizedMLPInputQuantSpec:
    if _INPUT_QUANT_KEY not in options:
        raise ValueError("Static quantized MLP fusion requires an explicit 'input_quant' kernel configuration.")
    input_quant_options = options[_INPUT_QUANT_KEY]
    endpoint = _endpoint_from_options(input_quant_options, pattern_name=_INPUT_QUANT_KEY)
    func = "channel_scale_quantize_fp8_static_bf16"
    if isinstance(input_quant_options, Mapping):
        func = input_quant_options.get("func", func)
    return StaticQuantizedMLPInputQuantSpec(kernel=endpoint, func=func)


def _gated_spec_from_options(options: str | Mapping[str, Any]) -> StaticQuantizedGatedMLPSpec:
    endpoint = _endpoint_from_options(options, pattern_name=_GATED_MLP_PATTERN)
    swiglu_func = "fp8_swiglu_mlp_bf16"
    geglu_func = "fp8_geglu_mlp_bf16"
    if isinstance(options, Mapping):
        swiglu_func = options.get("swiglu_func", swiglu_func)
        geglu_func = options.get("geglu_func", geglu_func)
    return StaticQuantizedGatedMLPSpec(kernel=endpoint, swiglu_func=swiglu_func, geglu_func=geglu_func)


def _dense_gelu_spec_from_options(options: str | Mapping[str, Any]) -> StaticQuantizedDenseGELUMLPSpec:
    endpoint = _endpoint_from_options(options, pattern_name=_DENSE_GELU_MLP_PATTERN)
    gelu_func = "fp8_gelu_mlp_bf16"
    if isinstance(options, Mapping):
        gelu_func = options.get("gelu_func", gelu_func)
    return StaticQuantizedDenseGELUMLPSpec(kernel=endpoint, gelu_func=gelu_func)


def get_static_quantized_mlp_fusion_spec(
    options: bool | Mapping[str, Any],
) -> StaticQuantizedMLPFusionSpec | None:
    if options is False:
        return None

    if options is True:
        raise ValueError("Static quantized MLP fusion requires explicit Hub kernel configurations.")
    if not isinstance(options, Mapping):
        raise ValueError(f"Static quantized MLP fusion config must be True, False, or a mapping, got {type(options)}")

    unknown_keys = set(options) - _SUPPORTED_PATTERN_KEYS
    if unknown_keys:
        raise ValueError(
            "Unknown static quantized MLP fusion option(s): "
            f"{sorted(unknown_keys)}. Supported options are {sorted(_SUPPORTED_PATTERN_KEYS)}."
        )

    gated_mlp = None
    dense_gelu_mlp = None
    if options.get(_GATED_MLP_PATTERN, False):
        gated_mlp = _gated_spec_from_options(options[_GATED_MLP_PATTERN])
    if options.get(_DENSE_GELU_MLP_PATTERN, False):
        dense_gelu_mlp = _dense_gelu_spec_from_options(options[_DENSE_GELU_MLP_PATTERN])

    if gated_mlp is None and dense_gelu_mlp is None:
        return None

    return StaticQuantizedMLPFusionSpec(
        input_quant=_input_quant_spec_from_options(options),
        gated_mlp=gated_mlp,
        dense_gelu_mlp=dense_gelu_mlp,
    )


@functools.cache
def _load_hub_kernel(
    repo_id: str,
    version: int | str | None,
    revision: str | None,
    trust_remote_code: bool,
) -> ModuleType:
    return get_kernel(repo_id, version=version, revision=revision, allow_all_kernels=trust_remote_code)


def _load_endpoint(endpoint: HubKernelEndpoint) -> ModuleType:
    return _load_hub_kernel(endpoint.repo_id, endpoint.version, endpoint.revision, endpoint.trust_remote_code)


def _is_scalar_float_scale(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and tensor.numel() == 1 and tensor.dtype == torch.float32


def _same_scalar_scale(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device.type == "meta" or rhs.device.type == "meta":
        return True
    return bool(torch.equal(lhs.detach().float().cpu(), rhs.detach().float().cpu()))


def _is_supported_static_fp8_linear(module: nn.Module, *, allow_bias: bool) -> bool:
    if not isinstance(module, FP8Linear):
        return False
    if module.activation_scheme != "static" or module.block_size is not None:
        return False
    if module.weight.dtype != _FP8_DTYPE:
        return False
    if not allow_bias and module.bias is not None:
        return False
    return _is_scalar_float_scale(module.weight_scale_inv) and _is_scalar_float_scale(module.activation_scale)


def _activation_matches(module: nn.Module, *activation_names: str) -> bool:
    activation = getattr(module, "act_fn", None)
    if activation is None:
        activation = getattr(module, "activation_fn", None)
    if activation is None:
        activation = getattr(module, "activation", None)
    if activation is None:
        return False

    for activation_name in activation_names:
        expected = ACT2FN[activation_name]
        if activation is expected or activation.__class__ is expected.__class__:
            return True
    return False


def _gated_activation_name(module: nn.Module) -> str | None:
    if _activation_matches(module, "silu"):
        return "silu"
    if _activation_matches(module, "gelu_pytorch_tanh"):
        return "gelu_pytorch_tanh"
    return None


class StaticFP8GatedMLP(nn.Module):
    """Runtime fused static-FP8 gated MLP backed by Hub kernels."""

    def __init__(
        self,
        original_mlp: nn.Module,
        input_quant_kernel: ModuleType,
        input_quant_spec: StaticQuantizedMLPInputQuantSpec,
        mlp_kernel: ModuleType,
        mlp_spec: StaticQuantizedGatedMLPSpec,
        activation_name: str,
    ):
        super().__init__()
        self.input_quant_kernel = input_quant_kernel
        self.input_quant_spec = input_quant_spec
        self.mlp_kernel = mlp_kernel
        self.mlp_spec = mlp_spec
        self.activation_name = activation_name

        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

        self.register_buffer(
            "gate_up_weight",
            torch.cat([self.gate_proj.weight.detach(), self.up_proj.weight.detach()], dim=0).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "input_channel_scale",
            torch.ones(self.gate_proj.in_features, device=self.gate_proj.weight.device, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        x = input.reshape(-1, input_shape[-1]).to(torch.bfloat16)
        quant_fn = getattr(self.input_quant_kernel, self.input_quant_spec.func)
        x_fp8 = quant_fn(
            x,
            to_local(self.input_channel_scale),
            to_local(self.gate_proj.activation_scale),
        )

        mlp_fn_name = self.mlp_spec.swiglu_func if self.activation_name == "silu" else self.mlp_spec.geglu_func
        mlp_fn = getattr(self.mlp_kernel, mlp_fn_name)
        output = mlp_fn(
            x_fp8,
            to_local(self.gate_up_weight),
            to_local(self.down_proj.weight),
            to_local(self.gate_proj.activation_scale),
            to_local(self.gate_proj.weight_scale_inv),
            to_local(self.down_proj.activation_scale),
            to_local(self.down_proj.weight_scale_inv),
        )
        return output.reshape(*input_shape[:-1], output.shape[-1]).to(input.dtype)


class StaticFP8DenseGELUMLP(nn.Module):
    """Runtime fused static-FP8 dense GELU MLP backed by Hub kernels."""

    def __init__(
        self,
        original_mlp: nn.Module,
        input_quant_kernel: ModuleType,
        input_quant_spec: StaticQuantizedMLPInputQuantSpec,
        mlp_kernel: ModuleType,
        mlp_spec: StaticQuantizedDenseGELUMLPSpec,
    ):
        super().__init__()
        self.input_quant_kernel = input_quant_kernel
        self.input_quant_spec = input_quant_spec
        self.mlp_kernel = mlp_kernel
        self.mlp_spec = mlp_spec

        self.fc1 = original_mlp.fc1
        self.fc2 = original_mlp.fc2

        fc1_bias = (
            self.fc1.bias.detach()
            if self.fc1.bias is not None
            else torch.zeros(self.fc1.out_features, device=self.fc1.weight.device)
        )
        fc2_bias = (
            self.fc2.bias.detach()
            if self.fc2.bias is not None
            else torch.zeros(self.fc2.out_features, device=self.fc2.weight.device)
        )
        self.register_buffer("fc1_bias_bf16", fc1_bias.to(torch.bfloat16), persistent=False)
        self.register_buffer("fc2_bias_bf16", fc2_bias.to(torch.bfloat16), persistent=False)
        self.register_buffer(
            "input_channel_scale",
            torch.ones(self.fc1.in_features, device=self.fc1.weight.device, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        x = input.reshape(-1, input_shape[-1]).to(torch.bfloat16)
        quant_fn = getattr(self.input_quant_kernel, self.input_quant_spec.func)
        x_fp8 = quant_fn(
            x,
            to_local(self.input_channel_scale),
            to_local(self.fc1.activation_scale),
        )

        mlp_fn = getattr(self.mlp_kernel, self.mlp_spec.gelu_func)
        output = mlp_fn(
            x_fp8,
            to_local(self.fc1.weight),
            to_local(self.fc1_bias_bf16),
            to_local(self.fc2.weight),
            to_local(self.fc2_bias_bf16),
            to_local(self.fc1.activation_scale),
            to_local(self.fc1.weight_scale_inv),
            to_local(self.fc2.activation_scale),
            to_local(self.fc2.weight_scale_inv),
        )
        return output.reshape(*input_shape[:-1], output.shape[-1]).to(input.dtype)


def _can_fuse_static_fp8_gated_mlp(module: nn.Module) -> tuple[bool, str | None]:
    if not all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return False, None

    gate_proj = module.gate_proj
    up_proj = module.up_proj
    down_proj = module.down_proj
    if not all(_is_supported_static_fp8_linear(proj, allow_bias=False) for proj in (gate_proj, up_proj, down_proj)):
        return False, None

    if gate_proj.in_features != up_proj.in_features or gate_proj.out_features != up_proj.out_features:
        return False, None
    if down_proj.in_features != gate_proj.out_features:
        return False, None
    if not _same_scalar_scale(gate_proj.activation_scale, up_proj.activation_scale):
        return False, None
    if not _same_scalar_scale(gate_proj.weight_scale_inv, up_proj.weight_scale_inv):
        return False, None

    activation_name = _gated_activation_name(module)
    return activation_name is not None, activation_name


def _can_fuse_static_fp8_dense_gelu_mlp(module: nn.Module) -> bool:
    if not all(hasattr(module, name) for name in ("fc1", "fc2")):
        return False

    fc1 = module.fc1
    fc2 = module.fc2
    if not all(_is_supported_static_fp8_linear(proj, allow_bias=True) for proj in (fc1, fc2)):
        return False
    if fc2.in_features != fc1.out_features:
        return False
    return _activation_matches(module, "gelu_pytorch_tanh")


def _validate_kernel_functions(kernel: ModuleType, repo_id: str, function_names: tuple[str, ...]) -> bool:
    missing = [name for name in function_names if not hasattr(kernel, name)]
    if missing:
        logger.warning_once(
            f"Static quantized MLP Hub kernel {repo_id!r} is missing required function(s) {missing}; "
            "skipping this fusion pattern."
        )
        return False
    return True


def replace_with_static_quantized_mlp(
    model: nn.Module,
    options: bool | Mapping[str, Any],
) -> nn.Module:
    fusion_spec = get_static_quantized_mlp_fusion_spec(options)
    if fusion_spec is None:
        return model

    try:
        input_quant_kernel = _load_endpoint(fusion_spec.input_quant.kernel)
    except Exception as e:
        logger.warning_once(
            "Unable to load static quantized MLP input-quant Hub kernel "
            f"{fusion_spec.input_quant.kernel.repo_id!r}; keeping existing MLP modules. {e}"
        )
        return model

    if not _validate_kernel_functions(
        input_quant_kernel,
        fusion_spec.input_quant.kernel.repo_id,
        (fusion_spec.input_quant.func,),
    ):
        return model

    gated_kernel = None
    if fusion_spec.gated_mlp is not None:
        try:
            gated_kernel = _load_endpoint(fusion_spec.gated_mlp.kernel)
        except Exception as e:
            logger.warning_once(
                f"Unable to load static quantized gated MLP Hub kernel "
                f"{fusion_spec.gated_mlp.kernel.repo_id!r}; skipping gated MLP fusion. {e}"
            )
            gated_kernel = None
        if gated_kernel is not None and not _validate_kernel_functions(
            gated_kernel,
            fusion_spec.gated_mlp.kernel.repo_id,
            fusion_spec.gated_mlp.required_functions,
        ):
            gated_kernel = None

    dense_kernel = None
    if fusion_spec.dense_gelu_mlp is not None:
        try:
            dense_kernel = _load_endpoint(fusion_spec.dense_gelu_mlp.kernel)
        except Exception as e:
            logger.warning_once(
                f"Unable to load static quantized dense GELU MLP Hub kernel "
                f"{fusion_spec.dense_gelu_mlp.kernel.repo_id!r}; skipping dense GELU MLP fusion. {e}"
            )
            dense_kernel = None
        if dense_kernel is not None and not _validate_kernel_functions(
            dense_kernel,
            fusion_spec.dense_gelu_mlp.kernel.repo_id,
            fusion_spec.dense_gelu_mlp.required_functions,
        ):
            dense_kernel = None

    replaced_gated = 0
    replaced_dense_gelu = 0
    for module_name, module in list(model.named_modules()):
        if gated_kernel is not None and fusion_spec.gated_mlp is not None:
            can_fuse, activation_name = _can_fuse_static_fp8_gated_mlp(module)
            if can_fuse:
                model.set_submodule(
                    module_name,
                    StaticFP8GatedMLP(
                        module,
                        input_quant_kernel,
                        fusion_spec.input_quant,
                        gated_kernel,
                        fusion_spec.gated_mlp,
                        activation_name,
                    ),
                )
                replaced_gated += 1
                continue

        if dense_kernel is not None and fusion_spec.dense_gelu_mlp is not None:
            if _can_fuse_static_fp8_dense_gelu_mlp(module):
                model.set_submodule(
                    module_name,
                    StaticFP8DenseGELUMLP(
                        module,
                        input_quant_kernel,
                        fusion_spec.input_quant,
                        dense_kernel,
                        fusion_spec.dense_gelu_mlp,
                    ),
                )
                replaced_dense_gelu += 1

    if fusion_spec.gated_mlp is not None and gated_kernel is not None and replaced_gated == 0:
        logger.warning_once("Static quantized gated MLP fusion was enabled, but no eligible modules were found.")
    if fusion_spec.dense_gelu_mlp is not None and dense_kernel is not None and replaced_dense_gelu == 0:
        logger.warning_once("Static quantized dense GELU MLP fusion was enabled, but no eligible modules were found.")

    if replaced_gated or replaced_dense_gelu:
        logger.info(
            "Replaced static quantized MLP module(s) with fused Hub kernels: "
            f"{replaced_gated} gated, {replaced_dense_gelu} dense GELU."
        )

    return model
