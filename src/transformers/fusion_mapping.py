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

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from .core_model_loading import Conv3dToLinear, WeightConverter
from .monkey_patching import register_patch_mapping
from .utils import logging


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

_FUSION_DISCOVERY_CACHE: dict[str, dict[type, tuple[dict[str, type[nn.Module]], list[WeightConverter]]]] = {}


@dataclass(frozen=True)
class ModuleFusionSpec:
    is_fusable: Callable[[nn.Module], bool]
    make_fused_class: Callable[[type[nn.Module], nn.Module], type[nn.Module]]
    make_weight_converter: Callable[[str, nn.Module], WeightConverter]
    empty_log_message: str


def _is_fusable_patch_embedding(module: nn.Module) -> bool:
    proj = getattr(module, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        return False

    return (
        proj.stride == proj.kernel_size
        and proj.padding == (0, 0, 0)
        and proj.dilation == (1, 1, 1)
        and proj.groups == 1
    )


def _make_fused_patch_embedding_class(original_cls: type[nn.Module], reference_module: nn.Module) -> type[nn.Module]:
    reference_proj = reference_module.proj

    # patch_volume = in_channels * temporal_patch_size * patch_height * patch_width
    patch_volume = reference_proj.in_channels * math.prod(reference_proj.kernel_size)

    class FusedPatchEmbedding(original_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            proj = self.proj
            linear_proj = nn.Linear(patch_volume, proj.out_channels, bias=proj.bias is not None)
            linear_proj = linear_proj.to(device=proj.weight.device, dtype=proj.weight.dtype)
            self.proj = linear_proj

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            target_dtype = self.proj.weight.dtype
            hidden_states = hidden_states.view(-1, patch_volume)
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
            return hidden_states.view(-1, self.embed_dim)

    FusedPatchEmbedding.__name__ = f"Fused{original_cls.__name__}"
    FusedPatchEmbedding.__qualname__ = f"Fused{original_cls.__qualname__}"
    return FusedPatchEmbedding


def _build_weight_converter_for_module(module_name: str, reference_module: nn.Module) -> WeightConverter:
    weight_name = f"{module_name}.proj.weight"
    proj = reference_module.proj
    return WeightConverter(
        source_patterns=weight_name,
        target_patterns=weight_name,
        operations=[Conv3dToLinear(in_channels=proj.in_channels, kernel_size=tuple(proj.kernel_size))],
    )


def _discover_fusable_modules(
    cls: "type[PreTrainedModel]",
    config,
    fusion_name: str,
    is_fusable: Callable[[nn.Module], bool],
    make_fused_class: Callable[[type[nn.Module], nn.Module], type[nn.Module]],
    make_weight_converter: Callable[[str, nn.Module], WeightConverter],
) -> tuple[dict[str, type[nn.Module]], list[WeightConverter]]:
    cache = _FUSION_DISCOVERY_CACHE.setdefault(fusion_name, {})
    if cls in cache:
        return cache[cls]

    with torch.device("meta"):
        model = cls(config)

    seen_classes = set()
    patch_mapping = {}
    converters = []
    for module_name, module in model.named_modules():
        if not is_fusable(module):
            continue

        module_cls = type(module)
        converters.append(make_weight_converter(module_name, module))
        if module_cls in seen_classes:
            continue

        seen_classes.add(module_cls)
        patch_mapping[module_cls.__name__] = make_fused_class(module_cls, module)

    cache[cls] = (patch_mapping, converters)
    return patch_mapping, converters


def _register_module_fusion(cls: "type[PreTrainedModel]", config, fusion_name: str, spec: ModuleFusionSpec) -> None:
    fusable_classes, converters = _discover_fusable_modules(
        cls,
        config,
        fusion_name=fusion_name,
        is_fusable=spec.is_fusable,
        make_fused_class=spec.make_fused_class,
        make_weight_converter=spec.make_weight_converter,
    )
    if not fusable_classes:
        logger.debug(spec.empty_log_message, cls.__name__)
        return

    register_patch_mapping(fusable_classes, overwrite=True)

    config_class = getattr(cls, "config_class", None)
    model_type = getattr(config_class, "model_type", None) if config_class is not None else None
    if model_type is None:
        return

    existing_converters = get_checkpoint_conversion_mapping(model_type)
    if existing_converters is not None:
        existing_converter_keys = {
            (tuple(existing.source_patterns), tuple(existing.target_patterns), type(existing))
            for existing in existing_converters
        }
        converters = existing_converters + [
            converter
            for converter in converters
            if (tuple(converter.source_patterns), tuple(converter.target_patterns), type(converter))
            not in existing_converter_keys
        ]

    register_checkpoint_conversion_mapping(model_type, converters, overwrite=True)


_FUSION_REGISTRY: dict[str, ModuleFusionSpec] = {
    "patch_embeddings": ModuleFusionSpec(
        is_fusable=_is_fusable_patch_embedding,
        make_fused_class=_make_fused_patch_embedding_class,
        make_weight_converter=_build_weight_converter_for_module,
        empty_log_message="No compatible patch-embedding classes found to fuse for %s",
    )
}


def _iter_enabled_fusions(fusion_config: Mapping[str, bool | Mapping[str, Any]]) -> list[str]:
    enabled_fusions = []
    for fusion_name, fusion_options in fusion_config.items():
        if fusion_name not in _FUSION_REGISTRY:
            raise ValueError(f"Unknown fusion type: {fusion_name}")
        if fusion_options is False:
            continue
        if fusion_options is not True and not isinstance(fusion_options, Mapping):
            raise ValueError(
                f"Invalid fusion config for {fusion_name}: expected `True`, `False`, or a mapping of options."
            )
        enabled_fusions.append(fusion_name)
    return enabled_fusions


def register_fusion_patches(
    cls: "type[PreTrainedModel]", config, fusion_config: Mapping[str, bool | Mapping[str, Any]] | None = None
) -> None:
    if not fusion_config:
        return

    for fusion_name in _iter_enabled_fusions(fusion_config):
        _register_module_fusion(cls, config, fusion_name, _FUSION_REGISTRY[fusion_name])
