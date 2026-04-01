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

"""Fusion registration helpers.

See `docs/source/en/fusion_mapping.md` for the design overview and extension guide.
"""

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from .core_model_loading import Conv3dToLinear, WeightConverter, WeightRenaming, WeightTransform
from .monkey_patching import register_patch_mapping
from .utils import logging


if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

_FUSION_DISCOVERY_CACHE: dict[str, dict[type, tuple[dict[str, type[nn.Module]], list[WeightTransform]]]] = {}


class ModuleFusionSpec:
    """Base recipe for a fusion family.

    A fusion spec decides which modules are eligible for a fusion, how to build
    the runtime replacement class, and which weight transforms are needed to map
    checkpoints between the original and fused layouts.
    """

    def get_empty_log(self, model_name: str) -> str:
        return f"No compatible {type(self).__name__} classes found to fuse for {model_name}"

    def is_fusable(self, module: nn.Module) -> bool:
        raise NotImplementedError

    def make_fused_class(self, original_cls: type[nn.Module], reference_module: nn.Module) -> type[nn.Module]:
        raise NotImplementedError

    def make_transforms(self, module_name: str, reference_module: nn.Module) -> list[WeightTransform]:
        raise NotImplementedError


class PatchEmbeddingsFusionSpec(ModuleFusionSpec):
    """Fusion spec for Conv3d patch embeddings that can be flattened into Linear layers."""

    def is_fusable(self, module: nn.Module) -> bool:
        if not isinstance(proj := getattr(module, "proj", None), nn.Conv3d):
            return False

        # no overlap between the patches
        return (
            proj.stride == proj.kernel_size
            and proj.padding == (0, 0, 0)
            and proj.dilation == (1, 1, 1)
            and proj.groups == 1
        )

    def make_fused_class(self, original_cls: type[nn.Module], reference_module: nn.Module) -> type[nn.Module]:
        # patch_volume = in_channels * temporal_patch_size * patch_height * patch_width
        patch_volume = reference_module.proj.in_channels * math.prod(reference_module.proj.kernel_size)

        class FusedPatchEmbedding(original_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.linear_proj = nn.Linear(
                    patch_volume,
                    self.proj.out_channels,
                    bias=self.proj.bias is not None,  # might not be used at all?
                    device=self.proj.weight.device,
                    dtype=self.proj.weight.dtype,
                )

                del self.proj

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                target_dtype = self.linear_proj.weight.dtype
                hidden_states = hidden_states.view(-1, patch_volume)
                hidden_states = self.linear_proj(hidden_states.to(dtype=target_dtype))
                return hidden_states.view(-1, self.embed_dim)

        FusedPatchEmbedding.__name__ = f"Fused{original_cls.__name__}"
        FusedPatchEmbedding.__qualname__ = f"Fused{original_cls.__qualname__}"
        return FusedPatchEmbedding

    def make_transforms(self, module_name: str, reference_module: nn.Module) -> list[WeightTransform]:
        source_weight_name = f"{module_name}.proj.weight"
        target_weight_name = f"{module_name}.linear_proj.weight"
        converters = [
            WeightConverter(
                source_patterns=source_weight_name,
                target_patterns=target_weight_name,
                operations=[
                    Conv3dToLinear(
                        in_channels=reference_module.proj.in_channels,
                        kernel_size=tuple(reference_module.proj.kernel_size),
                    )
                ],
            )
        ]
        if reference_module.proj.bias is not None:
            converters.append(
                WeightRenaming(
                    source_patterns=f"{module_name}.proj.bias",
                    target_patterns=f"{module_name}.linear_proj.bias",
                )
            )
        return converters


def _discover_fusable_modules(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    fusion_name: str,
    is_fusable: Callable[[nn.Module], bool],
    make_fused_class: Callable[[type[nn.Module], nn.Module], type[nn.Module]],
    make_transforms: Callable[[str, nn.Module], list[WeightTransform]],
) -> tuple[dict[str, type[nn.Module]], list[WeightTransform]]:
    """Meta-initialize `cls` and collect patch mappings plus weight transforms for one fusion family."""

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
        converters.extend(make_transforms(module_name, module))
        if module_cls in seen_classes:
            continue

        seen_classes.add(module_cls)
        patch_mapping[module_cls.__name__] = make_fused_class(module_cls, module)

    cache[cls] = (patch_mapping, converters)
    return patch_mapping, converters


def _register_module_fusion(
    cls: "type[PreTrainedModel]", config: "PretrainedConfig", fusion_name: str, spec: ModuleFusionSpec
) -> None:
    """Register one fusion family for `cls`.

    This wires together two parts of the loading stack:
    - monkey patching, so compatible module classes are replaced before model initialization
    - checkpoint conversion mapping, so fused runtime modules still load from the original checkpoint layout
    """

    fusable_classes, converters = _discover_fusable_modules(
        cls,
        config,
        fusion_name=fusion_name,
        is_fusable=spec.is_fusable,
        make_fused_class=spec.make_fused_class,
        make_transforms=spec.make_transforms,
    )
    if not fusable_classes:
        logger.info(spec.get_empty_log(cls.__name__))
        return

    register_patch_mapping(fusable_classes, overwrite=True)

    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config class or model type")
    model_type = cls.config_class.model_type

    existing_converters = get_checkpoint_conversion_mapping(model_type)
    if existing_converters is not None:
        # WeightConverter matching stops at the first matching source pattern, so
        # conflicting converters must fail fast instead of being appended.
        existing_converter_sources = {tuple(existing.source_patterns): existing for existing in existing_converters}
        for converter in converters:
            source_patterns = tuple(converter.source_patterns)
            existing_converter = existing_converter_sources.get(source_patterns)
            if existing_converter is None:
                continue

            if type(existing_converter) is not type(converter) or tuple(existing_converter.target_patterns) != tuple(
                converter.target_patterns
            ):
                raise ValueError(
                    f"Fusion {fusion_name} for model type {model_type} conflicts with an existing conversion mapping "
                    f"for source patterns {source_patterns}."
                )

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


_FUSION_REGISTRY: dict[str, ModuleFusionSpec] = {"patch_embeddings": PatchEmbeddingsFusionSpec()}


def _iter_enabled_fusions(fusion_config: Mapping[str, bool | Mapping[str, Any]]) -> list[str]:
    """Validate `fusion_config` and return the enabled fusion names in user-specified order."""

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
    """Register requested runtime fusions for `cls`.

    `fusion_config` is validated against `_FUSION_REGISTRY`, so adding a new
    fusion type is a matter of registering a new `ModuleFusionSpec` here.
    """

    if not fusion_config:
        return

    for fusion_name in _iter_enabled_fusions(fusion_config):
        _register_module_fusion(cls, config, fusion_name, _FUSION_REGISTRY[fusion_name])
