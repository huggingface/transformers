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

"""Fusion helpers.

See `docs/source/en/fusion_mapping.md` for the design overview and extension guide.
"""

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .core_model_loading import Conv3dToLinear, WeightConverter, WeightRenaming, WeightTransform
from .utils import logging


if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

_FUSION_DISCOVERY_CACHE: dict[str, dict[type, dict[str, type[nn.Module]]]] = {}


class ModuleFusionSpec:
    """Base recipe for a fusion family.

    A fusion spec decides which modules are eligible for a fusion, how to build
    the runtime replacement class, and which weight transforms are needed to map
    checkpoints between the original and fused layouts.
    """

    target_modules_patterns: tuple[str, ...] = ()

    def get_empty_log(self, model_name: str) -> str:
        """Return the log message emitted when no compatible modules are found."""
        return f"No compatible {type(self).__name__} classes found to fuse for {model_name}"

    def is_fusable(self, module: nn.Module) -> bool:
        """Return whether `module` is compatible with this fusion family."""
        raise NotImplementedError

    def make_fused_class(self, original_cls: type[nn.Module]) -> type[nn.Module]:
        """Build the runtime replacement class for a compatible module class."""
        raise NotImplementedError

    def is_fused(self, module: nn.Module) -> bool:
        """Return whether `module` is an instance of a class built by `make_fused_class`."""
        raise NotImplementedError

    def make_transforms(self, module: nn.Module) -> list[WeightTransform]:
        """Build the weight transforms that load and save the fused `module` in the original checkpoint layout."""
        raise NotImplementedError


class _FusedPatchEmbeddingMixin:
    def __init__(self, *args, **kwargs):
        # call the original_cls.__init__()
        super().__init__(*args, **kwargs)
        self.proj_in_channels = self.proj.in_channels
        self.proj_kernel_size = self.proj.kernel_size
        self.patch_volume = self.proj.in_channels * math.prod(self.proj.kernel_size)

        self.linear_proj = nn.Linear(
            self.patch_volume,
            self.proj.out_channels,
            bias=self.proj.bias is not None,
            device=self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )

        del self.proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.linear_proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.patch_volume)
        hidden_states = self.linear_proj(hidden_states.to(dtype=target_dtype))
        return hidden_states.view(-1, self.embed_dim)


class PatchEmbeddingsFusionSpec(ModuleFusionSpec):
    """Fuse compatible Conv3d patch embeddings into flattened Linear projections."""

    target_modules_patterns = (r"(^|\.)patch_embed$",)

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

    def make_fused_class(self, original_cls: type[nn.Module]) -> type[nn.Module]:
        fused_cls = type(f"Fused{original_cls.__name__}", (_FusedPatchEmbeddingMixin, original_cls), {})
        fused_cls.__qualname__ = f"Fused{original_cls.__qualname__}"
        return fused_cls

    def is_fused(self, module: nn.Module) -> bool:
        return isinstance(module, _FusedPatchEmbeddingMixin)

    def make_transforms(self, module: nn.Module) -> list[WeightTransform]:
        return [
            WeightConverter(
                source_patterns=r"patch_embed\.proj\.weight",
                target_patterns=r"patch_embed\.linear_proj\.weight",
                operations=[
                    Conv3dToLinear(
                        in_channels=module.proj_in_channels,
                        kernel_size=module.proj_kernel_size,
                    )
                ],
            ),
            WeightRenaming(
                source_patterns=r"patch_embed\.proj\.bias",
                target_patterns=r"patch_embed\.linear_proj\.bias",
            ),
        ]


def _discover_fusable_modules(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    fusion_name: str,
    spec: ModuleFusionSpec,
) -> dict[str, type[nn.Module]]:
    """Discover compatible module classes for one fusion family on a meta-initialized model.

    This function:
    - instantiates `cls(config)` on the meta device
    - scans `named_modules()` for candidate modules
    - optionally pre-filters them with `target_modules_patterns`
    - uses `is_fusable(...)` as the final structural check
    - builds the class-level patch mapping used by monkey patching

    Results are cached per `(fusion_name, cls)` to avoid repeated meta-initialization.
    This matches the current class-level fusion behavior, where one compatible
    module class maps to one fused replacement class.
    """

    cache = _FUSION_DISCOVERY_CACHE.setdefault(fusion_name, {})
    if cls in cache:
        return cache[cls]

    with torch.device("meta"):
        model = cls(config)

    seen_classes = set()
    patch_mapping = {}
    target_module_pattern = (
        re.compile("|".join(spec.target_modules_patterns)) if spec.target_modules_patterns else None
    )
    for module_name, module in model.named_modules():
        module_cls = type(module)
        if module_cls in seen_classes:
            continue
        if target_module_pattern is not None and target_module_pattern.search(module_name) is None:
            continue
        if not spec.is_fusable(module):
            continue

        seen_classes.add(module_cls)
        patch_mapping[module_cls.__name__] = spec.make_fused_class(module_cls)

    cache[cls] = patch_mapping
    return patch_mapping


_FUSION_REGISTRY: dict[str, ModuleFusionSpec] = {"patch_embeddings": PatchEmbeddingsFusionSpec()}


def _iter_enabled_fusions(fusion_config: Mapping[str, bool | Mapping[str, Any]]) -> list[str]:
    """Validate `fusion_config` and return enabled fusion names in user-specified order."""

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


def get_fusion_patch_mapping(
    cls: "type[PreTrainedModel]", config: "PretrainedConfig", fusion_config: Mapping[str, bool | Mapping[str, Any]]
) -> dict[str, type[nn.Module]]:
    """Return the module classes to replace while one model of class `cls` is initialized with `fusion_config`."""

    patch_mapping = {}
    for fusion_name in _iter_enabled_fusions(fusion_config):
        spec = _FUSION_REGISTRY[fusion_name]
        fusable_classes = _discover_fusable_modules(cls, config, fusion_name=fusion_name, spec=spec)
        if not fusable_classes:
            logger.info(spec.get_empty_log(cls.__name__))
        patch_mapping.update(fusable_classes)
    return patch_mapping


def get_fusion_weight_transforms(model: nn.Module) -> list[WeightTransform]:
    """Return the weight transforms that load and save the fused modules of `model` in the original checkpoint layout."""

    transforms = {}
    for module in model.modules():
        for spec in _FUSION_REGISTRY.values():
            if spec.is_fused(module):
                for transform in spec.make_transforms(module):
                    transforms.setdefault(tuple(transform.source_patterns), transform)
    return list(transforms.values())
