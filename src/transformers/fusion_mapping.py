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
from collections.abc import Mapping
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

_patch_embedding_fusion_cache: dict[type, tuple[dict[str, type[nn.Module]], list[WeightConverter]]] = {}


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


def _make_patch_shape_getter(in_channels: int, kernel_size: tuple[int, int, int]):
    temporal_patch_size, patch_height, patch_width = kernel_size

    def get_patch_shape(_):
        return in_channels, temporal_patch_size, (patch_height, patch_width)

    return get_patch_shape


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
        operations=[
            Conv3dToLinear(get_patch_shape=_make_patch_shape_getter(proj.in_channels, tuple(proj.kernel_size)))
        ],
    )


def _discover_fusable_patch_embedding_classes(
    cls: "type[PreTrainedModel]", config
) -> tuple[dict[str, type[nn.Module]], list[WeightConverter]]:
    if cls in _patch_embedding_fusion_cache:
        return _patch_embedding_fusion_cache[cls]

    with torch.device("meta"):
        model = cls(config)

    seen_classes = set()
    patch_mapping = {}
    converters = []
    for module_name, module in model.named_modules():
        module_cls = type(module)
        if not _is_fusable_patch_embedding(module):
            continue

        converters.append(_build_weight_converter_for_module(module_name, module))
        if module_cls in seen_classes:
            continue

        seen_classes.add(module_cls)
        patch_mapping[module_cls.__name__] = _make_fused_patch_embedding_class(module_cls, module)

    _patch_embedding_fusion_cache[cls] = (patch_mapping, converters)
    return patch_mapping, converters


def register_patch_embedding_patches(cls: "type[PreTrainedModel]", config) -> None:
    fusable_classes, converters = _discover_fusable_patch_embedding_classes(cls, config)
    if not fusable_classes:
        logger.debug("No compatible patch-embedding classes found to fuse for %s", cls.__name__)
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


def register_fusion_patches(
    cls: "type[PreTrainedModel]", config, fusion_config: Mapping[str, bool | Mapping[str, Any]] | None = None
) -> None:
    if not fusion_config:
        return

    for fusion_name, fusion_options in fusion_config.items():
        if fusion_options is False:
            continue
        if fusion_options is not True and not isinstance(fusion_options, Mapping):
            raise ValueError(
                f"Invalid fusion config for {fusion_name}: expected `True`, `False`, or a mapping of options."
            )
        if fusion_name != "patch_embeddings":
            raise ValueError(f"Unknown fusion type: {fusion_name}")

        register_patch_embedding_patches(cls, config)
