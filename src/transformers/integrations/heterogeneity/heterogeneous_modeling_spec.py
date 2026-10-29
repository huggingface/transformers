# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver


if TYPE_CHECKING:
    from torch import nn

    from transformers.modeling_utils import PreTrainedModel


MemberKey: TypeAlias = str | tuple[str, type]
SkipDescriptor: TypeAlias = dict[MemberKey, Callable[[], "nn.Module"]]

# A layer can have a key/value cache, a linear-attention cache, or both. `Cache.update` writes the key/value cache,
# and `update_conv_state` and `update_recurrent_state` write the linear-attention cache.
KEY_VALUE_CACHE = "key_value"
LINEAR_ATTENTION_CACHE = "linear_attention"
CACHES = frozenset({KEY_VALUE_CACHE, LINEAR_ATTENTION_CACHE})


def _validate_caches(member_key: MemberKey, caches: Iterable[str]) -> frozenset[str]:
    if isinstance(caches, str):
        raise TypeError(
            f"`cache_writers[{member_key!r}]` must be an iterable of caches, such as `[{caches!r}]`, not a string."
        )
    caches = frozenset(caches)
    if unknown_caches := caches - CACHES:
        raise ValueError(
            f"`cache_writers[{member_key!r}]` has unknown caches {sorted(unknown_caches)}. Valid caches are "
            f"{sorted(CACHES)}."
        )
    return caches


@dataclass(frozen=True)
class HeterogeneousModelingSpec:
    """Describes how generic heterogeneous modeling builds the layers of one layer class.

    Generic heterogeneous modeling is used when a model has a spec and its config is heterogeneous. It patches
    `layer_cls.__init__`, so that:

    - each layer is built from its own config, `config.per_layer_config[layer_idx]`.
    - each skip type in the `skip` list of the layer's config replaces the members that its skip descriptor lists with
      skip replacements, which are no-op modules.

    A model sets its specs in its `_heterogeneous_modeling_spec` class attribute, while built-in models can instead be
    listed in `MODEL_TYPE_TO_SPEC_FACTORY` (see `get_heterogeneous_modeling_specs`). For a walkthrough, see the generic
    heterogeneous modeling guide, `docs/source/en/generic_heterogeneous_modeling.md`.

    Skip descriptors, `attention_mask_consumers` and `cache_writers` use member keys to name the members of a layer,
    which are its submodules. A member key has one of two forms:

    - A path, such as `"self_attn"` or `"block.attn"`, which matches the member at this path whatever its class. The
      layer must have a member at this path.
    - A `(path, class)` tuple, which matches only if the layer has a member at this path and that member is an instance
      of the class.

    If both forms match the same member, the `(path, class)` key is used. If several `(path, class)` keys match the
    same member, the key with the class closest to the member's class is used. For example, a key with a subclass is
    used before a key with its base class.

    Args:
        layer_cls (`type[nn.Module]`):
            The layer class to patch, usually the model's decoder layer. The model must pass its own config object, not
            a copy, as the first argument of `layer_cls.__init__` after `self`, where the patched `__init__` replaces
            it with the layer's config. Only layers that receive this object are patched, and building the model raises
            an error if any layer isn't. `layer_cls.forward` must also take the attention mask as an argument named
            `attention_mask`, so that each layer gets its own mask.
        layer_idx_resolver (`LayerIdxResolver`):
            Finds the index of each layer while the layer's `__init__` runs. Most models use `LayerIdxFromArgument`.
        skip_descriptors (`dict[str, dict[str | tuple[str, type], Callable[[], nn.Module]]]`, *optional*):
            Maps each skip type that layers of class `layer_cls` can list in their `skip` list to its skip descriptor.
            A skip descriptor maps member keys to factories that build skip replacements for those members, usually
            created with `get_skip_replacement_factory`.

            When a layer's `skip` list contains the skip type, every path in the skip descriptor, including the paths
            in `(path, class)` keys, must point to a member of the layer, and at least one key with that path must
            match that member.
        attention_mask_consumers (`Iterable[str | tuple[str, type]]`, *optional*):
            The member keys of all members that use the attention mask. Include members that are never skipped, since
            otherwise a layer can lose its attention mask while one of its members still uses it, and use a
            `(path, class)` key for a member that exists in only some layers. Required if `skip_descriptors` is not
            empty; use `[]` when there are none.

            If a layer has attention mask consumers and its skips replace all of them, the attention mask of the layer
            is `None`.
        cache_writers (`dict[str | tuple[str, type], Iterable[str]]`, *optional*):
            Maps the member key of each member that writes to a cache to the caches that it writes. Include members
            that are never skipped, since otherwise a layer can lose a cache that one of its members still writes, and
            use a `(path, class)` key for a member that exists in only some layers. Required if `skip_descriptors` is
            not empty; use `{}` when there are none. The valid caches are:

            - `"key_value"`, for members that call `Cache.update`.
            - `"linear_attention"`, for members that call `update_conv_state` or `update_recurrent_state`.

            For example, `{"self_attn": ["key_value"]}`.
    """

    layer_cls: type[nn.Module]
    layer_idx_resolver: LayerIdxResolver
    skip_descriptors: dict[str, SkipDescriptor] | None = None
    attention_mask_consumers: Iterable[MemberKey] | None = None
    cache_writers: dict[MemberKey, Iterable[str]] | None = None

    def __post_init__(self):
        object.__setattr__(self, "skip_descriptors", self.skip_descriptors or {})
        if self.skip_descriptors and (self.attention_mask_consumers is None or self.cache_writers is None):
            raise ValueError(
                "A spec with skip descriptors must also set `attention_mask_consumers` (`[]` if there are none) and "
                "`cache_writers` (`{}` if there are none)."
            )

        if isinstance(self.attention_mask_consumers, str):
            raise TypeError(
                "`attention_mask_consumers` must be an iterable of members, such as "
                f"`[{self.attention_mask_consumers!r}]`, not a string."
            )
        object.__setattr__(self, "attention_mask_consumers", frozenset(self.attention_mask_consumers or ()))

        object.__setattr__(
            self,
            "cache_writers",
            {
                member_key: _validate_caches(member_key, caches)
                for member_key, caches in (self.cache_writers or {}).items()
            },
        )


def get_heterogeneous_modeling_specs(model: PreTrainedModel) -> tuple[HeterogeneousModelingSpec, ...]:
    """Returns the generic heterogeneous modeling specs of `model`, one for each layer class, or `()` if it has none.

    `MODEL_TYPE_TO_SPEC_FACTORY` provides the specs of the supported built-in models, by `model_type`. Other models,
    such as custom models, can add support by setting the `_heterogeneous_modeling_spec` class attribute to a spec or a
    tuple of specs. The attribute takes precedence over `MODEL_TYPE_TO_SPEC_FACTORY`, and setting it to `None` turns
    off generic patching.

    Raises:
        `ValueError`: If two or more specs have the same `layer_cls`.
    """
    if hasattr(model, "_heterogeneous_modeling_spec"):
        specs = model._heterogeneous_modeling_spec
    else:
        from transformers.integrations.heterogeneity.supported_models import MODEL_TYPE_TO_SPEC_FACTORY

        spec_factory = MODEL_TYPE_TO_SPEC_FACTORY.get(model.config.model_type)
        specs = spec_factory() if spec_factory is not None else None

    if specs is None:
        return ()

    specs = (specs,) if isinstance(specs, HeterogeneousModelingSpec) else tuple(specs)

    layer_classes = [spec.layer_cls for spec in specs]
    if duplicate_layer_classes := {layer_cls for layer_cls in layer_classes if layer_classes.count(layer_cls) > 1}:
        raise ValueError(
            f"{type(model).__name__} has several heterogeneous modeling specs for "
            f"{sorted(layer_cls.__name__ for layer_cls in duplicate_layer_classes)}. Each layer class needs exactly "
            "one spec."
        )

    return specs
