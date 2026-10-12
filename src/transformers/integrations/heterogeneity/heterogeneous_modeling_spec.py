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

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver


if TYPE_CHECKING:
    from torch import nn

    from transformers.modeling_utils import PreTrainedModel


# A layer member, as an attribute path like "self_attn" or "block.attn", or as a (path, class) pair that only matches
# members of that class. A (path, class) pair takes precedence over the plain path. When several classes match, the
# most specific one takes precedence.
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
    layer_cls: type[nn.Module]
    layer_idx_resolver: LayerIdxResolver
    skip_descriptors: dict[str, SkipDescriptor] | None = None
    # Required with skip descriptors. List every member that consumes the attention mask, and map every cache writer to
    # the caches it writes. A layer that skips all its attention mask consumers gets no attention mask, and when it
    # skips all writers of a cache, it no longer has that cache.
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


def _nest_member_key(member_key: MemberKey, parent_path: str) -> MemberKey:
    if isinstance(member_key, tuple):
        member_path, member_cls = member_key
        return f"{parent_path}.{member_path}", member_cls
    return f"{parent_path}.{member_key}"


def nest_spec_member_paths(spec: HeterogeneousModelingSpec, parent_path: str) -> HeterogeneousModelingSpec:
    """Return a copy of `spec` with every member path nested under `parent_path`.

    For example, with `parent_path="mtp_block"`, `"self_attn"` becomes `"mtp_block.self_attn"`.
    """
    return dataclasses.replace(
        spec,
        skip_descriptors={
            skip_type: {_nest_member_key(key, parent_path): factory for key, factory in targets.items()}
            for skip_type, targets in spec.skip_descriptors.items()
        },
        attention_mask_consumers=[_nest_member_key(key, parent_path) for key in spec.attention_mask_consumers],
        cache_writers={_nest_member_key(key, parent_path): caches for key, caches in spec.cache_writers.items()},
    )


def get_heterogeneous_modeling_spec(model: PreTrainedModel) -> HeterogeneousModelingSpec | None:
    """Return the generic heterogeneous modeling spec of `model`, or `None` if it has none.

    A model can provide its own spec in `_heterogeneous_modeling_spec`, which takes precedence over
    `MODEL_TYPE_TO_SPEC_FACTORY`. Models without a spec are left unpatched, and can handle `per_layer_config` in their
    own modeling code.
    """
    heterogeneous_modeling_spec = getattr(model, "_heterogeneous_modeling_spec", None)

    if heterogeneous_modeling_spec is not None:
        return heterogeneous_modeling_spec

    model_type = model.config.model_type

    from transformers.integrations.heterogeneity.supported_models import MODEL_TYPE_TO_SPEC_FACTORY

    spec_factory = MODEL_TYPE_TO_SPEC_FACTORY.get(model_type)
    return spec_factory() if spec_factory is not None else None
