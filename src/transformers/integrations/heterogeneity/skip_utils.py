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

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from transformers.utils import is_torch_available


if TYPE_CHECKING:
    from torch import nn

    from transformers.integrations.heterogeneity.configuration_utils import _GenericModelingState
    from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
        HeterogeneousModelingSpec,
        MemberKey,
        SkipDescriptor,
    )

if is_torch_available():
    import torch
    from torch import nn


@dataclass(frozen=True)
class ReturnEntry:
    arg_name: str
    transform: Callable


@dataclass(frozen=True)
class _ResolvedReturnEntry(ReturnEntry):
    position: int | None
    default: object

    def resolve(self, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        if self.arg_name in kwargs:
            return kwargs[self.arg_name]
        if self.position is not None and self.position < len(args):
            return args[self.position]
        if self.default is not inspect.Parameter.empty:
            return self.default
        raise TypeError(f"missing a required argument: '{self.arg_name}'")


def _resolve_return_entries(
    cls: type[nn.Module], return_entries: tuple[ReturnEntry | None, ...] | None
) -> tuple[_ResolvedReturnEntry | None, ...] | None:
    if return_entries is None:
        return None

    parameters = inspect.signature(cls.forward).parameters
    parameter_names = tuple(parameters)[1:]

    missing_names = [
        return_entry.arg_name
        for return_entry in return_entries
        if return_entry is not None and return_entry.arg_name not in parameter_names
    ]
    if missing_names:
        raise ValueError(
            f"In the skip replacement for {cls.__qualname__}, the following return entry arg names "
            f"are not arguments of {cls.__qualname__}.forward(): {missing_names}"
        )

    resolved_entries = []
    for return_entry in return_entries:
        if return_entry is None:
            resolved_entries.append(None)
            continue

        parameter = parameters[return_entry.arg_name]
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"Return entries cannot reference variadic argument '{return_entry.arg_name}' "
                f"in {cls.__qualname__}.forward()."
            )

        resolved_entries.append(
            _ResolvedReturnEntry(
                arg_name=return_entry.arg_name,
                transform=return_entry.transform,
                position=(
                    parameter_names.index(return_entry.arg_name)
                    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    else None
                ),
                default=parameter.default,
            )
        )

    return tuple(resolved_entries)


if is_torch_available():

    class _NoOpReplacement(nn.Module):
        def __init__(
            self,
            *,
            source_class: type[nn.Module],
            return_entries: tuple[_ResolvedReturnEntry | None, ...] | None = None,
            return_tuple: bool = False,
        ):
            super().__init__()
            self.source_class = source_class
            self._return_entries = return_entries
            self._return_tuple = return_tuple

            # Some layers read a member's weight even when the member is skipped. This empty buffer gives them one,
            # without adding memory, changing outputs, or being saved.
            self.register_buffer("weight", torch.empty(0), persistent=False)

        def extra_repr(self) -> str:
            return f"source_class={self.source_class.__name__}"

        def forward(self, *args, **kwargs):
            if self._return_entries is None:
                return None

            outputs = [None] * len(self._return_entries)
            for i, return_entry in enumerate(self._return_entries):
                if return_entry is None:
                    continue

                try:
                    arg_value = return_entry.resolve(args, kwargs)
                except TypeError:
                    raise TypeError(
                        f"In the skip replacement for {self.source_class.__qualname__}, "
                        f"required argument '{return_entry.arg_name}' was not provided"
                    ) from None

                try:
                    outputs[i] = return_entry.transform(arg_value)
                except Exception as e:
                    raise RuntimeError(
                        f"In the skip replacement for {self.source_class.__qualname__}, failed to apply transform "
                        f"{return_entry.transform!r} to argument '{return_entry.arg_name}' "
                        f"(value type: {type(arg_value).__name__}): {e}"
                    ) from e

            return tuple(outputs) if self._return_tuple else outputs[0]


def get_skip_replacement_factory(
    cls: type[nn.Module],
    to_return: ReturnEntry | list[ReturnEntry | None] | None,
) -> Callable[[], nn.Module]:
    """Return a factory for no-op modules with the specified forward outputs."""
    if to_return is None:
        return_entries = None
        return_tuple = False
    elif isinstance(to_return, ReturnEntry):
        return_entries = (to_return,)
        return_tuple = False
    else:
        return_entries = tuple(to_return)
        return_tuple = True

    return partial(
        _NoOpReplacement,
        source_class=cls,
        return_entries=_resolve_return_entries(cls, return_entries),
        return_tuple=return_tuple,
    )


def _get_member_path(member_key: MemberKey) -> str:
    return member_key[0] if isinstance(member_key, tuple) else member_key


def _get_key_rank(member_key: MemberKey, member_mro: tuple[type, ...]) -> int:
    # More specific keys rank lower: a (path, class) key ranks by the position of its class in the member's MRO, and a
    # plain path ranks after every class
    return member_mro.index(member_key[1]) if isinstance(member_key, tuple) else len(member_mro)


def _get_matching_member_keys(
    layer: nn.Module, member_keys: Iterable[MemberKey], *, error_prefix: str, missing_member_hint: str = ""
) -> list[MemberKey]:
    """Return the keys in `member_keys` that match members of `layer`, at most one per member path. A plain path must
    exist in the layer. A (path, class) key matches only when the layer has a member at the path and it's an
    instance of that class, and takes precedence over the plain path. Among matching (path, class) keys, the one with
    the most specific class takes precedence."""
    member_path_to_key = {}
    for member_key in member_keys:
        member_path = _get_member_path(member_key)
        try:
            member = layer.get_submodule(member_path)
        except AttributeError:
            if isinstance(member_key, tuple):
                continue
            raise AttributeError(
                f"{error_prefix}, but {layer.__class__.__name__} has no submodule '{member_path}'.{missing_member_hint}"
            ) from None

        member_mro = type(member).__mro__
        if isinstance(member_key, tuple) and member_key[1] not in member_mro:
            continue
        matched_key = member_path_to_key.get(member_path)
        if matched_key is None or _get_key_rank(member_key, member_mro) < _get_key_rank(matched_key, member_mro):
            member_path_to_key[member_path] = member_key
    return list(member_path_to_key.values())


def validate_skip_descriptors(
    per_layer_skip_types: list[list[str]], heterogeneous_modeling_specs: Iterable[HeterogeneousModelingSpec]
) -> None:
    """Raise if a skip type has no skip descriptor in any of the specs, before any layer is built. `apply_skips` checks
    each layer against the spec of its own class."""
    skip_types = {skip_type for layer_skip_types in per_layer_skip_types for skip_type in layer_skip_types}
    described_skip_types = {
        skip_type
        for heterogeneous_modeling_spec in heterogeneous_modeling_specs
        for skip_type in heterogeneous_modeling_spec.skip_descriptors
    }
    missing_descriptors = skip_types - described_skip_types
    if missing_descriptors:
        raise ValueError(f"Skip descriptors are missing for the following skip types: {missing_descriptors}")


def _apply_skip_descriptor(
    *,
    layer: nn.Module,
    skip_type: str,
    skip_descriptor: SkipDescriptor,
    layer_idx: int,
) -> None:
    """Replace the layer's members with the replacements of the matching keys in `skip_descriptor`."""
    error_prefix = f"Layer {layer_idx} skips '{skip_type}'"
    matching_member_keys = _get_matching_member_keys(layer, skip_descriptor, error_prefix=error_prefix)
    matched_member_paths = {_get_member_path(member_key) for member_key in matching_member_keys}

    # Before replacing anything, check that every member path in the descriptor has a matching key. Otherwise the skip
    # would only partly apply, and the layer would silently compute an unintended output.
    for member_key in skip_descriptor:
        member_path = _get_member_path(member_key)
        if member_path in matched_member_paths:
            continue
        try:
            member = layer.get_submodule(member_path)
        except AttributeError:
            raise AttributeError(
                f"{error_prefix}, but {layer.__class__.__name__} has no submodule '{member_path}'."
            ) from None
        raise ValueError(
            f"{error_prefix}, but that descriptor has no replacement for member '{member_path}' with class "
            f"{type(member).__name__}. Add a class-specific or generic replacement."
        )

    for member_key in matching_member_keys:
        member_path = _get_member_path(member_key)
        original = layer.get_submodule(member_path)
        replacement_factory = skip_descriptor[member_key]
        replacement = replacement_factory()
        replacement._heterogeneity_skipped_class = type(original)
        layer.set_submodule(member_path, replacement)


def apply_skips(
    *,
    layer: nn.Module,
    layer_idx: int,
    skip_types: list[str],
    heterogeneous_modeling_spec: HeterogeneousModelingSpec,
    generic_modeling_state: _GenericModelingState,
) -> None:
    """Apply the layer's skips, and record in `generic_modeling_state` whether they leave its attention mask
    unconsumed and which of its caches they leave unwritten."""
    # A model with several layer classes has a spec for each, and a skip type can be described in only some of them
    if undescribed_skip_types := set(skip_types) - heterogeneous_modeling_spec.skip_descriptors.keys():
        raise ValueError(
            f"Layer {layer_idx} skips {sorted(undescribed_skip_types)}, but the heterogeneous modeling spec for its "
            f"class, `{heterogeneous_modeling_spec.layer_cls.__name__}`, has no skip descriptors for them."
        )

    # Unlike skip descriptors, `attention_mask_consumers` and `cache_writers` can use a (path, class) key for a member
    # that only some layers build
    missing_member_hint = (
        " If the path is right and only some layers build this member, use a `(path, class)` key for it."
    )
    # Find the declared members before the skips replace them
    attention_mask_consumers = [
        layer.get_submodule(_get_member_path(member_key))
        for member_key in _get_matching_member_keys(
            layer,
            heterogeneous_modeling_spec.attention_mask_consumers,
            error_prefix=f"The spec's `attention_mask_consumers` lists a member of layer {layer_idx}",
            missing_member_hint=missing_member_hint,
        )
    ]
    cache_writers = {
        layer.get_submodule(_get_member_path(member_key)): heterogeneous_modeling_spec.cache_writers[member_key]
        for member_key in _get_matching_member_keys(
            layer,
            heterogeneous_modeling_spec.cache_writers,
            error_prefix=f"The spec's `cache_writers` lists a member of layer {layer_idx}",
            missing_member_hint=missing_member_hint,
        )
    }

    for skip_type in skip_types:
        _apply_skip_descriptor(
            layer=layer,
            skip_type=skip_type,
            skip_descriptor=heterogeneous_modeling_spec.skip_descriptors[skip_type],
            layer_idx=layer_idx,
        )

    remaining_modules = set(layer.modules())
    if attention_mask_consumers and remaining_modules.isdisjoint(attention_mask_consumers):
        generic_modeling_state.unconsumed_attention_mask_layers.add(layer_idx)

    declared_caches = set()
    written_caches = set()
    for writer, caches in cache_writers.items():
        declared_caches |= caches
        if writer in remaining_modules:
            written_caches |= caches
    if unwritten_caches := declared_caches - written_caches:
        generic_modeling_state.per_layer_unwritten_caches[layer_idx] = unwritten_caches
