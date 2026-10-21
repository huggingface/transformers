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
"""Fine-grained quantizer for models whose modules use different quantization methods.

It hosts one quantizer per config in the `dtype_plan`. Each module loads through the quantizer of its entry.
"""

import copy
import re
from typing import TYPE_CHECKING

from .base import HfQuantizer, get_keys_to_not_convert
from .quantizers_utils import checkpoint_module_namer, plan_entry_for, skip_pattern


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedConfig


class FineGrainedHfQuantizer(HfQuantizer):
    """Quantizer that hosts one member quantizer per config in the config's `dtype_plan`.

    A hook on one tensor goes to the member that owns its module. A hook on the whole model runs on every member.
    DeepSeek-V4 is an example, with MXFP4 experts and block FP8 linears.

    A registered quantizer can be a member. The host relies on it to:
    - build its skip-list with `get_modules_to_not_convert`. The host scopes each member through `dtype_plan_skip`.
      A member that converts a module of another entry raises.
    - take its loading options, such as `dequantize`, from its config. The host sets them on the member's own copy.
    - keep its whole-model hooks to its own modules, or make them safe to run once per member.

    Each member updates its own copy of the model's conversions. Conversions of several members that read the same
    keys become one converter, which runs each module's tensors through its own member's version.
    """

    quantization_config: "FineGrainedConfig"

    def __init__(self, quantization_config: "FineGrainedConfig", **kwargs):
        super().__init__(quantization_config, **kwargs)
        from .auto import AutoHfQuantizer

        # Equal entries share one config, so a per-layer export builds one member per format
        shared = {}
        quantization_config.dtype_plan = {
            pattern: None if config is None else shared.setdefault(config.to_json_string(), config)
            for pattern, config in quantization_config.dtype_plan.items()
        }
        members = {}
        for key, config in shared.items():
            # each member works on a copy, so its scope never reaches the saved config
            member_config = copy.deepcopy(config)
            # the host's loading options hold for every member
            for option in quantization_config._entry_loading_options:
                setattr(member_config, option, getattr(quantization_config, option, None))
            # the same dispatch as a config passed alone, e.g. bitsandbytes picks its 4-bit or 8-bit quantizer
            members[key] = AutoHfQuantizer.from_config(member_config, **kwargs)
        self.members = list(members.values())
        # each dispatching conversion's group, as merged: the model's version under `None`, a member's under it
        self._groups = {}
        # `{module regex: member}`, the plan with each config's member in its place. `None` stays in full precision.
        self._plan = {
            pattern: None if config is None else members[config.to_json_string()]
            for pattern, config in quantization_config.dtype_plan.items()
        }
        self.dequantized_saves_unquantized = all(member.dequantized_saves_unquantized for member in self.members)
        for member in self.members:
            # the host keeps the model's transforms for a dequantized save, once for all members
            member.dequantized_saves_unquantized = False

    def member_for(self, module_name: str) -> HfQuantizer | None:
        """Return the member whose entry covers `module_name`, or `None` if it stays in full precision."""
        return plan_entry_for(self._plan, module_name)

    def _member_for_param(self, param_name: str) -> HfQuantizer | None:
        return self.member_for(param_name.rpartition(".")[0])

    def update_dtype(self, dtype):
        for member in self.members:
            dtype = member.update_dtype(dtype)
        return dtype

    def update_device_map(self, device_map):
        for member in self.members:
            device_map = member.update_device_map(device_map)
        return device_map

    def adjust_max_memory(self, max_memory):
        for member in self.members:
            max_memory = member.adjust_max_memory(max_memory)
        return max_memory

    def update_attn_implementation(self, config):
        for member in self.members:
            config = member.update_attn_implementation(config)
        return config

    def validate_environment(self, *args, **kwargs):
        for member in self.members:
            member.validate_environment(*args, **kwargs)
        # a member that cannot run quantized dequantizes the whole model, so no half-dequantized model loads
        if any(getattr(member.quantization_config, "dequantize", False) for member in self.members):
            self.quantization_config.dequantize = True
            for member in self.members:
                member.quantization_config.dequantize = True

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        member = self._member_for_param(param_name)
        return member is not None and member.param_needs_quantization(model, param_name, **kwargs)

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param) -> float:
        member = self._member_for_param(param_name)
        return param.element_size() if member is None else member.param_element_size(model, param_name, param)

    def get_quantize_ops(self):
        from ..integrations.finegrained.host import MemberQuantize

        return MemberQuantize(self)

    def get_new_dtype(self, converter, param, dtype):
        from ..integrations.finegrained.host import MemberDispatch

        # a converter that members changed holds each member's version. The members must agree on the dtype.
        dispatch = next((op for op in getattr(converter, "operations", ()) if isinstance(op, MemberDispatch)), None)
        if dispatch is None:
            asked = [(member, converter) for member in self.members]
        else:
            asked = [(member, version) for member, version in dispatch.conversions.items() if member is not None]
        dtypes = {member.get_new_dtype(version, param, dtype) for member, version in asked}
        if len(dtypes) > 1:
            raise ValueError(f"The members of the dtype_plan load {converter.target_patterns} in different dtypes.")
        return dtypes.pop()

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        self._resolve_plan(model)
        # The fp32 modules and the model's default skips become `None` entries, so the plan alone says who owns a
        # module. The fp32 modules always win. A default skip, such as a tied `lm_head`, yields to an entry naming it.
        skipped = self.get_modules_to_not_convert(model, [], model._keep_in_fp32_modules)
        targeted = [pattern for pattern, owner in self._plan.items() if pattern != ".*" and owner is not None]
        defaults = get_keys_to_not_convert(model)
        skipped += [name for name in defaults if not any(re.search(pattern, name) for pattern in targeted)]
        self._plan.update(dict.fromkeys(map(skip_pattern, skipped)))
        to_checkpoint = checkpoint_module_namer(model)
        owners = {name: self.member_for(name) for name, _ in model.named_modules()}
        for member in self.members:
            own = [pattern for pattern, owner in self._plan.items() if owner is member]
            others = [pattern for pattern in self._plan if pattern not in own]
            # a skipped module takes its sub-tree with it, so a member never skips a parent of a module it owns
            parents = {parent for name, owner in owners.items() if owner is member for parent in _parents(name)}
            parents |= {to_checkpoint(parent) for parent in parents}
            member.dtype_plan_skip = _skip_unowned(own, others, parents)
            before = {name: type(module) for name, module in model.named_modules()}
            member._process_model_before_weight_loading(model, **kwargs)
            # a member that ignores `dtype_plan_skip` would take modules of other entries
            for name, module in model.named_modules():
                if name in before and type(module) is not before[name] and owners[name] is not member:
                    raise ValueError(
                        f"{type(member).__name__} converted {name!r}, which its `dtype_plan` entry does not cover. "
                        "A member must build its skip-list with `get_modules_to_not_convert`."
                    )

    def _resolve_plan(self, model: "PreTrainedModel"):
        """Add an entry by model name for each module a pattern matches by its checkpoint name.

        The saved config keeps the patterns as passed.
        """
        to_checkpoint = checkpoint_module_namer(model)
        plan = self._plan
        targeted = [pattern for pattern in plan if pattern != ".*"]
        resolved = dict(plan)
        for name, _ in model.named_modules():
            checkpoint_name = to_checkpoint(name)
            if checkpoint_name == name or not any(re.search(pattern, checkpoint_name) for pattern in targeted):
                continue
            source = plan_entry_for(plan, checkpoint_name, catch_all=False)
            if any(re.search(pattern, name) for pattern in targeted):
                if plan_entry_for(plan, name, catch_all=False) is not source:
                    raise ValueError(
                        f"Conflicting quantization entries for {name!r} and checkpoint module {checkpoint_name!r}"
                    )
                continue
            resolved[rf"^{re.escape(name)}$"] = source
        self._plan = resolved

    def update_parallel_plan(self, model, plan):
        super().update_parallel_plan(model, plan)
        for member in self.members:
            member.update_parallel_plan(model, plan)

    def update_weight_conversions(self, weight_conversions, model, **kwargs):
        # the base hook keeps the model's transforms for a dequantized save. The host has no names of its own.
        given = super().update_weight_conversions(weight_conversions, model)
        # each member updates its own copy, so the conversions it changes stay its own
        versions = [member.update_weight_conversions(copy.deepcopy(given), model, **kwargs) for member in self.members]
        return self._merge(given, versions)

    def _process_model_after_weight_loading(self, model, **kwargs):
        used = getattr(model, "_weight_conversions", None)
        if used is None:
            for member in self.members:
                model = member._process_model_after_weight_loading(model, **kwargs)
            return model
        # the loader keeps the conversions that matched a key. Each member gets its own version of those back.
        given, versions = [], [[] for _ in self.members]
        for conversion in used:
            group = self._groups.get(id(conversion), {None: conversion})
            if None in group:
                given.append(group[None])
            for own, member in zip(versions, self.members):
                if (version := group.get(member, group.get(None))) is not None:
                    own.append(version)
        updated = []
        for member, conversions in zip(self.members, versions):
            # a plain attribute, not the tensor or module `nn.Module.__setattr__` is typed for
            setattr(model, "_weight_conversions", copy.deepcopy(conversions))
            model = member._process_model_after_weight_loading(model, **kwargs)
            updated.append(getattr(model, "_weight_conversions", None) or [])
        setattr(model, "_weight_conversions", self._merge(given, updated))
        return model

    def _merge(self, given: list, versions: list[list]) -> list:
        """Return one conversion per group of conversions that read the same keys.

        `versions` holds each member's updated copy of `given`. A group whose versions agree and run only the model's
        ops stays as it is. Otherwise one converter runs each module's tensors through the version of its member. A
        member's own renamings come first, as each member puts them before the model's.
        """
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..integrations.finegrained.host import MemberDispatch

        groups, claimed = [], {}
        for owner, conversions in [(None, given), *zip(self.members, versions)]:
            for conversion in conversions:
                group = next((claimed[p] for p in conversion.source_patterns if p in claimed), None)
                if group is None:
                    group = {}
                    groups.append(group)
                claimed.update(dict.fromkeys(conversion.source_patterns, group))
                model_conversion = group.get(None)
                if owner is None or model_conversion is None or _signature(conversion) != _signature(model_conversion):
                    group[owner] = conversion
        renamings, merged = [], []
        for group in groups:
            conversions = list(group.values())
            quantizer_ops = [
                op for c in conversions for op in getattr(c, "operations", ()) if hasattr(op, "hf_quantizer")
            ]
            if len({_signature(conversion) for conversion in conversions}) == 1 and not quantizer_ops:
                # the members agree and run only the model's ops, e.g. a renaming they all add
                plain = isinstance(conversions[0], WeightRenaming) and None not in group
                (renamings if plain else merged).append(conversions[0])
                continue
            if any(not isinstance(conversion, WeightConverter) for conversion in conversions):
                raise ValueError(f"Two quantizers rename {conversions[0].source_patterns} differently.")
            if len({tuple(conversion.target_patterns) for conversion in conversions}) > 1:
                raise ValueError(f"Two quantizers load {conversions[0].source_patterns} into different keys.")
            # the patterns as passed, so the converter reverses like its versions on save
            sources = list(
                dict.fromkeys(p for conversion in conversions for p in conversion._original_source_patterns)
            )
            # a copy keeps the scope and the dequantized save of the model's conversion
            converter = group.get(None, conversions[0]).copy(
                sources,
                conversions[0]._original_target_patterns,
                operations=[MemberDispatch(self, group)],
                force_cpu=any(conversion.force_cpu for conversion in conversions),
            )
            merged.append(converter)
            self._groups[id(converter)] = group
        return renamings + merged

    def postprocess_model(self, model, **kwargs):
        if not self.pre_quantized and self.quantization_config.dequantize:
            # nothing was quantized, so the model loads as an unquantized one
            self.remove_quantization_config(model)
            return model
        return super().postprocess_model(model, **kwargs)

    def is_serializable(self):
        return all(member.is_serializable() for member in self.members)

    @property
    def is_trainable(self) -> bool:
        return all(member.is_trainable for member in self.members)

    @property
    def is_compileable(self) -> bool:
        return all(member.is_compileable for member in self.members)


def _parents(name: str) -> list[str]:
    """Return the names of the modules above `name`, e.g. `a` and `a.b` for `a.b.c`."""
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


def _skip_unowned(own: list[str], others: list[str], parents: set[str]) -> str:
    """Return a skip-list entry that matches every module a member with the `own` patterns does not own.

    Another entry's targeted pattern wins over a catch-all. A member without the catch-all owns only what its patterns
    match. The entry never matches one of `parents`, the modules above the ones the member owns.
    """
    unowned = []
    if targeted := [pattern for pattern in others if pattern != ".*"]:
        unowned.append(f"(?=.*(?:{'|'.join(targeted)}))")
    if ".*" not in own:
        unowned.append(f"(?!.*(?:{'|'.join(own)}))")
    if not unowned:
        return "(?!)"
    kept = f"(?!(?:{'|'.join(map(re.escape, sorted(parents)))})$)" if parents else ""
    return f"{kept}(?:{'|'.join(unowned)}).*"


def _signature(conversion) -> tuple:
    """Return what a conversion does: the keys it reads, where they go and the ops they run through."""
    operations = tuple(type(op).__name__ for op in getattr(conversion, "operations", ()))
    return type(conversion), tuple(conversion.source_patterns), tuple(conversion.target_patterns), operations
