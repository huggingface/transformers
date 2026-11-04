# Copyright 2026 HuggingFace Inc.
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

import copy
import math
import random
from contextlib import contextmanager
from typing import Any, ClassVar, Protocol, runtime_checkable

from transformers import (
    PreTrainedConfig,
    PreTrainedModel,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import torch_device


if is_torch_available():
    import torch

    from transformers import PreTrainedModel


global_rng = random.Random()


def compare_state_dicts(state_dict1, state_dict2) -> bool:
    """Make sure 2 state dicts are the exact same"""
    # Make sure the keys are the exact same
    if sorted(state_dict1.keys()) != sorted(state_dict2.keys()):
        in1_not2 = sorted(set(state_dict1.keys()) - set(state_dict2.keys()))
        in2_not1 = sorted(set(state_dict2.keys()) - set(state_dict1.keys()))
        raise ValueError(
            f"The keys of both state dict are not the same.\nKeys found in the first item but not second: {in1_not2}"
            f"\nKeys found in the second item but not first: {in2_not1}"
        )

    for k, v1 in state_dict1.items():
        v2 = state_dict2[k]
        try:
            torch.testing.assert_close(v1, v2)
        except Exception as e:
            raise AssertionError(f"For key {k}: {e}")

    return True


@contextmanager
def seeded_weight_init():
    """Add a seed before weight initialization, to get the same random weights deterministically"""
    try:
        # Monkey patch the method to add a seed (we do it on PreTrainedModel._initialize_weights, which wraps
        # `_init_weights` so that it can add the seed for composite models as well)
        original_initialize_weights = PreTrainedModel._initialize_weights

        def seeded_initialize_weights(*args, **kwargs):
            set_seed(42)
            original_initialize_weights(*args, **kwargs)

        PreTrainedModel._initialize_weights = seeded_initialize_weights

        yield
    finally:
        # Restore it
        PreTrainedModel._initialize_weights = original_initialize_weights


@contextmanager
def skip_weight_init():
    """Skip weight initialization by `_init_weights` altogether."""
    try:
        original_initialize_weights = PreTrainedModel._initialize_weights

        # Just do nothing instead
        def skip_initialize_weights(*args, **kwargs):
            pass

        PreTrainedModel._initialize_weights = skip_initialize_weights

        yield
    finally:
        # Restore it
        PreTrainedModel._initialize_weights = original_initialize_weights


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    # Iterate the config rather than its `__dict__`: a heterogeneous config raises on global reads
    # of its per-layer attributes, and iteration hides exactly those. Sub-configs are still yielded,
    # so a heterogeneous one (e.g. `Gemma4Config.text_config`) is still recursed into.
    for key in list(configs_no_init):
        if (
            "init_range" in key
            or "initializer_range" in key
            or "_std" in key
            or "initializer_factor" in key
            or ("layer_scale" in key and key != "use_layer_scale")
        ):
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PreTrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def _get_nested_configs(config):
    """The config and all the sub-configs nested in it, at any depth."""
    nested_configs = [config]
    for subconfig_key in config.sub_configs:
        if (subconfig := getattr(config, subconfig_key, None)) is not None:
            nested_configs.extend(_get_nested_configs(subconfig))
    return nested_configs


def _set_subconfig_attributes(config, attribute_name, value):
    """Helper function to recursively set a config attr to a given value"""
    for k in config.sub_configs:
        if (
            attribute_name == "output_attentions"
            and k == "vision_config"
            and "Timm" in getattr(config, k).__class__.__name__
        ):  # skip because it's causes errors with TimmBackbone
            continue
        if getattr(config, k) is not None:
            setattr(getattr(config, k), attribute_name, value)
            _set_subconfig_attributes(getattr(config, k), attribute_name, value)


def prepare_config_headdim(config, requested_dim):
    """
    This method allows to update the head dim for all model types including
    composite models and models that do not support head dim by themselves.
    Why? A lot of kernels including flex attention rely on triton for compilation.
    However, triton cannot handle hidden dimensions of less than 16 for example.
    (There are many more examples especially now that the `kernels` library is
    supported)
    """
    config = copy.deepcopy(config)

    def update_config_headdim(config, requested_dim):
        # Flex Attention cannot use dropout
        if hasattr(config, "attention_dropout"):
            config.attention_dropout = 0.0
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = 0.0
        if hasattr(config, "dropout_rate"):
            config.dropout_rate = 0.0

        # Update the head dim and try to update hidden size as well if present in config
        # NOTE: some models may have none if the values in sub-config, thus we check for `Noneness`
        head_dim = None
        # On a heterogeneous config `head_dim` cannot be read globally, so probe for it without
        # tripping the guard. Per-layer overrides are bumped below.
        global_head_dim = config._getattr_without_heterogeneous_validation("head_dim", None)
        if global_head_dim is not None:
            head_dim = global_head_dim
            config.head_dim = max(requested_dim, global_head_dim)
            if config.is_heterogeneous and "head_dim" in config.per_layer_attributes:
                overrides = copy.deepcopy(config._heterogeneity_spec.per_layer_overrides)
                for layer_overrides in overrides.values():
                    if "head_dim" in layer_overrides:
                        layer_overrides["head_dim"] = max(requested_dim, layer_overrides["head_dim"])
                config.per_layer_config = overrides

        cross_head_dim = None
        if hasattr(config, "cross_head_dim") and config.cross_head_dim is not None:
            cross_head_dim = config.cross_head_dim
            config.cross_head_dim = max(requested_dim, config.cross_head_dim)
        if (
            getattr(config, "hidden_size", None) is not None
            and getattr(config, "num_attention_heads", None) is not None
        ):
            # For some models, num_attention_heads is a list of ints: we take the max to maximize the multiplier
            num_attn_heads = getattr(config, "num_attention_heads")
            num_attn_heads = num_attn_heads if isinstance(num_attn_heads, int) else max(num_attn_heads)
            head_dim = head_dim if head_dim is not None else config.hidden_size // num_attn_heads
            config.hidden_size *= max(math.ceil(requested_dim / head_dim), 1)

        if (
            getattr(config, "decoder_hidden_size", None) is not None
            and getattr(config, "decoder_num_attention_heads", None) is not None
        ):
            decoder_head_dim = config.decoder_hidden_size // config.decoder_num_attention_heads
            config.decoder_hidden_size *= max(math.ceil(requested_dim / decoder_head_dim), 1)

        if (
            getattr(config, "cross_hidden_size", None) is not None
            and getattr(config, "cross_num_attention_heads", None) is not None
        ):
            cross_head_dim = (
                cross_head_dim
                if cross_head_dim is not None
                else config.cross_hidden_size // config.cross_num_attention_heads
            )
            config.cross_hidden_size *= max(math.ceil(requested_dim / cross_head_dim), 1)

        # 3d rope also depends on the head dim
        # (we assume easy shapes here where we get to the requested head dim at least)
        if (
            getattr(config, "rope_parameters", None) is not None
            and len(config.rope_parameters.get("mrope_section", [])) > 0
        ):
            scaling_factor = max(requested_dim // (sum(config.rope_parameters["mrope_section"]) * 2), 1)
            config.rope_parameters["mrope_section"] = [
                section * scaling_factor for section in config.rope_parameters["mrope_section"]
            ]

    def update_config_headdim_recursive(config, requested_dim):
        # Needs to recurse since we have some models with 3-level nesting
        update_config_headdim(config, requested_dim)
        for key in getattr(config, "sub_configs", {}):
            sub_config = getattr(config, key, None)
            if sub_config is not None:
                update_config_headdim_recursive(sub_config, requested_dim)

    # Update config values
    update_config_headdim_recursive(config, requested_dim)

    return config


def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)


def _mock_all_init_weights(self):
    import transformers.modeling_utils

    if transformers.modeling_utils._init_weights:
        for module in self.modules():
            module._is_hf_initialized = False
        # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()


def find_parent_traceback(full_param_name: str, model: PreTrainedModel) -> tuple[str, str, str]:
    """From a given parameter or buffer `full_param_name`, find its immediate parent class name and immediate
    PreTrainedModel parent class name."""
    parent_name, name = full_param_name.rsplit(".", 1) if "." in full_param_name else ("", full_param_name)
    parent = model.get_submodule(parent_name)
    immediate_parent_class = type(parent).__name__
    # Go back recursively to find the first PreTrainedModel from which we inherit
    while not isinstance(parent, PreTrainedModel):
        parent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""
        parent = model.get_submodule(parent_name)
    # Get the exact XXXPreTrainedModel
    pretrained_parent_class = next(x.__name__ for x in type(parent).mro() if "PreTrainedModel" in x.__name__)
    # Some models directly inherit from `PreTrainedModel` instead of `XXXPreTrainedModel`
    if pretrained_parent_class == "PreTrainedModel":
        pretrained_parent_class = type(parent).__name__

    return name, immediate_parent_class, pretrained_parent_class


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()


def _config_supports_rope_scaling(config: PreTrainedConfig) -> bool:
    """Returns whether a certain model config supports RoPE scaling parameterization."""
    # Has rope_scaling -> model was designed with rope scaling in mind
    # Has rope_theta (and no rope_scaling) -> probably an older model, but should support rope scaling as well
    main_config_scales_rope = hasattr(config, "rope_parameters")

    # Axial rope doesn't scale as images usually have a pre-defined length
    # so the config will have no `max_position_embeddings` field defined
    if not hasattr(config, "max_position_embeddings"):
        main_config_scales_rope = False
    return main_config_scales_rope


def _set_config_rope_params(config: PreTrainedConfig, rope_params: dict) -> bool:
    """Recursively sets RoPE parameters on configs and subconfigs, by duplicating the same RoPE values."""
    config.rope_parameters = copy.deepcopy(getattr(config, "rope_parameters", {}) or {})

    # Nested rope parameters per layer type, not all models with `layer-types` use different RoPE thus we check `issubset`
    # Deepseekv4 has `layer_types` which are different from `_rope_type_labels`
    layer_types = getattr(config, "_rope_type_labels", getattr(config, "layer_types", None))
    if layer_types is not None and set(config.rope_parameters.keys()).issubset(layer_types):
        for layer_type in layer_types:
            # skip NoPE layers if any
            if config.rope_parameters[layer_type] is None:
                continue
            # Don't update gemma4 proportional rope, it is quite special and return `dim // 4` freqs
            if config.rope_parameters[layer_type].get("rope_type") != "proportional":
                config.rope_parameters.setdefault(layer_type, {})
                config.rope_parameters[layer_type].update(rope_params)
    else:
        config.rope_parameters.update(rope_params)

    for sub_config in config.sub_configs.keys():
        _set_config_rope_params(getattr(config, sub_config), rope_params)
    return config


class ModelTesterProtocol(Protocol):
    """What the mixins need from `self.model_tester` (the per-model `XxxModelTester`)."""

    def get_config(self) -> PreTrainedConfig: ...

    def prepare_config_and_inputs_for_common(self) -> tuple[PreTrainedConfig, dict[str, Any]]: ...


@runtime_checkable
class ModelTesterHost(Protocol):
    """
    Contract between the input-agnostic mixins and the class that hosts them (a `unittest.TestCase`
    composed from `ModelTesterMixin`). Defaults live ONCE in `ModelTesterMixin`; model test files override
    them as plain class attributes. Nothing is assigned here, so this class can never shadow those defaults.
    """

    # --- Lego constructors, provided by each model's test class ---
    model_tester: ModelTesterProtocol
    all_model_classes: ClassVar[tuple[type[PreTrainedModel], ...]]
    all_generative_model_classes: ClassVar[tuple[type[PreTrainedModel], ...]]

    # --- Capability flags (defaults in ModelTesterMixin) ---
    has_attentions: ClassVar[bool]
    _is_composite: ClassVar[bool]

    # --- Opt-out flags, one per mixin/group ---
    test_mismatched_shapes: ClassVar[bool]
    test_missing_keys: ClassVar[bool]
    test_can_init_all_missing_weights_before_2023: ClassVar[bool]
