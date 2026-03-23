# Copyright (C) 2025 the HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING

import torch
from torch import nn

from .activations import ACT2FN
from .core_model_loading import Concatenate, WeightConverter
from .monkey_patching import register_patch_mapping


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel


def _make_fused_mlp(original_cls):
    """
    Create a fused MLP class from an original that has separate ``gate_proj`` and
    ``up_proj``.  The returned subclass replaces those with a single ``gate_up_proj``
    and overrides ``_compute_gate_up`` to use it.
    """

    class FusedMLP(original_cls):
        _weight_converter = WeightConverter(
            source_patterns=[".gate_proj.weight$", ".up_proj.weight$"],
            target_patterns=".gate_up_proj.weight$",
            operations=[Concatenate(dim=0)],
        )

        def __init__(self, config):
            super().__init__(config)

            del self.gate_proj
            del self.up_proj
            self.gate_up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size * 2, bias=config.mlp_bias
            )
            self.act_fn = ACT2FN[config.hidden_act + "_and_mul"]

        def _compute_gate_up(self, x):
            return self.act_fn(self.gate_up_proj(x))

    FusedMLP.__name__ = f"Fused{original_cls.__name__}"
    FusedMLP.__qualname__ = f"Fused{original_cls.__qualname__}"
    return FusedMLP


def _make_fused_attention(original_cls):
    """
    Create a fused attention class from an original that has separate ``q_proj``,
    ``k_proj``, ``v_proj``.  The returned subclass replaces those with a single
    ``qkv_proj`` and overrides ``_project_qkv`` to split the fused output.
    """

    class FusedAttention(original_cls):
        _weight_converter = WeightConverter(
            source_patterns=[".q_proj.weight$", ".k_proj.weight$", ".v_proj.weight$"],
            target_patterns=".qkv_proj.weight$",
            operations=[Concatenate(dim=0)],
        )

        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)

            del self.q_proj
            del self.k_proj
            del self.v_proj

            self.q_size = config.num_attention_heads * self.head_dim
            self.kv_size = config.num_key_value_heads * self.head_dim
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                self.q_size + 2 * self.kv_size,
                bias=config.attention_bias,
            )

        def _project_qkv(self, hidden_states, hidden_shape):
            qkv = self.qkv_proj(hidden_states)
            q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
            return (
                q.view(hidden_shape).transpose(1, 2),
                k.view(hidden_shape).transpose(1, 2),
                v.view(hidden_shape).transpose(1, 2),
            )

    FusedAttention.__name__ = f"Fused{original_cls.__name__}"
    FusedAttention.__qualname__ = f"Fused{original_cls.__qualname__}"
    return FusedAttention


_fusion_cache: dict[type, dict[str, type[nn.Module]]] = {}


def _discover_fusable_classes(cls: "type[PreTrainedModel]", config) -> dict[str, type[nn.Module]]:
    """
    Instantiate *cls* on the meta device and walk ``model.modules()`` to find
    ``nn.Module`` subclasses that expose ``_compute_gate_up`` (MLP fusion) or
    ``_project_qkv`` (attention fusion).  Returns a mapping from original
    class name → fused replacement class.
    """
    if cls in _fusion_cache:
        return _fusion_cache[cls]

    with torch.device("meta"):
        model = cls(config)

    seen: set[type] = set()
    patch_mapping: dict[str, type[nn.Module]] = {}
    for submodule in model.modules():
        subcls = type(submodule)
        if subcls in seen:
            continue
        seen.add(subcls)
        if hasattr(subcls, "_compute_gate_up"):
            patch_mapping[subcls.__name__] = _make_fused_mlp(subcls)
        elif hasattr(subcls, "_project_qkv"):
            patch_mapping[subcls.__name__] = _make_fused_attention(subcls)

    _fusion_cache[cls] = patch_mapping
    return patch_mapping


def _update_tp_plan(tp_plan: dict[str, str]) -> None:
    """Rewrite *tp_plan* in-place to reflect fused projections."""
    for q_key in [k for k in tp_plan if k.endswith(".q_proj")]:
        prefix = q_key.rsplit(".q_proj", 1)[0]
        k_key, v_key = f"{prefix}.k_proj", f"{prefix}.v_proj"
        if k_key in tp_plan and v_key in tp_plan:
            del tp_plan[q_key], tp_plan[k_key], tp_plan[v_key]
            tp_plan[f"{prefix}.qkv_proj"] = "colwise_qkv"

    for gate_key in [k for k in tp_plan if k.endswith(".gate_proj")]:
        prefix = gate_key.rsplit(".gate_proj", 1)[0]
        up_key = f"{prefix}.up_proj"
        if up_key in tp_plan:
            del tp_plan[gate_key], tp_plan[up_key]
            tp_plan[f"{prefix}.gate_up_proj"] = "colwise_merged"


def register_fusion_patches(cls: "type[PreTrainedModel]", config) -> None:
    """
    Register all fusion-related changes for *cls*:

    1. Monkey-patches into the global patch mapping (for ``apply_patches()``)
    2. Weight converters into the checkpoint conversion mapping
       (for ``get_model_conversion_mapping()``)
    3. TP plan updates on ``config_class.base_model_tp_plan``
    """
    from .conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping

    fusable_classes = _discover_fusable_classes(cls, config)
    if not fusable_classes:
        return

    # 1. monkey-patches
    register_patch_mapping(fusable_classes)

    # 2. weight converters
    config_class = getattr(cls, "config_class", None)
    model_type = getattr(config_class, "model_type", None) if config_class is not None else None
    if model_type is not None:
        converters = [fused_cls._weight_converter for fused_cls in fusable_classes.values()]
        existing = get_checkpoint_conversion_mapping(model_type)
        if existing is not None:
            converters = existing + converters
        register_checkpoint_conversion_mapping(model_type, converters, overwrite=True)

    # 3. tp plan
    if config_class is not None and config_class.base_model_tp_plan is not None:
        _update_tp_plan(config_class.base_model_tp_plan)
