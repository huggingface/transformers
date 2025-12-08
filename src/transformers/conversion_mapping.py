# coding=utf-8
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

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from .core_model_loading import Concatenate, MergeModulelist, WeightConverter, WeightRenaming
from .utils import is_torch_available


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


def _build_checkpoint_conversion_mapping():
    mapping = {
        "mixtral": [
            WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
            WeightConverter(
                source_patterns=[
                    "block_sparse_moe.experts.*.w1.weight",
                    "block_sparse_moe.experts.*.w3.weight",
                ],  # you give me a list of 2 keys, I collect a list of a list of tensors
                target_patterns="mlp.experts.gate_up_proj",  # target key gets the list of two tensors
                operations=[
                    MergeModulelist(
                        dim=0
                    ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                    Concatenate(dim=1),  # each process has 2 tensors, gate and up, we concat them into gate_up
                ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
            ),
            WeightConverter(
                source_patterns=[
                    "block_sparse_moe.experts.*.w2.weight",
                ],
                target_patterns="mlp.experts.down_proj",  # target key gets the list of two tensors
                operations=[
                    MergeModulelist(
                        dim=0
                    ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
            ),
        ],
        "qwen2_moe": [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.down_proj.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "phimoe": [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.w1.weight",
                    "mlp.experts.*.w3.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.w2.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "lfm2_moe": [
            WeightConverter(
                source_patterns=[
                    "feed_forward.experts.*.w1.weight",
                    "feed_forward.experts.*.w3.weight",
                ],
                target_patterns="feed_forward.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="feed_forward.experts.*.w2.weight",
                target_patterns="feed_forward.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "jamba": [
            WeightConverter(
                source_patterns=[
                    "feed_forward.experts.*.gate_proj.weight",
                    "feed_forward.experts.*.up_proj.weight",
                ],
                target_patterns="feed_forward.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="feed_forward.experts.*.down_proj.weight",
                target_patterns="feed_forward.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "timm_wrapper": [
            # Simply add the prefix `timm_model`
            # TODO: Would be probably much cleaner with a `add_prefix` argument in WeightRenaming
            WeightRenaming(
                source_patterns=r"(.+)",
                target_patterns=r"timm_model.\1",
            )
        ],
        "legacy": [
            WeightRenaming(
                source_patterns="LayerNorm.gamma",
                target_patterns="LayerNorm.weight",
            ),
            WeightRenaming(
                source_patterns="LayerNorm.beta",
                target_patterns="LayerNorm.bias",
            ),
        ],
    }
    if hasattr(torch.nn.utils.parametrizations, "weight_norm"):
        mapping["legacy"] += [
            WeightRenaming(
                source_patterns="weight_g",
                target_patterns="parametrizations.weight.original0",
            ),
            WeightRenaming(
                source_patterns="weight_v",
                target_patterns="parametrizations.weight.original1",
            ),
        ]
    else:
        mapping["legacy"] += [
            WeightRenaming(
                source_patterns="parametrizations.weight.original0",
                target_patterns="weight_g",
            ),
            WeightRenaming(
                source_patterns="parametrizations.weight.original1",
                target_patterns="weight_v",
            ),
        ]

    mapping["deepseek_v2"] = mapping["qwen2_moe"].copy()
    mapping["deepseek_v3"] = mapping["qwen2_moe"].copy()
    mapping["dots1"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["glm4_moe"] = mapping["qwen2_moe"].copy()
    mapping["glm4v_moe"] = mapping["qwen2_moe"].copy()
    mapping["longcat_flash"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_omni_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_next"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_vl_moe"] = mapping["qwen2_moe"].copy()
    mapping["hunyuan_v1_moe"] = mapping["qwen2_moe"].copy()
    mapping["minimax"] = mapping["mixtral"].copy()
    mapping["flex_olmo"] = mapping["qwen2_moe"].copy()
    mapping["olmoe"] = mapping["qwen2_moe"].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))


# DO NOT MODIFY, KEPT FOR BC ONLY
VLMS = [
    "aria",
    "ayavision",
    "colpali",
    "emu3",
    "fuyu",
    "gotocr2",
    "gemma3",
    "internvl",
    "llava",  # all llava prefixed models fall under this check
    "mistral3",
    "mllama",
    "paligemma",
    "shieldgemma2",
    "qwen2vl",
    "qwen2_5_vl",
    "videollava",
    "vipllava",
    "sam3_video",
    "sam3",
    "sam3_tracker",
    "sam3_tracker_video",
]


def get_model_conversion_mapping(
    model: PreTrainedModel,
    key_mapping: dict[str, str] | None = None,
    hf_quantizer: HfQuantizer | None = None,
    add_legacy: bool = True,
) -> list[WeightConverter | WeightRenaming]:
    """
    For a given `model`, obtain the weight conversion mapping if any are registered either as a simple renaming
    `_checkpoint_conversion_mapping` class argument, or in the general WeightConverter mapping.
    """
    weight_conversions = []

    # Load models with explicit, user-provided key mapping
    if key_mapping is not None:
        weight_conversions = [WeightRenaming(source_patterns=k, target_patterns=v) for k, v in key_mapping.items()]
    elif any(
        allowed_name in class_name.__name__.lower()
        for class_name in model.__class__.__mro__[:-1]
        for allowed_name in VLMS
    ):
        weight_conversions = [
            WeightRenaming(source_patterns=k, target_patterns=v)
            for k, v in model._checkpoint_conversion_mapping.items()
        ]

    # TODO: should be checked recursively on submodels!!
    model_type = getattr(model.config, "model_type", None)
    if model_type is not None:
        model_specific_conversions = get_checkpoint_conversion_mapping(model_type)
        if model_specific_conversions is not None:
            weight_conversions.extend(model_specific_conversions)

    if add_legacy:
        weight_conversions.extend(get_checkpoint_conversion_mapping("legacy"))

    # Add the ones from the quantizer as well if provided
    if hf_quantizer is not None:
        weight_conversions.extend(hf_quantizer.get_weight_conversions())

    return weight_conversions
