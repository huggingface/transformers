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

from copy import deepcopy

from .core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    ModulelistSplitAndFuse,
    WeightConverter,
    WeightRenaming,
)
from .utils import is_torch_available


if is_torch_available():
    import torch


def _build_checkpoint_conversion_mapping():
    mapping = {
        "mixtral": [
            WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
            WeightConverter(
                source_keys=[
                    "block_sparse_moe.experts.*.w1.weight",
                    "block_sparse_moe.experts.*.w3.weight",
                ],  # you give me a list of 2 keys, I collect a list of a list of tensors
                target_keys="mlp.experts.gate_up_proj",  # target key gets the list of two tensors
                operations=[
                    MergeModulelist(
                        dim=0
                    ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                    Concatenate(dim=1),  # each process has 2 tensors, gate and up, we concat them into gate_up
                ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
            ),
            WeightConverter(
                source_keys=[
                    "block_sparse_moe.experts.*.w2.weight",
                ],
                target_keys="mlp.experts.down_proj",  # target key gets the list of two tensors
                operations=[
                    MergeModulelist(
                        dim=0
                    ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
            ),
        ],
        "qwen2_moe": [
            WeightConverter(
                source_keys=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_keys="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_keys=["mlp.experts.*.down_proj.weight"],
                target_keys="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "ernie4_5_vl": [
            # vision
            WeightRenaming("^vision_model", "model.vision_tower"),
            # resampler
            WeightRenaming("resampler_model.spatial_linear.0", "resampler_model.spatial_linear.fc1"),
            WeightRenaming("resampler_model.spatial_linear.2", "resampler_model.spatial_linear.fc2"),
            WeightRenaming("resampler_model.spatial_linear.3", "resampler_model.spatial_linear.ln"),
            WeightRenaming("resampler_model.temporal_linear.0", "resampler_model.temporal_linear.fc1"),
            WeightRenaming("resampler_model.temporal_linear.2", "resampler_model.temporal_linear.fc2"),
            WeightRenaming("resampler_model.temporal_linear.3", "resampler_model.temporal_linear.ln"),
            # language model
            WeightRenaming("^model.norm", "model.language_model.norm"),
            WeightRenaming("^model.embed_tokens", "model.language_model.embed_tokens"),
            WeightRenaming("^model.layers", "model.language_model.layers"),
            WeightRenaming("mlp.gate.weight", "mlp.text_moe.gate.weight"),
            WeightRenaming("mlp.gate.weight_1", "mlp.vision_moe.gate.weight"),
            WeightConverter(
                source_keys=["mlp.moe_statics.e_score_correction_bias"],
                target_keys=[
                    "mlp.text_moe.gate.moe_statics.e_score_correction_bias",
                    "mlp.vision_moe.gate.moe_statics.e_score_correction_bias"
                ],
                operations=[Chunk(dim=0, chunks=2)]
            ),
            WeightConverter(
                source_keys=["experts.*.down_proj.weight"],
                target_keys=[
                    "text_moe.experts.down_proj",
                    "vision_moe.experts.down_proj",
                ],
                operations=[ModulelistSplitAndFuse(stack_dim=0, concat_dim=1)]
            ),
            WeightConverter(
                source_keys=[
                    "experts.*.gate_proj.weight",
                    "experts.*.up_proj.weight",
                ],
                target_keys=[
                    "text_moe.experts.gate_up_proj",
                    "vision_moe.experts.gate_up_proj",
                ],
                operations=[ModulelistSplitAndFuse(stack_dim=0, concat_dim=1)]
            ),
        ],
        "legacy": [
            WeightRenaming(
                source_keys="LayerNorm.gamma",
                target_keys="LayerNorm.weight",
            ),
            WeightRenaming(
                source_keys="LayerNorm.beta",
                target_keys="LayerNorm.bias",
            ),
        ],
    }
    if hasattr(torch.nn.utils.parametrizations, "weight_norm"):
        mapping["legacy"] += [
            WeightRenaming(
                source_keys="weight_g",
                target_keys="parametrizations.weight.original0",
            ),
            WeightRenaming(
                source_keys="weight_v",
                target_keys="parametrizations.weight.original1",
            ),
        ]
    else:
        mapping["legacy"] += [
            WeightRenaming(
                source_keys="parametrizations.weight.original0",
                target_keys="weight_g",
            ),
            WeightRenaming(
                source_keys="parametrizations.weight.original1",
                target_keys="weight_v",
            ),
        ]

    mapping["phimoe"] = mapping["mixtral"].copy()
    mapping["deepseek_v2"] = mapping["qwen2_moe"].copy()
    mapping["deepseek_v3"] = mapping["qwen2_moe"].copy()
    mapping["dot1"] = mapping["qwen2_moe"].copy()
    mapping["ernie_4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["glm4_moe"] = mapping["qwen2_moe"].copy()
    mapping["glm4v_moe"] = mapping["qwen2_moe"].copy()
    mapping["jamba"] = mapping["qwen2_moe"].copy()
    mapping["lfm2_moe"] = mapping["mixtral"].copy()
    mapping["long_cat_flash"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_omni_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_next"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_vl_moe"] = mapping["qwen2_moe"].copy()
    mapping["hunyuan_v1_moe"] = mapping["qwen2_moe"].copy()
    mapping["minimax"] = mapping["mixtral"].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    globals()["_checkpoint_conversion_mapping"] = _checkpoint_conversion_mapping_cache
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type, None))
