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

from .core_model_loading import Concatenate, MergeModulelist, WeightConverter


_checkpoint_conversion_mapping = {
    "mixtral": [
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
        # WeightConverter(
        #     ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        #     "self_attn.qkv_proj",
        #     Concatenate(dim=0),  # more like stack?
        # ),
        WeightConverter("*.block_sparse_moe.", "*.mlp."),
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
}
_checkpoint_conversion_mapping["phimoe"] = _checkpoint_conversion_mapping["mixtral"].copy()
_checkpoint_conversion_mapping["deepseek_v2"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["deepseek_v3"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["dot1"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["ernie_4_5_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["glm4_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["glm4v_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["jamba"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["lfm2_moe"] = _checkpoint_conversion_mapping["mixtral"].copy()
_checkpoint_conversion_mapping["long_cat_flash"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["qwen3_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["qwen3_omni_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["qwen3_next"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["qwen3_vl_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["hunyuan_v1_moe"] = _checkpoint_conversion_mapping["qwen2_moe"].copy()
_checkpoint_conversion_mapping["minimax"] = _checkpoint_conversion_mapping["mixtral"].copy()
