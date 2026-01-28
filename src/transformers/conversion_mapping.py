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

from .core_model_loading import (
    Chunk,
    Concatenate,
    ErnieFuseAndSplitTextVisionExperts,
    Force16BytesAlignment,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
)
from .utils import is_torch_available


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


_MODEL_TO_CONVERSION_PATTERN = {
    # Mixtral-style MoE
    "mixtral": "mixtral",
    "minimax": "mixtral",
    "minimax_m2": "mixtral",
    # Qwen2-style MoE
    "qwen2_moe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "dots1": "qwen2_moe",
    "ernie4_5_moe": "qwen2_moe",
    "glm4_moe": "qwen2_moe",
    "glm4_moe_lite": "qwen2_moe",
    "glm4v_moe": "qwen2_moe",
    "longcat_flash": "qwen2_moe",
    "solar_open": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "qwen3_omni_moe": "qwen2_moe",
    "qwen3_omni_moe_thinker": "qwen2_moe",
    "qwen3_next": "qwen2_moe",
    "hunyuan_v1_moe": "qwen2_moe",
    "flex_olmo": "qwen2_moe",
    "olmoe": "qwen2_moe",
    "nomic_bert": "nomic_bert",
}


def _build_checkpoint_conversion_mapping():
    mapping = {
        "gpt_oss": [
            # NOTE: These converters are only applied if the model is being loaded from pre-dequantized checkpoint.
            # If you are dequantizing the model on the fly, these converters will be ignored because the tensors
            # that match these patterns are only created after dequantization.
            # That's not an issue for now since the dequantization converters already ensure 16 bytes alignment
            # by enforcing contiguity.
            WeightConverter(
                source_patterns="mlp.experts.gate_up_proj$",
                target_patterns="mlp.experts.gate_up_proj",
                operations=[Force16BytesAlignment()],
            ),
            WeightConverter(
                source_patterns="mlp.experts.down_proj$",
                target_patterns="mlp.experts.down_proj",
                operations=[Force16BytesAlignment()],
            ),
        ],
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
        "qwen3_vl_moe": [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1), Transpose(1, 2)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.*.down_proj.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[MergeModulelist(dim=0), Transpose(1, 2)],
            ),
        ],
        "phimoe": [
            WeightRenaming(".block_sparse_moe.gate", ".mlp.router"),
            WeightConverter(
                source_patterns=[
                    "block_sparse_moe.experts.*.w1.weight",
                    "block_sparse_moe.experts.*.w3.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="block_sparse_moe.experts.*.w2.weight",
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
        "ernie4_5_vl_moe": [
            # vision
            WeightRenaming("vision_model", "vision_tower"),
            # resampler
            WeightRenaming("spatial_linear.0", "spatial_linear.fc1"),
            WeightRenaming("spatial_linear.2", "spatial_linear.fc2"),
            WeightRenaming("spatial_linear.3", "spatial_linear.ln"),
            WeightRenaming("temporal_linear.0", "temporal_linear.fc1"),
            WeightRenaming("temporal_linear.2", "temporal_linear.fc2"),
            WeightRenaming("temporal_linear.3", "temporal_linear.ln"),
            # language model
            WeightRenaming(r"(?<!language_model\.)embed_tokens", "language_model.embed_tokens"),
            WeightRenaming(r"(?<!language_model\.)layers", "language_model.layers"),
            WeightConverter(
                source_patterns="mlp.gate.weight_1",
                target_patterns="mlp.vision_moe.gate.weight",
                operations=[Transpose(dim0=0, dim1=1)],
            ),
            WeightConverter(
                source_patterns="mlp.gate.weight",
                target_patterns="mlp.text_moe.gate.weight",
                operations=[Transpose(dim0=0, dim1=1)],
            ),
            WeightConverter(
                source_patterns=["mlp.moe_statics.e_score_correction_bias"],
                target_patterns=[
                    "mlp.text_moe.gate.moe_statics.e_score_correction_bias",
                    "mlp.vision_moe.gate.moe_statics.e_score_correction_bias",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=["experts.*.down_proj.weight"],
                target_patterns=[
                    "text_moe.experts.down_proj",
                    "vision_moe.experts.down_proj",
                ],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
            ),
            WeightConverter(
                source_patterns=[
                    "experts.*.gate_proj.weight",
                    "experts.*.up_proj.weight",
                ],
                target_patterns=[
                    "text_moe.experts.gate_up_proj",
                    "vision_moe.experts.gate_up_proj",
                ],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
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
        "nomic_bert": [
            # 1. Embeddings & Pooler (No 'bert.' prefix)
            WeightRenaming("embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
            WeightRenaming("embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight"),
            WeightRenaming("emb_ln.weight", "embeddings.LayerNorm.weight"),
            WeightRenaming("emb_ln.bias", "embeddings.LayerNorm.bias"),
            WeightRenaming("pooler.dense.weight", "pooler.dense.weight"),
            WeightRenaming("pooler.dense.bias", "pooler.dense.bias"),
            # 2. Encoder Layers Renaming (Regex works here for name changes)
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.attn\.out_proj\.weight", r"encoder.layer.\1.attention.output.dense.weight"
            ),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.attn\.out_proj\.bias", r"encoder.layer.\1.attention.output.dense.bias"
            ),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.mlp\.fc11\.weight", r"encoder.layer.\1.intermediate.gate_proj.weight"
            ),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.mlp\.fc11\.bias", r"encoder.layer.\1.intermediate.gate_proj.bias"
            ),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.mlp\.fc12\.weight", r"encoder.layer.\1.intermediate.up_proj.weight"
            ),
            WeightRenaming(r"encoder\.layers\.(\d+)\.mlp\.fc12\.bias", r"encoder.layer.\1.intermediate.up_proj.bias"),
            WeightRenaming(r"encoder\.layers\.(\d+)\.mlp\.fc2\.weight", r"encoder.layer.\1.output.dense.weight"),
            WeightRenaming(r"encoder\.layers\.(\d+)\.mlp\.fc2\.bias", r"encoder.layer.\1.output.dense.bias"),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.norm1\.weight", r"encoder.layer.\1.attention.output.LayerNorm.weight"
            ),
            WeightRenaming(
                r"encoder\.layers\.(\d+)\.norm1\.bias", r"encoder.layer.\1.attention.output.LayerNorm.bias"
            ),
            WeightRenaming(r"encoder\.layers\.(\d+)\.norm2\.weight", r"encoder.layer.\1.output.LayerNorm.weight"),
            WeightRenaming(r"encoder\.layers\.(\d+)\.norm2\.bias", r"encoder.layer.\1.output.LayerNorm.bias"),
            # 3. Explicit QKV Splits (Manually listed to avoid \1 or * errors)
            WeightConverter(
                source_patterns="encoder.layers.0.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.0.attention.self.q_proj.weight",
                    "encoder.layer.0.attention.self.k_proj.weight",
                    "encoder.layer.0.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.1.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.1.attention.self.q_proj.weight",
                    "encoder.layer.1.attention.self.k_proj.weight",
                    "encoder.layer.1.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.2.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.2.attention.self.q_proj.weight",
                    "encoder.layer.2.attention.self.k_proj.weight",
                    "encoder.layer.2.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.3.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.3.attention.self.q_proj.weight",
                    "encoder.layer.3.attention.self.k_proj.weight",
                    "encoder.layer.3.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.4.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.4.attention.self.q_proj.weight",
                    "encoder.layer.4.attention.self.k_proj.weight",
                    "encoder.layer.4.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.5.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.5.attention.self.q_proj.weight",
                    "encoder.layer.5.attention.self.k_proj.weight",
                    "encoder.layer.5.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.6.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.6.attention.self.q_proj.weight",
                    "encoder.layer.6.attention.self.k_proj.weight",
                    "encoder.layer.6.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.7.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.7.attention.self.q_proj.weight",
                    "encoder.layer.7.attention.self.k_proj.weight",
                    "encoder.layer.7.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.8.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.8.attention.self.q_proj.weight",
                    "encoder.layer.8.attention.self.k_proj.weight",
                    "encoder.layer.8.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.9.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.9.attention.self.q_proj.weight",
                    "encoder.layer.9.attention.self.k_proj.weight",
                    "encoder.layer.9.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.10.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.10.attention.self.q_proj.weight",
                    "encoder.layer.10.attention.self.k_proj.weight",
                    "encoder.layer.10.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="encoder.layers.11.attn.Wqkv.weight",
                target_patterns=[
                    "encoder.layer.11.attention.self.q_proj.weight",
                    "encoder.layer.11.attention.self.k_proj.weight",
                    "encoder.layer.11.attention.self.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
        ],
    }
    if hasattr(torch.nn.utils.parametrizations, "weight_norm"):
        mapping["legacy"] += [
            WeightRenaming(
                source_patterns=".weight_g$",
                target_patterns=".parametrizations.weight.original0",
            ),
            WeightRenaming(
                source_patterns=".weight_v$",
                target_patterns=".parametrizations.weight.original1",
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

    mapping["ernie4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] += [
        WeightRenaming("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias")
    ]
    mapping["minimax_m2"] = mapping["mixtral"].copy()
    mapping["minimax_m2"] += [
        WeightRenaming(".block_sparse_moe.e_score_correction_bias", ".mlp.e_score_correction_bias"),
    ]

    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():
        if model_type in mapping:
            continue
        mapping[model_type] = mapping[base_pattern].copy()

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))


def register_checkpoint_conversion_mapping(
    model_type: str, mapping: list[WeightConverter | WeightRenaming], overwrite: bool = False
) -> None:
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    if model_type in _checkpoint_conversion_mapping_cache and not overwrite:
        raise ValueError(f"Model type {model_type} already exists in the checkpoint conversion mapping.")
    _checkpoint_conversion_mapping_cache[model_type] = mapping


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
    "paddleocrvl",
    "ernie4_5_vl_moe",
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
        # NOTE: Since get_weight_conversions() only serve to dequantize, we would normally want to apply them first.
        # However, for now it's not possible to cascade converters (i.e., applying model-specific conversions on top
        # of tensors created by the dequantization conversions)
        # This means that if a model has model-specific conversions and is being dequantized, the model-specific conversion
        # that relies on tensors created by dequantization conversions will not be applied.
        # GptOss example: with Mxfp4Config(dequantize=True), Force16BytesAlignment converters are ignored because the tensors
        # "mlp.experts.gate_up_proj$" and "mlp.experts.down_proj$" are only created after dequantization conversions are applied.
        weight_conversions.extend(hf_quantizer.get_weight_conversions())

    return weight_conversions
