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

    mapping["deepseek_v2"] = mapping["qwen2_moe"].copy()
    mapping["deepseek_v3"] = mapping["qwen2_moe"].copy()
    mapping["dots1"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] += [
        WeightRenaming("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias")
    ]
    mapping["glm4_moe"] = mapping["qwen2_moe"].copy()
    mapping["glm4_moe_lite"] = mapping["qwen2_moe"].copy()
    mapping["glm4v_moe"] = mapping["qwen2_moe"].copy()
    mapping["longcat_flash"] = mapping["qwen2_moe"].copy()
    mapping["solar_open"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_omni_moe"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_omni_moe_thinker"] = mapping["qwen2_moe"].copy()
    mapping["qwen3_next"] = mapping["qwen2_moe"].copy()
    mapping["hunyuan_v1_moe"] = mapping["qwen2_moe"].copy()
    mapping["minimax"] = mapping["mixtral"].copy()
    mapping["deepseek_ocr"] = [
        WeightRenaming(r"\\\.", "."),  # for renaming issues... namely too many escapes
        WeightRenaming(r"^model\.model\.", "model."),
        WeightRenaming(r"^model\.sam_model\.blocks\.", "sam_model.layers."),
        WeightRenaming(r"^sam_model\.blocks\.", "sam_model.layers."),
        WeightRenaming(r"^model\.sam_model\.blocks\..*\.norm1\.", "sam_model.layers.layer_norm1."),
        WeightRenaming(r"^sam_model\.blocks\..*\.norm1\.", "sam_model.layers.layer_norm1."),
        WeightRenaming(r"^model\.sam_model\.blocks\..*\.norm2\.", "sam_model.layers.layer_norm2."),
        WeightRenaming(r"^sam_model\.blocks\..*\.norm2\.", "sam_model.layers.layer_norm2."),
        WeightRenaming(r"^sam_model\.layers\.(\d+)\.norm1\.", r"model.sam_model.layers.\1.layer_norm1."),
        WeightRenaming(r"^sam_model\.layers\.(\d+)\.norm2\.", r"model.sam_model.layers.\1.layer_norm2."),
        WeightRenaming(r"^model\.sam_model\.patch_embed\.proj\.", "sam_model.patch_embed.projection."),
        WeightRenaming(r"^sam_model\.patch_embed\.proj\.", "sam_model.patch_embed.projection."),
        # vision stuff
        WeightRenaming(r"^model\.sam_model\.neck\.0\.", "sam_model.neck.conv1."),
        WeightRenaming(r"^sam_model\.neck\.0\.", "sam_model.neck.conv1."),
        WeightRenaming(r"^model\.sam_model\.neck\.1\.", "sam_model.neck.layer_norm1."),
        WeightRenaming(r"^sam_model\.neck\.1\.", "sam_model.neck.layer_norm1."),
        WeightRenaming(r"^model\.sam_model\.neck\.2\.", "sam_model.neck.conv2."),
        WeightRenaming(r"^sam_model\.neck\.2\.", "sam_model.neck.conv2."),
        WeightRenaming(r"^model\.sam_model\.neck\.3\.", "sam_model.neck.layer_norm2."),
        WeightRenaming(r"^sam_model\.neck\.3\.", "sam_model.neck.layer_norm2."),
        WeightRenaming(r"^model\.vision_model\.transformer\.layers\.", "clip_model.vision_model.encoder.layers."),
        WeightRenaming(r"^vision_model\.transformer\.layers\.", "clip_model.vision_model.encoder.layers."),
        WeightRenaming(r"^model\.vision_model\.embeddings\.", "clip_model.vision_model.embeddings."),
        WeightRenaming(r"^vision_model\.embeddings\.", "clip_model.vision_model.embeddings."),
        WeightRenaming(r"^model\.vision_model\.pre_layrnorm\.", "clip_model.vision_model.pre_layrnorm."),
        WeightRenaming(r"^vision_model\.pre_layrnorm\.", "clip_model.vision_model.pre_layrnorm."),
        WeightRenaming(r"^model\.vision_model\.", "clip_model.vision_model."),
        WeightRenaming(r"^vision_model\.", "clip_model.vision_model."),
        WeightRenaming(r"^model\.projector\.", "multi_modal_projector."),
        WeightRenaming(r"^projector\.", "multi_modal_projector."),
        # embeddings,layers, experts
        WeightRenaming(r"^model\.embed_tokens\.", "language_model.embed_tokens."),
        WeightRenaming(r"^embed_tokens\.", "language_model.embed_tokens."),
        WeightRenaming(r"^model\.norm\.", "language_model.norm."),
        WeightRenaming(r"^norm\.", "language_model.norm."),
        WeightRenaming(r"^model\.layers\.", "language_model.layers."),
        WeightRenaming(r"^layers\.", "language_model.layers."),
        WeightConverter(
            source_patterns="qkv_proj",
            target_patterns=["q_proj", "k_proj", "v_proj"],
            operations=[Chunk(dim=0)],
        ),
    ]
    mapping["deepseek_vl_v2"] = mapping["deepseek_ocr"].copy()
    mapping["minimax_m2"] = mapping["mixtral"].copy()
    mapping["minimax_m2"] += [
        WeightRenaming(".block_sparse_moe.e_score_correction_bias", ".mlp.e_score_correction_bias"),
    ]
    mapping["flex_olmo"] = mapping["qwen2_moe"].copy()
    mapping["olmoe"] = mapping["qwen2_moe"].copy()

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
    "deepseek_ocr",
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
