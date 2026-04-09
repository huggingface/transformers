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
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
)


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


_MODEL_TO_CONVERSION_PATTERN = {
    # Mixtral-style MoE
    "minimax": "mixtral",
    "minimax_m2": "mixtral",
    # Qwen2-style MoE
    "afmoe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "dots1": "qwen2_moe",
    "ernie4_5_moe": "qwen2_moe",
    "glm4_moe": "qwen2_moe",
    "glm4_moe_lite": "qwen2_moe",
    "glm_moe_dsa": "qwen2_moe",
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
    "exaone_moe": "qwen2_moe",
    "rt_detr_v2": "rt_detr",
    "pp_doclayout_v2": "rt_detr",
    "pp_doclayout_v3": "rt_detr",
    "paligemma": "llava",
    "aya_vision": "llava",
    "fuyu": "llava",
    "got_ocr2": "llava",
    "shieldgemma2": "llava",
    "gemma3": "llava",
    "internvl": "llava",
    "llava_next": "llava",
    "llava_next_video": "llava",
    "llava_onevision": "llava",
    "vipllava": "llava",
    "video_llava": "llava",
    "mistral3": "llava",
    "mllama": "llava",
    "qwen2_5_vl": "qwen2_vl",
    "sam3_tracker_video": "sam3_tracker",
    "pp_chart2table": "llava",
    "gemma3n_text": "qwen3_5_text",
    "qwen3_5_moe_text": "qwen3_5_text",
}


def _build_checkpoint_conversion_mapping():
    mapping = {
        "llava": [
            WeightRenaming(source_patterns=r"language_model.model", target_patterns="language_model"),
            WeightRenaming(source_patterns=r"language_model.lm_head", target_patterns="lm_head"),
        ],
        "emu3": [
            WeightRenaming(source_patterns=r"text_model.model", target_patterns="text_model"),
            WeightRenaming(source_patterns=r"text_model.lm_head", target_patterns="lm_head"),
        ],
        "paddleocr_vl": [
            WeightRenaming(source_patterns=r"mlp_AR", target_patterns="model.projector"),
            WeightRenaming(
                source_patterns=r"^model(?!(\.visual|\.projector|\.language_model))",
                target_patterns="model.language_model",
            ),
        ],
        "qwen2_vl": [
            WeightRenaming(
                source_patterns=r"(?<!_)model(?!\.(language_model|visual))", target_patterns="model.language_model"
            ),
        ],
        "colqwen2": [
            WeightRenaming(source_patterns=r"vlm.model", target_patterns="vlm"),
            WeightRenaming(source_patterns=r"vlm(?!\.(language_model|visual))", target_patterns="vlm.language_model"),
        ],
        "timm_wrapper": [
            # Simply add the prefix `timm_model`. Similar to `base_model_prefix` but also removes prefix
            # when saving. TODO: Would be probably much cleaner with a `add_prefix` argument in WeightRenaming
            # Note: we don't add `timm_model` when it is part of a bigger VLM, because they already have `timm_model`
            # saved in state dict keys. Thus the look behind check. Should be fixed by proper `add_prefix`!
            WeightRenaming(
                source_patterns=r"^(?!(?:model\.|backbone\.|tower\.))(.+)$",
                target_patterns=r"timm_model.\1",
            )
        ],
        "pi0": [
            WeightRenaming(source_patterns=r"state_proj", target_patterns="embed_action_time.state_proj"),
            WeightRenaming(source_patterns=r"action_in_proj", target_patterns="embed_action_time.action_in_proj"),
            WeightRenaming(
                source_patterns=r"action_time_mlp_in", target_patterns="embed_action_time.action_time_mlp_in"
            ),
            WeightRenaming(
                source_patterns=r"action_time_mlp_out", target_patterns="embed_action_time.action_time_mlp_out"
            ),
            WeightRenaming(source_patterns=r"^paligemma_with_expert.paligemma.model", target_patterns="model.vlm"),
            WeightRenaming(source_patterns=r"^paligemma_with_expert.gemma_expert.model", target_patterns="model.dit"),
            # Weight on the hub have only `lm_head` saved, but PI0 doesn't create any lm-head initialized!
            WeightRenaming(
                source_patterns=r"^paligemma_with_expert.gemma_expert.lm_head",
                target_patterns="model.dit.embed_tokens",
            ),
            WeightRenaming(
                source_patterns=r"^paligemma_with_expert.paligemma.lm_head",
                target_patterns="model.vlm.language_model.embed_tokens",
            ),
        ],
        "dinov3_convnext": [WeightRenaming(r"(?<!model\.)stages", r"model.stages")],
        "dinov3_vit": [WeightRenaming(r"(?<!model\.)layer.", r"model.layer.")],
        "timesfm2_5": [
            WeightRenaming("ff0", "fc1"),
            WeightRenaming("ff1", "fc2"),
        ],
        "olmo_hybrid": [
            WeightRenaming("attention_layer_norm", "input_layernorm"),
            WeightRenaming("feedforward_layer_norm", "post_attention_layernorm"),
        ],
        "qwen3_5_text": [
            # Note: the lookbehind on the target is to avoid replacing bigger matches when the model is a submodel of
            # the ForConditionalGeneration model
            WeightRenaming(source_patterns=r"^model.language_model.", target_patterns=r"^model.(?!language_model.)"),
        ],
        "sam3_tracker": [
            WeightRenaming(
                source_patterns=r"detector_model.vision_encoder.backbone.", target_patterns="vision_encoder.backbone."
            ),
            WeightRenaming(source_patterns=r"tracker_neck.", target_patterns="vision_encoder.neck."),
        ],
        "t5gemma2_encoder": [
            WeightRenaming(r"(?<!decoder\.)(?<!text_model\.)embed_tokens\.", "text_model.embed_tokens."),
            WeightRenaming(r"(?<!decoder\.)(?<!text_model\.)(?<!layer)(?<!_)norm\.", "text_model.norm."),
            WeightRenaming(r"(?<!vision_model.encoder\.)(?<!decoder\.)(?<!text_model\.)layers.", "text_model.layers."),
        ],
        "mixtral": [
            WeightRenaming(".block_sparse_moe.", ".mlp."),
            WeightConverter(
                source_patterns=[
                    ".experts.*.w1.weight",
                    ".experts.*.w3.weight",
                ],  # you give me a list of 2 keys, I collect a list of a list of tensors
                target_patterns=".experts.gate_up_proj",  # target key gets the list of two tensors
                operations=[
                    MergeModulelist(
                        dim=0
                    ),  # each process has two lists of tensors, we cat each list. -> we end up with 2 tensors
                    Concatenate(dim=1),  # each process has 2 tensors, gate and up, we concat them into gate_up
                ],  # we want the loading to add this shard operation here. Though we can't shard after concats and merge, needs to be first
            ),
            WeightConverter(
                source_patterns=[
                    ".experts.*.w2.weight",
                ],
                target_patterns=".experts.down_proj",  # target key gets the list of two tensors
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
                source_patterns="mlp.experts.gate_up_proj",
                target_patterns="mlp.experts.gate_up_proj",
                operations=[Transpose(1, 2, check_dims=True)],
            ),
            WeightConverter(
                source_patterns="mlp.experts.down_proj",
                target_patterns="mlp.experts.down_proj",
                operations=[Transpose(1, 2, check_dims=True)],
            ),
        ],
        "phimoe": [
            WeightRenaming(".block_sparse_moe.", ".mlp."),
            WeightRenaming(".gate.weight", ".router.weight"),
            WeightConverter(
                source_patterns=[
                    ".experts.*.w1.weight",
                    ".experts.*.w3.weight",
                ],
                target_patterns=".experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns=".experts.*.w2.weight",
                target_patterns=".experts.down_proj",
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
            WeightRenaming(r"(?<!_)(?<!\w)norm\.", "language_model.norm."),
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
        "detr": [
            WeightRenaming("backbone.conv_encoder", "backbone"),
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
            # `DetrForSegmentation`
            WeightRenaming("bbox_attention.q_linear", "bbox_attention.q_proj"),
            WeightRenaming("bbox_attention.k_linear", "bbox_attention.k_proj"),
            # Mask head refactor
            WeightRenaming("mask_head.lay1", "mask_head.conv1.conv"),
            WeightRenaming("mask_head.gn1", "mask_head.conv1.norm"),
            WeightRenaming("mask_head.lay2", "mask_head.conv2.conv"),
            WeightRenaming("mask_head.gn2", "mask_head.conv2.norm"),
            WeightRenaming("mask_head.adapter1", "mask_head.fpn_stages.0.fpn_adapter"),
            WeightRenaming("mask_head.lay3", "mask_head.fpn_stages.0.refine.conv"),
            WeightRenaming("mask_head.gn3", "mask_head.fpn_stages.0.refine.norm"),
            WeightRenaming("mask_head.adapter2", "mask_head.fpn_stages.1.fpn_adapter"),
            WeightRenaming("mask_head.lay4", "mask_head.fpn_stages.1.refine.conv"),
            WeightRenaming("mask_head.gn4", "mask_head.fpn_stages.1.refine.norm"),
            WeightRenaming("mask_head.adapter3", "mask_head.fpn_stages.2.fpn_adapter"),
            WeightRenaming("mask_head.lay5", "mask_head.fpn_stages.2.refine.conv"),
            WeightRenaming("mask_head.gn5", "mask_head.fpn_stages.2.refine.norm"),
            WeightRenaming("mask_head.out_lay", "mask_head.output_conv"),
        ],
        "rt_detr": [
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
            WeightRenaming(r"encoder.encoder.(\d+).layers", r"encoder.aifi.\1.layers"),
        ],
        "conditional_detr": [
            WeightRenaming("backbone.conv_encoder", "backbone"),
            WeightRenaming("self_attn.out_proj", "self_attn.o_proj"),
            WeightRenaming("encoder_attn.out_proj", "encoder_attn.o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
            # Decoder self-attention projections moved into self_attn module
            WeightRenaming(r"decoder.layers.(\d+).sa_qcontent_proj", r"decoder.layers.\1.self_attn.q_content_proj"),
            WeightRenaming(r"decoder.layers.(\d+).sa_qpos_proj", r"decoder.layers.\1.self_attn.q_pos_proj"),
            WeightRenaming(r"decoder.layers.(\d+).sa_kcontent_proj", r"decoder.layers.\1.self_attn.k_content_proj"),
            WeightRenaming(r"decoder.layers.(\d+).sa_kpos_proj", r"decoder.layers.\1.self_attn.k_pos_proj"),
            WeightRenaming(r"decoder.layers.(\d+).sa_v_proj", r"decoder.layers.\1.self_attn.v_proj"),
            # Decoder cross-attention projections moved into encoder_attn module
            WeightRenaming(r"decoder.layers.(\d+).ca_qcontent_proj", r"decoder.layers.\1.encoder_attn.q_content_proj"),
            WeightRenaming(r"decoder.layers.(\d+).ca_qpos_proj", r"decoder.layers.\1.encoder_attn.q_pos_proj"),
            WeightRenaming(r"decoder.layers.(\d+).ca_kcontent_proj", r"decoder.layers.\1.encoder_attn.k_content_proj"),
            WeightRenaming(r"decoder.layers.(\d+).ca_kpos_proj", r"decoder.layers.\1.encoder_attn.k_pos_proj"),
            WeightRenaming(r"decoder.layers.(\d+).ca_v_proj", r"decoder.layers.\1.encoder_attn.v_proj"),
            WeightRenaming(
                r"decoder.layers.(\d+).ca_qpos_sine_proj", r"decoder.layers.\1.encoder_attn.q_pos_sine_proj"
            ),
            # The rest of patterns are used only in `ConditionalDetrForSegmentation`
            WeightRenaming("bbox_attention.q_linear", "bbox_attention.q_proj"),
            WeightRenaming("bbox_attention.k_linear", "bbox_attention.k_proj"),
            # Mask head refactor
            WeightRenaming("mask_head.lay1", "mask_head.conv1.conv"),
            WeightRenaming("mask_head.gn1", "mask_head.conv1.norm"),
            WeightRenaming("mask_head.lay2", "mask_head.conv2.conv"),
            WeightRenaming("mask_head.gn2", "mask_head.conv2.norm"),
            WeightRenaming("mask_head.adapter1", "mask_head.fpn_stages.0.fpn_adapter"),
            WeightRenaming("mask_head.lay3", "mask_head.fpn_stages.0.refine.conv"),
            WeightRenaming("mask_head.gn3", "mask_head.fpn_stages.0.refine.norm"),
            WeightRenaming("mask_head.adapter2", "mask_head.fpn_stages.1.fpn_adapter"),
            WeightRenaming("mask_head.lay4", "mask_head.fpn_stages.1.refine.conv"),
            WeightRenaming("mask_head.gn4", "mask_head.fpn_stages.1.refine.norm"),
            WeightRenaming("mask_head.adapter3", "mask_head.fpn_stages.2.fpn_adapter"),
            WeightRenaming("mask_head.lay5", "mask_head.fpn_stages.2.refine.conv"),
            WeightRenaming("mask_head.gn5", "mask_head.fpn_stages.2.refine.norm"),
            WeightRenaming("mask_head.out_lay", "mask_head.output_conv"),
        ],
        "deformable_detr": [
            WeightRenaming("backbone.conv_encoder", "backbone"),
            WeightRenaming("self_attn.out_proj", "self_attn.o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
        ],
        "d_fine": [
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.layers.0"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.layers.1"),
            WeightRenaming(r"encoder.encoder.(\d+).layers", r"encoder.aifi.\1.layers"),
        ],
        "nemotron_h": [
            WeightRenaming("backbone.", "model."),
            WeightRenaming("embedding.weight", "embeddings.weight"),
            WeightConverter(
                source_patterns=[
                    "mixer.experts.*.up_proj.weight",
                ],
                target_patterns="mixer.experts.up_proj",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                source_patterns=[
                    "mixer.experts.*.down_proj.weight",
                ],
                target_patterns="mixer.experts.down_proj",
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
            WeightRenaming(r"encoder.layers", r"layers"),
            WeightRenaming(r"emb_ln", r"embeddings.LayerNorm"),
            WeightRenaming(r"attn.out_proj", r"self_attn.o_proj"),
            WeightRenaming(r"fc11", r"up_proj"),
            WeightRenaming(r"fc12", r"gate_proj"),
            WeightRenaming(r"fc2", r"down_proj"),
            WeightRenaming(r"norm1", r"post_attention_layernorm"),
            WeightRenaming(
                r"norm2",
                r"post_mlp_layernorm",
            ),
            WeightConverter(
                source_patterns=["attn.Wqkv"],
                target_patterns=[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                operations=[Chunk(dim=0)],
            ),
        ],
        "jina_embeddings_v3": [
            WeightRenaming(source_patterns="emb_ln", target_patterns="embeddings.LayerNorm"),
            WeightRenaming(source_patterns="encoder.layers", target_patterns="layers"),
            WeightConverter(
                source_patterns="mixer.Wqkv",
                target_patterns=[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming(source_patterns="mixer.out_proj", target_patterns="self_attn.o_proj"),
            WeightRenaming(source_patterns="norm1", target_patterns="post_attention_layernorm"),
            WeightRenaming(source_patterns="norm2", target_patterns="post_mlp_layernorm"),
        ],
    }
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

    mapping["ernie4_5_moe"] = [
        WeightRenaming("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias"),
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
    ]
    mapping["minimax_m2"] = mapping["mixtral"].copy()
    mapping["minimax_m2"] += [
        WeightRenaming(".block_sparse_moe.e_score_correction_bias", ".mlp.e_score_correction_bias"),
    ]
    mapping["exaone_moe"] = mapping["qwen2_moe"].copy()
    mapping["exaone_moe"] += [WeightRenaming("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")]

    mapping["solar_open"] = [
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
    ]

    mapping["cohere_asr"] = [
        WeightRenaming(r"encoder\.pre_encode\.conv\.", r"encoder.subsampling.layers."),
        WeightRenaming(r"encoder\.pre_encode\.out\.", r"encoder.subsampling.linear."),
        WeightRenaming(r"transf_decoder\._embedding\.position_embedding\.pos_enc", r"decoder.pos_emb.weight"),
        WeightRenaming(r"transf_decoder\._embedding\.token_embedding", r"decoder.embed_tokens"),
        WeightRenaming(r"transf_decoder\._embedding\.layer_norm", r"decoder.embedding_layernorm"),
        WeightRenaming(r"transf_decoder\._decoder\.final_layer_norm", r"decoder.norm"),
        WeightRenaming(r"transf_decoder\._decoder\.layers", r"decoder.layers"),
        WeightRenaming(r"encoder_decoder_proj\.", r"decoder.proj."),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_q", r"encoder.(.+).self_attn.q_proj"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_k", r"encoder.(.+).self_attn.k_proj"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_v", r"encoder.(.+).self_attn.v_proj"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_out", r"encoder.(.+).self_attn.o_proj"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_pos", r"encoder.(.+).self_attn.relative_k_proj"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.pos_bias_u", r"encoder.(.+).self_attn.bias_u"),
        WeightRenaming(r"encoder\.(.+)\.self_attn\.pos_bias_v", r"encoder.(.+).self_attn.bias_v"),
        WeightRenaming(r"\.first_sub_layer\.query_net", r".self_attn.q_proj"),
        WeightRenaming(r"\.first_sub_layer\.key_net", r".self_attn.k_proj"),
        WeightRenaming(r"\.first_sub_layer\.value_net", r".self_attn.v_proj"),
        WeightRenaming(r"\.first_sub_layer\.out_projection", r".self_attn.o_proj"),
        WeightRenaming(r"\.second_sub_layer\.query_net", r".encoder_attn.q_proj"),
        WeightRenaming(r"\.second_sub_layer\.key_net", r".encoder_attn.k_proj"),
        WeightRenaming(r"\.second_sub_layer\.value_net", r".encoder_attn.v_proj"),
        WeightRenaming(r"\.second_sub_layer\.out_projection", r".encoder_attn.o_proj"),
        WeightRenaming(r"\.third_sub_layer\.dense_in", r".mlp.fc1"),
        WeightRenaming(r"\.third_sub_layer\.dense_out", r".mlp.fc2"),
        WeightRenaming(r"\.layer_norm_1\.", r".input_layernorm."),
        WeightRenaming(r"\.layer_norm_2\.", r".post_attention_layernorm."),
        WeightRenaming(r"\.layer_norm_3\.", r".final_layernorm."),
        WeightRenaming(r"\.conv\.batch_norm", r".conv.norm"),
        WeightRenaming(r"log_softmax\.mlp\.layer0", r"proj_out"),
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
    model_type: str,
    mapping: list[WeightConverter | WeightRenaming],
    overwrite: bool = False,
) -> None:
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    if model_type in _checkpoint_conversion_mapping_cache and not overwrite:
        raise ValueError(f"Model type {model_type} already exists in the checkpoint conversion mapping.")
    _checkpoint_conversion_mapping_cache[model_type] = mapping


def extract_weight_conversions_for_model(model: PreTrainedModel) -> list[WeightConverter | WeightRenaming] | None:
    model_type = getattr(model.config, "model_type", None)
    if model_type is not None:
        model_specific_conversions = get_checkpoint_conversion_mapping(model_type)
        return model_specific_conversions
    return None


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
    # Lazy import to avoid circular import issues
    from .modeling_utils import PreTrainedModel

    # note: this function is used in PEFT, so changing the API requires coordination
    weight_conversions = []

    # Load models with explicit, user-provided key mapping
    if key_mapping is not None:
        weight_conversions = [WeightRenaming(source_patterns=k, target_patterns=v) for k, v in key_mapping.items()]

    # Model have several `PreTrainedModel` within with the same model type
    # For ex: XForConditionalGeneration -> XModel. We don't want to apply the same
    # conversion pattern twice because of that
    seen_model_types = set()
    if (conversions := extract_weight_conversions_for_model(model)) is not None:
        weight_conversions.extend(conversions)
        seen_model_types.add(model.config.model_type)

    # Recurse over submodules and collect all conversions
    for submodule in model.modules():
        if (
            submodule is not model
            and isinstance(submodule, PreTrainedModel)
            and submodule.config.model_type not in seen_model_types
        ):
            conversions = extract_weight_conversions_for_model(submodule)
            if conversions is not None:
                weight_conversions.extend(conversions)
                seen_model_types.add(submodule.config.model_type)

    if add_legacy:
        weight_conversions.extend(get_checkpoint_conversion_mapping("legacy"))

    # Add the ones from the quantizer as well if provided
    if hf_quantizer is not None:
        weight_conversions.extend(hf_quantizer.get_weight_conversions())

    return weight_conversions
