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

from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

from .core_model_loading import (
    Chunk,
    Concatenate,
    ErnieFuseAndSplitTextVisionExperts,
    MergeModulelist,
    PrefixChange,
    Transpose,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
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
    "sam3_tracker_video": "sam3_tracker",
    "altclip_vision_model": "clip_vision_model",
    "chinese_clip_vision_model": "clip_vision_model",
    "clipseg_vision_model": "clip_vision_model",
    "metaclip_2_vision_model": "clip_vision_model",
    "mlcd_vision": "clip_vision_model",
    "mlcd": "clip_vision_model",
    "siglip_vision_model": "clip_vision_model",
    "siglip2_vision_model": "clip_vision_model",
    "xclip_vision_model": "clip_vision_model",
    "clipseg_text_model": "clip_text_model",
    "metaclip_2_text_model": "clip_text_model",
    "siglip_text_model": "clip_text_model",
    "siglip2_text_model": "clip_text_model",
    "xclip_text_model": "clip_text_model",
    "paligemma": "llava",
    "aya_vision": "llava",
    "got_ocr2": "llava",
    "gemma3": "llava",
    "internvl": "llava",
    "vipllava": "llava",
    "mistral3": "llava",
    "pp_chart2table": "llava",
    "llava_next_video": "llava_next",
    "llava_onevision": "llava_next",
    # class-based mappings
    "PaliGemmaModel": "LlavaModel",
    "AyaVisionModel": "LlavaModel",
    "GotOcr2Model": "LlavaModel",
    "Gemma3Model": "LlavaModel",
    "InternVLModel": "LlavaModel",
    "VipLlavaModel": "LlavaModel",
    "Mistral3Model": "LlavaModel",
    "PPChart2TableModel": "LlavaModel",
    "LlavaNextModel": "LlavaModel",
    "LlavaNextVideoModel": "LlavaModel",
    "LlavaOnevisionModel": "LlavaModel",
    "FuyuModel": "LlavaModel",
    "MllamaModel": "LlavaModel",
    "MaskFormerDetrDecoder": "DetrModel",
    "Qwen2_5_VLForConditionalGeneration": "Qwen2VLForConditionalGeneration",
}


def _build_checkpoint_conversion_mapping():
    mapping = {
        "altclip": [
            WeightRenaming(source_patterns=r"layer\.", target_patterns="layers."),
        ],
        "deepseek_v4": [
            # Upstream V4-Flash checkpoint uses a flatter V3-style namespace: `attn` /
            # `ffn` instead of `self_attn` / `mlp`, `attn_norm` / `ffn_norm`
            # instead of `input_layernorm` / `post_attention_layernorm`, `hc_attn_*`
            # / `hc_ffn_*` for the Hyper-Connection params (wrapped here as
            # `attn_hc` / `ffn_hc` submodules), `embed` / `head` / bare `norm`
            # for the model head, `hc_head_*` for the final HC collapse, and indexer
            # weights nested under `attn.indexer.compressor.*` upstream but flattened
            # onto the Indexer module here.
            #
            # All targets stay in the bare base-model namespace (no `model.` prefix).
            # `convert_and_load_state_dict_in_model` consults
            # :attr:`DeepseekV4PreTrainedModel.base_model_prefix = "model"` and adds /
            # strips the `model.` prefix automatically based on whether the loader
            # target is the base model or a head model.
            #
            # Ordering matters for save round-tripping: :func:`revert_weight_conversion`
            # reverses the order *and* each transform, so a structural prefix-only rule
            # placed before a specific in-prefix rename would steal the reverse match
            # and emit `layers.X.attn.sinks` instead of `layers.X.attn.attn_sink`.
            # We split into two passes: structural prefix renames first (so they apply
            # last on save / first on load), then specific in-prefix renames that
            # operate on the already-prefixed keys. FP8 `.scale` → `.weight_scale_inv`
            # rename lives in the FP8 quantizer's `update_weight_conversions` (only
            # active under FP8 dequant), so the V4 static mapping below stays free of
            # FP8-only rules.
            # ---- Pass 1: top-level + structural prefix renames ----
            WeightRenaming(source_patterns=r"^embed\.weight$", target_patterns="embed_tokens.weight"),
            WeightRenaming(source_patterns=r"^head\.weight$", target_patterns="lm_head.weight"),
            WeightRenaming(source_patterns=r"^norm\.weight$", target_patterns="norm.weight"),
            WeightRenaming(source_patterns=r"^hc_head_fn$", target_patterns="hc_head.hc_fn"),
            WeightRenaming(source_patterns=r"^hc_head_base$", target_patterns="hc_head.hc_base"),
            WeightRenaming(source_patterns=r"^hc_head_scale$", target_patterns="hc_head.hc_scale"),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.attn_norm\.",
                target_patterns=r"layers.\1.input_layernorm.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.ffn_norm\.",
                target_patterns=r"layers.\1.post_attention_layernorm.",
            ),
            WeightRenaming(source_patterns=r"^layers\.(\d+)\.hc_attn_fn$", target_patterns=r"layers.\1.attn_hc.fn"),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.hc_attn_base$", target_patterns=r"layers.\1.attn_hc.base"
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.hc_attn_scale$", target_patterns=r"layers.\1.attn_hc.scale"
            ),
            WeightRenaming(source_patterns=r"^layers\.(\d+)\.hc_ffn_fn$", target_patterns=r"layers.\1.ffn_hc.fn"),
            WeightRenaming(source_patterns=r"^layers\.(\d+)\.hc_ffn_base$", target_patterns=r"layers.\1.ffn_hc.base"),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.hc_ffn_scale$", target_patterns=r"layers.\1.ffn_hc.scale"
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.attn\.",
                target_patterns=r"layers.\1.self_attn.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.ffn\.",
                target_patterns=r"layers.\1.mlp.",
            ),
            # ---- Pass 2: in-prefix specific renames (operate on already-prefixed keys) ----
            # These can safely run after the structural prefix renames because their
            # source patterns include the `layers.X.self_attn.` / `layers.X.mlp.`
            # prefix. On reverse the order flips so these undo first, restoring the
            # specific upstream names *before* the structural rules strip the prefix.
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.attn_sink$",
                target_patterns=r"layers.\1.self_attn.sinks",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.norm\.",
                target_patterns=r"layers.\1.self_attn.compressor.indexer.kv_norm.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.ape$",
                target_patterns=r"layers.\1.self_attn.compressor.indexer.position_bias",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.",
                target_patterns=r"layers.\1.self_attn.compressor.indexer.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.indexer\.",
                target_patterns=r"layers.\1.self_attn.compressor.indexer.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.norm\.",
                target_patterns=r"layers.\1.self_attn.compressor.kv_norm.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.ape$",
                target_patterns=r"layers.\1.self_attn.compressor.position_bias",
            ),
            # Attention / compressor / indexer leaf weights: upstream uses paper notation
            # (`wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b` / `wgate`); we
            # rename to the standard transformers `*_proj` form. Compressor / Indexer
            # `wkv` / `wgate` are caught by the same patterns since they sit under
            # `self_attn.` after the Pass 1 prefix rewrite.
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wq_a\.",
                target_patterns=r"layers.\1.self_attn.\2.q_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wq_b\.",
                target_patterns=r"layers.\1.self_attn.\2.q_b_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wkv\.",
                target_patterns=r"layers.\1.self_attn.\2.kv_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wgate\.",
                target_patterns=r"layers.\1.self_attn.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_a\.",
                target_patterns=r"layers.\1.self_attn.\2.o_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_b\.",
                target_patterns=r"layers.\1.self_attn.\2.o_b_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.wq_a\.",
                target_patterns=r"layers.\1.self_attn.q_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.wq_b\.",
                target_patterns=r"layers.\1.self_attn.q_b_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.wkv\.",
                target_patterns=r"layers.\1.self_attn.kv_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.wo_a\.",
                target_patterns=r"layers.\1.self_attn.o_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.wo_b\.",
                target_patterns=r"layers.\1.self_attn.o_b_proj.",
            ),
            # Norm rename: upstream ships `q_norm` (the LoRA-rank RMSNorm sitting between
            # q_a_proj and q_b_proj); we register it as `q_a_norm` so the suffix matches
            # the surrounding `q_a_proj` / `q_b_proj` / `q_b_norm` symmetry. The
            # unweighted `q_b_norm` has no learnable weight, so no upstream key.
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.self_attn\.q_norm\.",
                target_patterns=r"layers.\1.self_attn.q_a_norm.",
            ),
            # Aux-loss-free routing bias: upstream ships `gate.bias` (V3 convention);
            # we register it as `e_score_correction_bias` (cross-model standard name).
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.gate\.bias$",
                target_patterns=r"layers.\1.mlp.gate.e_score_correction_bias",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.shared_experts\.w1\.",
                target_patterns=r"layers.\1.mlp.shared_experts.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.shared_experts\.w2\.",
                target_patterns=r"layers.\1.mlp.shared_experts.down_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.shared_experts\.w3\.",
                target_patterns=r"layers.\1.mlp.shared_experts.up_proj.",
            ),
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
        "LlavaModel": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="language_model"),
        ],
        "llava": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^vision_tower", target_patterns="model.vision_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
        ],
        "llava_next": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^vision_tower", target_patterns="model.vision_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
            WeightRenaming(source_patterns=r"^image_newline", target_patterns="model.image_newline"),
        ],
        "clip_vision_model": [PrefixChange(prefix_to_remove="vision_model")],
        "clip_text_model": [PrefixChange(prefix_to_remove="text_model")],
        "VideoLlavaModel": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="language_model"),
        ],
        "video_llava": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^image_tower", target_patterns="model.image_tower"),
            WeightRenaming(source_patterns=r"^video_tower", target_patterns="model.video_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
        ],
        "fuyu": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^vision_embed_tokens", target_patterns="model.vision_embed_tokens"),
        ],
        "mllama": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^vision_model", target_patterns="model.vision_model"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
        ],
        "Emu3Model": [
            WeightRenaming(source_patterns=r"^text_model.model", target_patterns="text_model"),
        ],
        "emu3": [
            WeightRenaming(source_patterns=r"^text_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^text_model", target_patterns="model.text_model"),
            WeightRenaming(source_patterns=r"^vqmodel", target_patterns="model.vqmodel"),
        ],
        "paddleocr_vl": [
            WeightRenaming(source_patterns=r"^mlp_AR", target_patterns="model.projector"),
            WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
            WeightRenaming(
                source_patterns=r"^model(?!(\.visual|\.projector|\.language_model))",
                target_patterns="model.language_model",
            ),
        ],
        "Qwen2VLForConditionalGeneration": [
            WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
            WeightRenaming(
                source_patterns=r"^model(?!\.(language_model|visual))", target_patterns="model.language_model"
            ),
        ],
        "colqwen2": [PrefixChange(prefix_to_remove="model", model_prefix="vlm")],
        "shieldgemma2": [PrefixChange(prefix_to_add="model", model_prefix="model")],
        "timm_wrapper": [PrefixChange(prefix_to_add="timm_model")],
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
        "esm": [
            WeightRenaming(
                "encoder.layer.*.attention.self.rotary_embeddings.inv_freq",
                "rotary_embeddings.inv_freq",
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
        "qwen3_5_text": [PrefixChange(prefix_to_remove="language_model", model_prefix="model")],
        "sam3_tracker": [
            WeightRenaming(
                source_patterns=r"detector_model.vision_encoder.backbone.", target_patterns="vision_encoder.backbone."
            ),
            WeightRenaming(source_patterns=r"tracker_neck.", target_patterns="vision_encoder.neck."),
            # the regex allows to remove the prefix, and add it back in revert mode
            WeightRenaming(source_patterns=r"tracker_model.(.+)", target_patterns=r"\1"),
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
        "DetrModel": [
            WeightRenaming("backbone.conv_encoder", "backbone"),
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
        ],
        "rt_detr": [
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.fc1"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.fc2"),
            WeightRenaming(r"encoder.encoder.(\d+).layers", r"encoder.aifi.\1.layers"),
        ],
        "RfDetrModel": [
            # RfDetrConvEncoder — backbone checkpoint layout + projector stages
            WeightRenaming(r"backbone.0.encoder.encoder", r"backbone.backbone"),
            WeightRenaming(r"backbone.0.projector", r"backbone.projector"),
            WeightRenaming(r"projector.stages.0.0.cv1.conv", r"projector.projector_layer.conv1.conv"),
            WeightRenaming(r"projector.stages.0.0.cv1.bn", r"projector.projector_layer.conv1.norm"),
            WeightRenaming(r"projector.stages.0.0.cv2.conv", r"projector.projector_layer.conv2.conv"),
            WeightRenaming(r"projector.stages.0.0.cv2.bn", r"projector.projector_layer.conv2.norm"),
            WeightRenaming(r"projector.stages.0.1", r"projector.layer_norm"),
            WeightRenaming(
                r"projector.stages.0.0.m.(\d+).cv1.conv", r"projector.projector_layer.bottlenecks.\1.conv1.conv"
            ),
            WeightRenaming(
                r"projector.stages.0.0.m.(\d+).cv1.bn", r"projector.projector_layer.bottlenecks.\1.conv1.norm"
            ),
            WeightRenaming(
                r"projector.stages.0.0.m.(\d+).cv2.conv", r"projector.projector_layer.bottlenecks.\1.conv2.conv"
            ),
            WeightRenaming(
                r"projector.stages.0.0.m.(\d+).cv2.bn", r"projector.projector_layer.bottlenecks.\1.conv2.norm"
            ),
            # RfDetrDecoder
            WeightRenaming(r"transformer.decoder", r"decoder"),
            WeightRenaming(r"decoder.layers.(\d+).norm1", r"decoder.layers.\1.self_attn_layer_norm"),
            WeightRenaming(r"decoder.layers.(\d+).norm2", r"decoder.layers.\1.cross_attn_layer_norm"),
            WeightRenaming(r"decoder.layers.(\d+).linear1", r"decoder.layers.\1.mlp.fc1"),
            WeightRenaming(r"decoder.layers.(\d+).linear2", r"decoder.layers.\1.mlp.fc2"),
            WeightRenaming(r"decoder.layers.(\d+).norm3", r"decoder.layers.\1.layer_norm"),
            WeightRenaming(r"decoder.norm", r"decoder.layernorm"),
            WeightRenaming(r"^transformer\.enc_output_norm", r"enc_output_norm"),
            WeightRenaming(r"^transformer\.enc_output", r"enc_output"),
            WeightRenaming(r"transformer.enc_out_class_embed", r"enc_out_class_embed"),
            WeightRenaming(r"transformer.enc_out_bbox_embed", r"enc_out_bbox_embed"),
            WeightRenaming(r"refpoint_embed\.weight", r"reference_point_embed.weight"),
            # RfDetrAttention
            WeightRenaming(r"self_attn.out_proj", r"self_attn.o_proj"),
            WeightConverter(
                source_patterns=r"self_attn.in_proj_bias",
                target_patterns=[r"self_attn.q_proj.bias", r"self_attn.k_proj.bias", r"self_attn.v_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"self_attn.in_proj_weight",
                target_patterns=[r"self_attn.q_proj.weight", r"self_attn.k_proj.weight", r"self_attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ],
        "RfDetrForObjectDetection": [
            WeightRenaming(source_patterns="^", target_patterns="model."),
        ],
        "RfDetrForInstanceSegmentation": [
            WeightRenaming(source_patterns="^(?!segmentation_head)", target_patterns="model."),
            WeightRenaming(r"segmentation_head\.query_features_block\.layers\.0", r"query_features_block.mlp.fc1"),
            WeightRenaming(r"segmentation_head\.query_features_block\.layers\.2", r"query_features_block.mlp.fc2"),
            WeightRenaming(r"segmentation_head\.query_features_block\.norm_in", r"query_features_block.norm"),
            WeightRenaming(r"segmentation_head\.blocks\.(\d+)\.norm", r"blocks.\1.layernorm"),
            WeightRenaming(r"segmentation_head\.blocks\.(\d+)\.dwconv", r"blocks.\1.depthwise_conv"),
            WeightRenaming(r"segmentation_head\.blocks\.(\d+)\.pwconv1", r"blocks.\1.pointwise_conv"),
            WeightRenaming(r"segmentation_head\.spatial_features_proj", r"spatial_features_proj"),
            WeightRenaming(r"segmentation_head\.query_features_proj", r"query_features_proj"),
            WeightRenaming(r"segmentation_head\.bias", r"segmentation_bias"),
        ],
        "ConditionalDetrModel": [
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
            WeightRenaming(source_patterns="mixer.out_proj", target_patterns="self_attn.o_proj"),
            WeightRenaming(source_patterns="norm1", target_patterns="post_attention_layernorm"),
            WeightRenaming(source_patterns="norm2", target_patterns="post_mlp_layernorm"),
            WeightConverter(
                source_patterns="mixer.Wqkv",
                target_patterns=[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                operations=[Chunk(dim=0)],
            ),
        ],
        "cohere_asr": [
            WeightRenaming(r"encoder\.pre_encode\.conv\.", r"encoder.subsampling.layers."),
            WeightRenaming(r"encoder\.pre_encode\.out\.", r"encoder.subsampling.linear."),
            WeightRenaming(r"transf_decoder\._embedding\.position_embedding\.pos_enc", r"decoder.pos_emb.weight"),
            WeightRenaming(r"transf_decoder\._embedding\.token_embedding", r"decoder.embed_tokens"),
            WeightRenaming(r"transf_decoder\._embedding\.layer_norm", r"decoder.embedding_layernorm"),
            WeightRenaming(r"transf_decoder\._decoder\.final_layer_norm", r"decoder.norm"),
            WeightRenaming(r"transf_decoder\._decoder\.layers", r"decoder.layers"),
            WeightRenaming(r"encoder_decoder_proj\.", r"decoder.proj."),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_q", r"encoder.\1.self_attn.q_proj"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_k", r"encoder.\1.self_attn.k_proj"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_v", r"encoder.\1.self_attn.v_proj"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_out", r"encoder.\1.self_attn.o_proj"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.linear_pos", r"encoder.\1.self_attn.relative_k_proj"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.pos_bias_u", r"encoder.\1.self_attn.bias_u"),
            WeightRenaming(r"encoder\.(.+)\.self_attn\.pos_bias_v", r"encoder.\1.self_attn.bias_v"),
            WeightRenaming(r"decoder\.(.+)\.first_sub_layer\.query_net", r"decoder.\1.self_attn.q_proj"),
            WeightRenaming(r"decoder\.(.+)\.first_sub_layer\.key_net", r"decoder.\1.self_attn.k_proj"),
            WeightRenaming(r"decoder\.(.+)\.first_sub_layer\.value_net", r"decoder.\1.self_attn.v_proj"),
            WeightRenaming(r"decoder\.(.+)\.first_sub_layer\.out_projection", r"decoder.\1.self_attn.o_proj"),
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
        ],
        "qianfan_ocr": [
            WeightRenaming(r"^vision_model\.", r"model\.vision_tower\."),
            WeightRenaming(r"encoder\.layers\.", r"layers\."),
            WeightRenaming(r"\.ls1", r"\.lambda_1"),
            WeightRenaming(r"\.ls2", r"\.lambda_2"),
            WeightRenaming(r"(layers\.\d+)\.attn\.proj\.", r"\1.attention.projection_layer."),
            WeightRenaming(r"\.norm1\.", r"\.layernorm_before\."),
            WeightRenaming(r"\.norm2\.", r"\.layernorm_after\."),
            WeightRenaming(r"\.embeddings\.class_embedding", r"\.embeddings\.cls_token"),
            WeightRenaming(r"\.embeddings\.position_embedding", r"\.embeddings\.position_embeddings"),
            WeightRenaming(r"\.embeddings\.patch_embedding\.", r"\.embeddings\.patch_embeddings\.projection\."),
            WeightRenaming(r"^language_model\.model\.", r"model\.language_model\."),
            WeightRenaming(r"^language_model\.lm_head\.", r"lm_head\."),
            WeightRenaming(r"^mlp1\.0\.", r"model\.multi_modal_projector\.layer_norm\."),
            WeightRenaming(r"^mlp1\.1\.", r"model\.multi_modal_projector\.linear_1\."),
            WeightRenaming(r"^mlp1\.3\.", r"model\.multi_modal_projector\.linear_2\."),
            WeightConverter(
                source_patterns=["attn.qkv.weight"],
                target_patterns=["attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=["attn.qkv.bias"],
                target_patterns=["attention.q_proj.bias", "attention.k_proj.bias", "attention.v_proj.bias"],
                operations=[Chunk(dim=0)],
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
    }
    # The legacy mapping is added to the esm model here since the extra weight renaming do not apply to the esm model.
    mapping["esm"] += mapping["legacy"].copy()

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
    # Base DetrModel/ConditionalDetrModel transforms are picked up automatically as
    # scoped sub-module transforms; only the segmentation-specific patterns are needed here.
    mapping["DetrForSegmentation"] = [
        WeightRenaming("bbox_attention.q_linear", "bbox_attention.q_proj"),
        WeightRenaming("bbox_attention.k_linear", "bbox_attention.k_proj"),
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
    ]
    mapping["ConditionalDetrForSegmentation"] = mapping["DetrForSegmentation"].copy()

    mapping["ernie4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] += [
        WeightRenaming("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias")
    ]

    mapping["minimax_m2"] = mapping["mixtral"].copy()
    mapping["minimax_m2"] += [
        WeightRenaming(".block_sparse_moe.e_score_correction_bias", ".mlp.e_score_correction_bias"),
    ]
    mapping["exaone_moe"] = mapping["qwen2_moe"].copy()
    mapping["exaone_moe"] += [WeightRenaming("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")]

    # HYV3: qwen2_moe expert fusion + attribute renames for MiniMaxM2-style inheritance
    mapping["hy_v3"] = mapping["qwen2_moe"].copy()
    mapping["hy_v3"] += [
        WeightRenaming(source_patterns=r"mlp\.router\.gate\.weight", target_patterns="mlp.gate.weight"),
        WeightRenaming(source_patterns=r"mlp\.expert_bias", target_patterns="mlp.e_score_correction_bias"),
        WeightRenaming(source_patterns=r"mlp\.shared_mlp\.", target_patterns="mlp.shared_experts."),
    ]
    mapping["qwen3_5_moe_text"] = mapping["qwen3_5_text"].copy()
    mapping["qwen3_5_moe_text"] += mapping["qwen2_moe"].copy()

    mapping["laguna"] = mapping["qwen2_moe"].copy()
    mapping["laguna"] += [
        WeightRenaming("mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
        WeightRenaming("mlp.shared_expert.", "mlp.shared_experts."),
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
    model_type_or_class_name: str,
    mapping: list[WeightConverter | WeightRenaming],
    overwrite: bool = False,
) -> None:
    """
    Register a conversion mapping for a model type string or a class name.

    Class names take priority over `model_type` strings during lookup (see
    `extract_weight_conversions_for_model`), making it possible to define
    task-head-specific or class-specific conversions that differ from the shared
    `model_type` baseline.
    """
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    if model_type_or_class_name in _checkpoint_conversion_mapping_cache and not overwrite:
        raise ValueError(
            f"Conversion mapping for '{model_type_or_class_name}' already exists. Pass overwrite=True to replace it."
        )
    _checkpoint_conversion_mapping_cache[model_type_or_class_name] = mapping


def extract_weight_conversions_for_model(
    model: PreTrainedModel,
) -> list[WeightTransform] | None:
    """
    Return the registered conversion list for `model`, or `None` if none exists.

    Looks up by class name first (enables task-head-specific overrides), then
    falls back to `model.config.model_type`.  Transforms are returned
    unmodified; the caller sets `scope_prefix` on each transform for sub-module isolation.
    """
    class_name = type(model).__name__
    model_type = model.config.model_type

    # Class name takes priority — allows ForXxx-specific overrides
    conversions = get_checkpoint_conversion_mapping(class_name)
    if conversions is None and model_type:
        conversions = get_checkpoint_conversion_mapping(model_type)
    return conversions


def get_model_conversion_mapping(
    model: PreTrainedModel,
    key_mapping: dict[str, str] | None = None,
    hf_quantizer: HfQuantizer | None = None,
    add_legacy: bool = True,
) -> list[WeightTransform]:
    """
    Collect the ordered list of weight transforms for `model` (used during
    loading and, when reversed, during saving).

    Each `PreTrainedModel` sub-module is looked up by class name then
    `model_type`.  Root transforms are applied globally; sub-module transforms
    have their `scope_prefix` set so they only match keys under that prefix.  After any
    sub-module is processed, both its class name and `model_type` are marked
    seen to prevent `XForY` / `XModel` pairs from applying the same mapping
    twice via different lookup paths.
    """
    # note: this function is used in PEFT, so changing the API requires coordination
    weight_conversions = []

    # Load models with explicit, user-provided key mapping
    if key_mapping is not None:
        weight_conversions = [WeightRenaming(source_patterns=k, target_patterns=v) for k, v in key_mapping.items()]

    # Maps each identifier (class name or model_type) to the module paths that have
    # already claimed it.  A later module is skipped only when one of those paths is
    # an ancestor of the current module path — siblings are never ancestors of each
    # other, so two sibling sub-models with the same model_type both get their own
    # scoped transforms.  A child is an ancestor of everything nested under it, which
    # prevents a parent's transforms from being duplicated with a scoped copy for the child.
    seen_identifiers: defaultdict[str, list[str]] = defaultdict(list)

    named_pretrained = model._named_pretrained_submodules
    for module_name, submodule in named_pretrained:
        class_name = type(submodule).__name__
        model_type = submodule.config.model_type

        # Skip if an ancestor already claimed this class (its unscoped transforms already cover this subtree).
        if any(seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[class_name]):
            continue

        # Class name takes priority — a class-specific mapping bypasses the model_type
        # deduplication check (e.g. LlavaModel nested inside LlavaForConditionalGeneration
        # must still get its own scoped mapping even after "llava" is marked seen).
        conversions = get_checkpoint_conversion_mapping(class_name)
        found_via_class = conversions is not None

        if not found_via_class:
            # Same ancestor check as above, but via model_type for modules without a class-specific mapping.
            if model_type and any(
                seen == "" or module_name.startswith(seen + ".") for seen in seen_identifiers[model_type]
            ):
                continue
            if model_type is not None:
                conversions = get_checkpoint_conversion_mapping(model_type)

        if conversions is None:
            continue

        is_root_model = module_name == ""
        if not is_root_model:
            # Scope each transform so it only matches keys under this sub-module's prefix.
            for transform in conversions:
                transform.scope_prefix = module_name
        weight_conversions.extend(conversions)

        seen_identifiers[class_name].append(module_name)
        # Only record model_type when the hit was via model_type. When the hit was via
        # class name, other sub-modules sharing the same model_type but without a
        # class-specific mapping (e.g. DetrModel under DetrForSegmentation) must still
        # be reachable so their base transforms are picked up and scoped.
        if not found_via_class and model_type:
            seen_identifiers[model_type].append(module_name)

    if add_legacy:
        weight_conversions.extend(get_checkpoint_conversion_mapping("legacy"))

    # Let the quantizer rewrite / augment the conversion pipeline. This is where the
    # FP8 dequantizer (when `dequantize=True`) prepends a `Fp8Dequantize` op to
    # every existing converter so that per-block scales are applied *before* any
    # expert-merge / concat ops flatten the per-expert structure away.
    if hf_quantizer is not None:
        weight_conversions = hf_quantizer.update_weight_conversions(weight_conversions)

    return weight_conversions
