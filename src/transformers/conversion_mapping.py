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
    GroupWeightRename,
    Interleave,
    MergeModulelist,
    PermuteForRope,
    PrefixChange,
    Split,
    SwapHalves,
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
    "deepseek_v32": "qwen2_moe",
    "dots1": "qwen2_moe",
    "ernie4_5_moe": "qwen2_moe",
    "glm4_moe": "qwen2_moe",
    "glm4_moe_lite": "qwen2_moe",
    "glm_moe_dsa": "qwen2_moe",
    "glm4v_moe": "qwen2_moe",
    "longcat_flash": "qwen2_moe",
    "solar_open": "qwen2_moe",
    "mellum": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "beam": "qwen2_moe",
    "qwen3_omni_moe": "qwen2_moe",
    "qwen3_omni_moe_thinker": "qwen2_moe",
    "qwen3_omni_moe_text": "qwen2_moe",
    "qwen3_next": "qwen2_moe",
    "flex_olmo": "qwen2_moe",
    "olmoe": "qwen2_moe",
    "exaone_moe": "qwen2_moe",
    "cohere2_moe": "qwen2_moe",
    "rt_detr_v2": "rt_detr",
    "pp_doclayout_v2": "rt_detr",
    "pp_doclayout_v3": "rt_detr",
    "sam3_tracker_video": "sam3_tracker",
    "qwen3_vl_moe_vision": "qwen3_vl_vision",
    "qwen3_5_vision": "qwen3_vl_vision",
    "qwen3_5_moe_vision": "qwen3_vl_vision",
    "qwen3_omni_moe_vision_encoder": "qwen3_vl_vision",
    "qwen4_exp_vision": "qwen3_vl_vision",
    "cohere_compass_vision": "qwen3_vl_vision",
    "vibevoice_acoustic_tokenizer_encoder": "vibevoice_acoustic_tokenizer",
    "vibevoice_acoustic_tokenizer_decoder": "vibevoice_acoustic_tokenizer",
    "AltCLIPVisionModel": "CLIPVisionModel",
    "ChineseCLIPVisionModel": "CLIPVisionModel",
    "CLIPSegVisionModel": "CLIPVisionModel",
    "MetaClip2VisionModel": "CLIPVisionModel",
    "mlcd_vision": "CLIPVisionModel",
    "mlcd": "CLIPVisionModel",
    "SiglipVisionModel": "CLIPVisionModel",
    "Siglip2VisionModel": "CLIPVisionModel",
    "xCLIPVisionModel": "CLIPVisionModel",
    "CLIPSegTextModel": "CLIPTextModel",
    "MetaClip2TextModel": "CLIPTextModel",
    "SiglipTextModel": "CLIPTextModel",
    "Siglip2TextModel": "CLIPTextModel",
    "xCLIPTextModel": "CLIPTextModel",
    "paligemma": "llava",
    "aya_vision": "llava",
    "got_ocr2": "llava",
    "gemma3": "llava",
    "internvl": "llava",
    "vipllava": "llava",
    "mistral3": "llava",
    "pp_chart2table": "llava",
    "voxtral": "qwen2_audio",
    "voxtral_realtime": "qwen2_audio",
    "audioflamingo3": "qwen2_audio",
    "glmasr": "qwen2_audio",
    "musicflamingo": "qwen2_audio",
    "granite_speech_plus": "granite_speech",
    "gemma3n_text": "qwen3_5_text",
    "glm5_next_text": "glm5_next",
    "qwen3_5_moe_text": "qwen3_5_text",
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
    "VoxtralModel": "Qwen2AudioModel",
    "VoxtralRealtimeModel": "Qwen2AudioModel",
    "AudioFlamingo3Model": "Qwen2AudioModel",
    "GlmAsrModel": "Qwen2AudioModel",
    "MusicFlamingoModel": "Qwen2AudioModel",
    "GraniteSpeechPlusModel": "GraniteSpeechModel",
    "MaskFormerDetrDecoder": "DetrModel",
    "Qwen2_5_VLModel": "Qwen2VLModel",
    "Qwen2_5_VLForConditionalGeneration": "Qwen2VLForConditionalGeneration",
    "qwen2_5_omni_thinker": "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen3OmniMoeThinkerModel": "Qwen2_5OmniThinkerModel",
    "Qwen3OmniMoeThinkerForConditionalGeneration": "Qwen2_5OmniThinkerForConditionalGeneration",
    "Tipsv2VisionBackbone": "Tipsv2VisionModel",
    "UnlimitedOcrTextModel": "qwen2_moe",
    # ViT-style vision models (old HuggingFace checkpoint format → new modular format)
    "ASTModel": "ViTModel",
    "BeitModel": "ViTModel",
    "DeiTModel": "ViTModel",
    "IJepaModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
    "ViTMSNModel": "ViTModel",
    "VivitModel": "ViTModel",
    "Dinov2Backbone": "Dinov2Model",
    "Dinov2WithRegistersBackbone": "Dinov2WithRegistersModel",
}


# not preceded by `layer.<N>.` (singular, as in the Dinov2 backbone of RF-DETR; its DETR decoder uses `layers.<N>.`)
_NOT_BACKBONE_LAYER = r"(?<!r\.\d\.)(?<!r\.\d\d\.)"


def _build_checkpoint_conversion_mapping():
    vit_layer_renames = [
        WeightRenaming(r"(layers?\.\d+)\.attention\.", r"\1.self_attn."),
        WeightRenaming(r"(layers?\.\d+)\.layernorm_before\.", r"\1.input_layernorm."),
        WeightRenaming(r"(layers?\.\d+)\.layernorm_after\.", r"\1.post_attention_layernorm."),
    ]
    dino_layer_renames = [
        WeightRenaming(r"(layers?\.\d+)\.attention\.", r"\1.self_attn."),
        WeightRenaming(r"(layers?\.\d+)\.norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"(layers?\.\d+)\.norm2\.", r"\1.post_attention_layernorm."),
    ]
    clip_layer_renames = [
        WeightRenaming(r"((?:layers?|svtr_block)\.\d+)\.layer_norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"((?:layers?|svtr_block)\.\d+)\.layer_norm2\.", r"\1.post_attention_layernorm."),
    ]
    granitemoe_block_renames = [
        WeightRenaming(r"(layers\.\d+)\.block_sparse_moe\.router\.", r"\1.mlp.gate."),
        WeightRenaming(r"(layers\.\d+)\.block_sparse_moe\.experts\.", r"\1.mlp.experts."),
    ]
    mapping = {
        "hy_v4": [
            # General HC prefix which is dropped
            WeightRenaming(r"\.hc_pre\.hc_", ".hc_"),
            # Attn gating + sinks
            WeightRenaming(r"\.learnable_sink_param", ".sinks"),
            WeightRenaming(r"\.linear_gate", ".gate_proj"),
            # Follow DSv4 HC standards
            WeightRenaming(r"\.hc_attn_layer\.hc_fn", ".attn_hc.fn"),
            WeightRenaming(r"\.hc_attn_layer\.hc_base", ".attn_hc.base"),
            WeightRenaming(r"\.hc_attn_layer\.hc_scale", ".attn_hc.scale"),
            WeightRenaming(r"\.hc_mlp_layer\.hc_fn", ".ffn_hc.fn"),
            WeightRenaming(r"\.hc_mlp_layer\.hc_base", ".ffn_hc.base"),
            WeightRenaming(r"\.hc_mlp_layer\.hc_scale", ".ffn_hc.scale"),
            WeightRenaming(r"\.hc_head_fn", ".hc_fn"),
            WeightRenaming(r"\.hc_head_base", ".hc_base"),
            WeightRenaming(r"\.hc_head_scale", ".hc_scale"),
        ],
        # Cosmos3 Edge's composite checkpoint stores its dense reasoner text tower as conventional attention + MLP
        # blocks. The visual/projector tensors already use their native module names and intentionally need no mapping.
        "cosmos3_edge": [
            WeightRenaming(r"^embed_tokens\.", "model.language_model.embed_tokens."),
            WeightRenaming(r"^norm\.", "model.language_model.norm."),
            WeightRenaming(r"^layers\.", "model.language_model.layers."),
            WeightRenaming(r"\.self_attn\.to_q\.", ".self_attn.q_proj."),
            WeightRenaming(r"\.self_attn\.to_k\.", ".self_attn.k_proj."),
            WeightRenaming(r"\.self_attn\.to_v\.", ".self_attn.v_proj."),
            WeightRenaming(r"\.self_attn\.to_out\.", ".self_attn.o_proj."),
            WeightRenaming(r"\.mlp\.up_proj\.", ".mlp.fc1."),
            WeightRenaming(r"\.mlp\.down_proj\.", ".mlp.fc2."),
        ],
        "inkling_mm_model": [
            WeightRenaming(source_patterns=r"model\.llm\.layers", target_patterns=r"model.language_model.layers"),
            WeightRenaming(
                source_patterns=r"model\.llm\.embed_norm\.weight",
                target_patterns=r"model.language_model.embed_tokens.embed_norm.weight",
            ),
            WeightRenaming(
                source_patterns=r"model\.llm\.embed\.weight",
                target_patterns=r"model.language_model.embed_tokens.weight",
            ),
            WeightRenaming(
                source_patterns=r"model\.llm\.norm\.weight", target_patterns=r"model.language_model.norm.weight"
            ),
            WeightRenaming(source_patterns=r"model\.llm\.unembed\.weight", target_patterns=r"lm_head.weight"),
            WeightRenaming(source_patterns=r"model\.audio\.", target_patterns=r"model.audio_tower."),
            WeightRenaming(source_patterns=r"model\.visual", target_patterns=r"model.vision_tower"),
            # Vision encoder internals (run after the tower namespace renames)
            WeightRenaming(
                source_patterns=r"vision_tower.layers.linear_(\d+)",
                target_patterns=r"vision_tower.encoder_layers.\1.projection",
            ),
            WeightRenaming(
                source_patterns=r"vision_tower.layers.norm_(\d+)",
                target_patterns=r"vision_tower.encoder_layers.\1.layer_norm",
            ),
            # Audio tower internals. These run AFTER the generic `model.audio.` -> `model.audio_tower.`
            # rename above and substring-match its output. The audio dMel embedding moved from
            # `audio.encoder` to an `InklingAudioModelEmbeddings` submodule, and `final_norm` was
            # renamed to `norm`.
            WeightRenaming(
                source_patterns=r"audio_tower.encoder.weight",
                target_patterns=r"audio_tower.embed_audio_tokens.embed_audio_tokens.weight",
            ),
            WeightRenaming(
                source_patterns=r"audio_tower.final_norm.weight", target_patterns=r"audio_tower.norm.weight"
            ),
            # MoE and MLP
            # no Transpose ops here: the TP loader shards the raw tensor but validates the shard
            # shape on the target param, so dim-permuting conversions break sharded loads cc @cyril
            WeightConverter(
                source_patterns="shared_w13_weight",
                target_patterns=["gate_proj", "up_proj"],
                operations=[Interleave(dim=1), Chunk(dim=1)],
            ),
            WeightRenaming(source_patterns=r"shared_w2_weight", target_patterns=r"down_proj"),
            WeightConverter(
                source_patterns="mlp.experts.w13_weight",
                target_patterns=["mlp.experts.gate_up_proj"],
                operations=[Interleave(dim=1)],
            ),
            WeightRenaming(source_patterns=r"mlp.experts.w2_weight", target_patterns=r"mlp.experts.down_proj"),
            WeightConverter(
                source_patterns="mlp.w13_dn.weight",
                target_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                operations=[Interleave(dim=0), Chunk(dim=0)],
            ),
            WeightRenaming(source_patterns=r"mlp.w2_md.weight", target_patterns=r"mlp.down_proj.weight"),
            WeightRenaming(source_patterns=r"mlp.gate.bias", target_patterns=r"mlp.gate.e_score_correction_bias"),
            # Attn
            WeightRenaming(source_patterns=r"attn\.wq_du", target_patterns=r"self_attn.q_proj"),
            WeightRenaming(source_patterns=r"attn\.wk_dv", target_patterns=r"self_attn.k_proj"),
            WeightRenaming(source_patterns=r"attn\.wv_dv", target_patterns=r"self_attn.v_proj"),
            WeightRenaming(source_patterns=r"attn\.wr_du", target_patterns=r"self_attn.r_proj"),
            WeightRenaming(source_patterns=r"attn\.wo_ud", target_patterns=r"self_attn.o_proj"),
            WeightRenaming(source_patterns=r"\.attn\.q_norm", target_patterns=r".self_attn.q_norm"),
            WeightRenaming(source_patterns=r"\.attn\.k_norm", target_patterns=r".self_attn.k_norm"),
            WeightRenaming(source_patterns=r"\.attn\.k_sconv", target_patterns=r".self_attn.k_sconv.conv1d"),
            WeightRenaming(source_patterns=r"\.attn\.v_sconv", target_patterns=r".self_attn.v_sconv.conv1d"),
            WeightRenaming(source_patterns=r"\.attn\.rel_logits_proj", target_patterns=r".self_attn.rel_logits_proj"),
            WeightRenaming(source_patterns=r"attn_sconv\.weight", target_patterns=r"attn_sconv.conv1d.weight"),
            WeightRenaming(source_patterns=r"mlp_sconv\.weight", target_patterns=r"mlp_sconv.conv1d.weight"),
            WeightRenaming(source_patterns=r"mlp_norm", target_patterns=r"post_attention_layernorm"),
            WeightRenaming(source_patterns=r"attn_norm", target_patterns=r"input_layernorm"),
        ],
        "axk2": [
            # The A.X-K2 hub checkpoints store the routed experts as individual `experts.{i}` projections
            # (DeepSeek-V3 layout); the model packs them into the stacked expert tensors.
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
            # The gated norms store their low-rank gate MLP as `W_down` / `W_up`. Patterns are unanchored (no
            # surrounding dots) so the same rename also normalizes the bare module names in an fp8 checkpoint's
            # `modules_to_not_convert` (`W_down`/`W_up` -> `gate_mlp.up_proj`/`gate_mlp.down_proj`); a dot-anchored
            # pattern would leave those bare names untouched and the gate MLP would be quantized by mistake.
            WeightRenaming(source_patterns=r"W_down", target_patterns="gate_mlp.up_proj"),
            WeightRenaming(source_patterns=r"W_up", target_patterns="gate_mlp.down_proj"),
            # The released checkpoints fuse the query up-projection and the attention output gate into a
            # single `q_b_proj` (vLLM layout); the model keeps it fused under the clearer name `q_gate_proj`
            # and splits the activation in the forward. A plain rename (kept fp8-safe: no weight split, and
            # the prefix also renames the `weight_scale_inv` companion).
            WeightRenaming(source_patterns=r"self_attn\.q_b_proj\.", target_patterns="self_attn.q_gate_proj."),
        ],
        "gemma4_unified": [
            WeightRenaming(source_patterns=r"vision_embedder\.patch_ln1", target_patterns="embed_vision.patch_ln1"),
            WeightRenaming(
                source_patterns=r"vision_embedder\.patch_dense", target_patterns="embed_vision.patch_dense"
            ),
            WeightRenaming(source_patterns=r"vision_embedder\.patch_ln2", target_patterns="embed_vision.patch_ln2"),
            WeightRenaming(
                source_patterns=r"vision_embedder\.pos_embedding", target_patterns="embed_vision.pos_embedding"
            ),
            WeightRenaming(source_patterns=r"vision_embedder\.pos_norm", target_patterns="embed_vision.pos_norm"),
            WeightRenaming(
                source_patterns=r"embed_vision\.embedding_projection",
                target_patterns="embed_vision.multimodal_embedder.embedding_projection",
            ),
        ],
        "radio": [
            WeightRenaming("radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"),
            WeightRenaming("radio_model.model.patch_generator.embedder", "embeddings.patch_projection"),
            WeightRenaming("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding"),
            WeightRenaming("radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"),
            WeightRenaming("radio_model.model.blocks", "encoder.layer"),
            WeightRenaming("attn.proj", "attention.o_proj"),
            WeightRenaming("radio_model.input_conditioner", "input_conditioner"),
            WeightRenaming("radio_model.summary_idxs", "summary_idxs"),
            WeightConverter(
                source_patterns="attn.qkv",
                target_patterns=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                operations=[Chunk(dim=0)],
            ),
        ],
        "NemotronH_Omni_Reasoning_V3": [
            WeightRenaming(r"^mlp1\.0\.", r"multi_modal_projector\.layer_norm\."),
            WeightRenaming(r"^mlp1\.1\.", r"multi_modal_projector\.linear_1\."),
            WeightRenaming(r"^mlp1\.3\.", r"multi_modal_projector\.linear_2\."),
            WeightRenaming(r"^sound_encoder\.encoder\.", r"audio_tower\."),
            WeightRenaming(r"^sound_projection\.norm\.", r"embed_audio\.layer_norm\."),
            WeightRenaming(r"^sound_projection\.linear1\.", r"embed_audio\.linear_1\."),
            WeightRenaming(r"^sound_projection\.linear2\.", r"embed_audio\.linear_2\."),
        ],
        "hrm_text": [
            WeightConverter(
                source_patterns="mlp.gate_up_proj.weight",
                target_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="attn.gqkv_proj.weight",
                target_patterns=[
                    "self_attn.gate_proj.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming(source_patterns=r"\.attn\.o_proj\.", target_patterns=".self_attn.o_proj."),
        ],
        "hunyuan_vl": [
            WeightRenaming(source_patterns=r"perceive\.proj\.0\.", target_patterns="perceive.proj_conv."),
            WeightRenaming(source_patterns=r"perceive\.proj\.2\.", target_patterns="perceive.proj_out."),
            WeightRenaming(source_patterns=r"perceive\.", target_patterns="patch_merger."),
            WeightRenaming(
                source_patterns=r"^model(?!\.(language_model|vit|vision_tower))",
                target_patterns="model.language_model",
            ),
            WeightRenaming(source_patterns=r"^model\.vit", target_patterns="model.vision_tower"),
            WeightRenaming(source_patterns=r"^vit", target_patterns="model.vision_tower"),
        ],
        "ViTModel": [
            WeightRenaming(r"encoder\.layer\.", "layers."),
            WeightRenaming("attention.query", "q_proj"),
            WeightRenaming("attention.key", "k_proj"),
            WeightRenaming("attention.value", "v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightRenaming("intermediate.dense", "mlp.up_proj"),
            WeightRenaming("output.dense", "mlp.down_proj"),
            *vit_layer_renames,
        ],
        "Dinov2Model": [
            WeightRenaming("attention.attention.query", "attention.q_proj"),
            WeightRenaming("attention.attention.key", "attention.k_proj"),
            WeightRenaming("attention.attention.value", "attention.v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightConverter(
                source_patterns="mlp.weights_in.weight",
                target_patterns=["swiglu_ffn.gate_proj.weight", "swiglu_ffn.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="mlp.weights_in.bias",
                target_patterns=["swiglu_ffn.gate_proj.bias", "swiglu_ffn.up_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.weights_out", "swiglu_ffn.down_proj"),
            *dino_layer_renames,
        ],
        "Dinov2WithRegistersModel": [
            WeightRenaming("attention.attention.query", "attention.q_proj"),
            WeightRenaming("attention.attention.key", "attention.k_proj"),
            WeightRenaming("attention.attention.value", "attention.v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightConverter(
                source_patterns="mlp.weights_in.weight",
                target_patterns=["swiglu_ffn.gate_proj.weight", "swiglu_ffn.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="mlp.weights_in.bias",
                target_patterns=["swiglu_ffn.gate_proj.bias", "swiglu_ffn.up_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.weights_out", "swiglu_ffn.down_proj"),
            *dino_layer_renames,
        ],
        "ViTMSNForImageClassification": [
            WeightRenaming(r"^encoder\.", "vit.encoder."),
        ],
        "ViTMAEForPreTraining": [
            WeightRenaming("attention.query", "q_proj"),
            WeightRenaming("attention.key", "k_proj"),
            WeightRenaming("attention.value", "v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightRenaming("intermediate.dense", "mlp.up_proj"),
            WeightRenaming("output.dense", "mlp.down_proj"),
            *vit_layer_renames,
        ],
        "BeitBackbone": [
            WeightRenaming(r"^fpn1\.0\.", "fpn.fpn1.conv_transpose1."),
            WeightRenaming(r"^fpn1\.1\.", "fpn.fpn1.normalization."),
            WeightRenaming(r"^fpn1\.3\.", "fpn.fpn1.conv_transpose2."),
            WeightRenaming(r"^fpn2\.0\.", "fpn.fpn2."),
            WeightRenaming(r"^encoder\.layer\.", "beit.encoder.layer."),
            WeightRenaming(r"^embeddings\.", "beit.embeddings."),
            WeightRenaming("attention.query", "q_proj"),
            WeightRenaming("attention.key", "k_proj"),
            WeightRenaming("attention.value", "v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightRenaming("intermediate.dense", "mlp.up_proj"),
            WeightRenaming("output.dense", "mlp.down_proj"),
        ],
        "BeitForSemanticSegmentation": [
            WeightRenaming(r"(?<!psp_modules\.[0-9]\.1\.)bn\.", "normalization."),
            WeightRenaming(r"(?<!psp_modules\.[0-9]\.1\.)conv\.weight", "convolution.weight"),
            WeightRenaming(r"^fpn1\.0\.", "fpn.fpn1.conv_transpose1."),
            WeightRenaming(r"^fpn1\.1\.", "fpn.fpn1.normalization."),
            WeightRenaming(r"^fpn1\.3\.", "fpn.fpn1.conv_transpose2."),
            WeightRenaming(r"^fpn2\.0\.", "fpn.fpn2."),
            WeightRenaming("decode_head.bottleneck.", "decode_head.psp_bottleneck."),
            WeightRenaming(
                r"decode_head\.psp_modules\.(\d+)\.1\.conv\.weight",
                r"decode_head.psp_modules.blocks.\1.conv.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.psp_modules\.(\d+)\.1\.bn\.",
                r"decode_head.psp_modules.blocks.\1.conv.normalization.",
            ),
        ],
        "lw_detr": [
            WeightRenaming("attention.attention.query", "attention.q_proj"),
            WeightRenaming("attention.attention.key", "attention.k_proj"),
            WeightRenaming("attention.attention.value", "attention.v_proj"),
            WeightRenaming("attention.output", "attention.o_proj"),
        ],
        "SegformerModel": [
            WeightRenaming(r"encoder.patch_embeddings.(\d+).", r"stages.\1.patch_embeddings."),
            WeightRenaming(r"encoder.block.(\d+).", r"stages.\1.blocks."),
            WeightRenaming(r"encoder.layer_norm.(\d+)", r"stages.\1.layer_norm"),
            WeightRenaming("attention.self.query", "attention.q_proj"),
            WeightRenaming("attention.self.key", "attention.k_proj"),
            WeightRenaming("attention.self.value", "attention.v_proj"),
            WeightRenaming("attention.self.sr", "attention.sequence_reduction.sequence_reduction"),
            WeightRenaming("attention.self.layer_norm", "attention.sequence_reduction.layer_norm"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightRenaming("mlp.dense1", "mlp.up_proj"),
            WeightRenaming("mlp.dense2", "mlp.down_proj"),
            WeightRenaming("layer_norm_1", "layernorm_before"),
            WeightRenaming("layer_norm_2", "layernorm_after"),
        ],
        "SegformerForSemanticSegmentation": [WeightRenaming("decode_head.linear_c", "decode_head.linear_projections")],
        "videomae": [
            WeightRenaming(r"attention\.attention\.q_bias", "attention.attention.query.bias"),
            WeightRenaming(r"attention\.attention\.v_bias", "attention.attention.value.bias"),
        ],
        "swin": [
            WeightRenaming("attention.self.query", "attention.q_proj"),
            WeightRenaming("attention.self.key", "attention.k_proj"),
            WeightRenaming("attention.self.value", "attention.v_proj"),
            WeightRenaming(
                "attention.self.relative_position_bias_table",
                "attention.relative_position_bias.relative_position_bias_table",
            ),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            WeightRenaming("intermediate.dense", "mlp.up_proj"),
            WeightRenaming("output.dense", "mlp.down_proj"),
        ],
        "SwinBackbone": [
            WeightRenaming(r"^encoder\.", "swin.encoder."),
            WeightRenaming(r"^embeddings\.", "swin.embeddings."),
        ],
        "altclip": [
            WeightRenaming(source_patterns=r"layer\.", target_patterns="layers."),
        ],
        "kimi_k25": [
            # Same as llava, add `model` as prefix if needed
            WeightRenaming(source_patterns=r"language_model.model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"language_model.lm_head", target_patterns="lm_head"),
            # norm/fc/wo renames come before encoder/blocks renames so their source
            # patterns match the original checkpoint format (vision_tower.encoder.blocks.X.*)
            GroupWeightRename(
                source_patterns=[
                    r"vision_tower.encoder.blocks.(\d+).norm0",
                    r"vision_tower.encoder.blocks.(\d+).norm1",
                ],
                target_patterns=[
                    r"vision_tower.encoder.blocks.\1.norm1",
                    r"vision_tower.encoder.blocks.\1.norm2",
                ],
            ),
            GroupWeightRename(
                source_patterns=[
                    r"vision_tower.encoder.blocks.(\d+).mlp.fc0",
                    r"vision_tower.encoder.blocks.(\d+).mlp.fc1",
                ],
                target_patterns=[
                    r"vision_tower.encoder.blocks.\1.mlp.fc1",
                    r"vision_tower.encoder.blocks.\1.mlp.fc2",
                ],
            ),
            WeightRenaming(
                source_patterns=r"vision_tower.encoder.blocks.(\d+).wo",
                target_patterns=r"vision_tower.encoder.blocks.\1.attn.proj",
            ),
            WeightRenaming(source_patterns=r"vision_tower.encoder", target_patterns="model.vision_tower"),
            # Rename projection modules
            WeightRenaming(source_patterns=r"mm_projector.proj.0", target_patterns="model.mm_projector.in_proj"),
            WeightRenaming(source_patterns=r"mm_projector.proj.2", target_patterns="model.mm_projector.out_proj"),
            WeightRenaming(source_patterns=r"blocks", target_patterns="layers"),
            WeightRenaming(
                source_patterns=r"vision_tower.patch_embed.pos_emb.weight",
                target_patterns="model.vision_tower.patch_embed.pos_emb.position_embeddings",
            ),
            # Unfuse qkv and apply rope permutation
            WeightConverter(
                source_patterns=[r"wqkv"],
                target_patterns=[
                    r"attn.q_proj",
                    r"attn.k_proj",
                    r"attn.v_proj",
                ],
                operations=[
                    Chunk(dim=0),
                    PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj", "k_proj"]),
                ],
            ),
        ],
        "deepseek_v4": [
            # General mid-key renames acting on several keys
            WeightRenaming(source_patterns=r"\.attn\.", target_patterns=r".self_attn."),
            WeightRenaming(source_patterns=r"\.ffn\.", target_patterns=r".mlp."),
            WeightRenaming(source_patterns=r"\.indexer\.compressor\.", target_patterns=r"\.compressor\.indexer\."),
            # Leaf patterns
            WeightRenaming(source_patterns=r"^embed\.weight", target_patterns="embed_tokens.weight"),
            WeightRenaming(source_patterns=r"^head\.weight", target_patterns="lm_head.weight"),
            WeightRenaming(source_patterns=r"^hc_head_fn", target_patterns="hc_head.hc_fn"),
            WeightRenaming(source_patterns=r"^hc_head_base", target_patterns="hc_head.hc_base"),
            WeightRenaming(source_patterns=r"^hc_head_scale", target_patterns="hc_head.hc_scale"),
            WeightRenaming(source_patterns=r"\.attn_norm\.", target_patterns=r".input_layernorm."),
            WeightRenaming(source_patterns=r"\.ffn_norm\.", target_patterns=r".post_attention_layernorm."),
            WeightRenaming(source_patterns=r"\.hc_attn_fn", target_patterns=r".attn_hc.fn"),
            WeightRenaming(source_patterns=r"\.hc_attn_base", target_patterns=r".attn_hc.base"),
            WeightRenaming(source_patterns=r"\.hc_attn_scale", target_patterns=r".attn_hc.scale"),
            WeightRenaming(source_patterns=r"\.hc_ffn_fn", target_patterns=r".ffn_hc.fn"),
            WeightRenaming(source_patterns=r"\.hc_ffn_base", target_patterns=r".ffn_hc.base"),
            WeightRenaming(source_patterns=r"\.hc_ffn_scale", target_patterns=r".ffn_hc.scale"),
            WeightRenaming(source_patterns=r"\.attn_sink", target_patterns=r".sinks"),
            WeightRenaming(
                source_patterns=r"\.indexer\.weights_proj\.",
                target_patterns=r"\.compressor\.indexer\.scorer\.weights_proj\.",
            ),
            WeightRenaming(source_patterns=r"\.indexer\.wq_b\.", target_patterns=r"\.compressor\.indexer\.q_b_proj\."),
            WeightRenaming(source_patterns=r"\.norm\.", target_patterns=r"\.kv_norm\."),
            WeightRenaming(source_patterns=r"\.ape", target_patterns=r"\.position_bias"),
            WeightRenaming(source_patterns=r"\.wq_a\.", target_patterns=r".q_a_proj."),
            WeightRenaming(source_patterns=r"\.self_attn\.wq_b\.", target_patterns=r"\.self_attn\.q_b_proj\."),
            WeightRenaming(source_patterns=r"\.wkv\.", target_patterns=r".kv_proj."),
            WeightRenaming(source_patterns=r"\.wgate\.", target_patterns=r".gate_proj."),
            WeightRenaming(source_patterns=r"\.wo_a\.", target_patterns=r".o_a_proj."),
            WeightRenaming(source_patterns=r"\.wo_b\.", target_patterns=r".o_b_proj."),
            WeightRenaming(source_patterns=r"\.q_norm\.", target_patterns=r".q_a_norm."),
            WeightRenaming(source_patterns=r"\.gate\.bias", target_patterns=r".gate.e_score_correction_bias"),
            WeightRenaming(source_patterns=r"\.shared_experts\.w1\.", target_patterns=r".shared_experts.gate_proj."),
            WeightRenaming(source_patterns=r"\.shared_experts\.w2\.", target_patterns=r".shared_experts.down_proj."),
            WeightRenaming(source_patterns=r"\.shared_experts\.w3\.", target_patterns=r".shared_experts.up_proj."),
            WeightConverter(
                source_patterns=[
                    r"\.experts.*.w1.weight",
                    r"\.experts.*.w3.weight",
                ],
                target_patterns=r"\.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns=r"\.experts.*.w2.weight",
                target_patterns=r"\.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ],
        "glm5_next": [
            WeightRenaming(
                source_patterns=r"self_attn\.f_a_proj\.",
                target_patterns=r"self_attn.forget_gate.f_a_proj.",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.f_b_proj\.",
                target_patterns=r"self_attn.forget_gate.f_b_proj.",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.dt_bias",
                target_patterns=r"self_attn.forget_gate.dt_bias",
            ),
            WeightRenaming(
                source_patterns=r"self_attn\.A_log",
                target_patterns=r"self_attn.forget_gate.A_log",
            ),
            WeightRenaming(source_patterns="hc_attn_fn", target_patterns="attn_hc.fn"),
            WeightRenaming(source_patterns="hc_attn_base", target_patterns="attn_hc.base"),
            WeightRenaming(source_patterns="hc_attn_scale", target_patterns="attn_hc.scale"),
            WeightRenaming(source_patterns="hc_ffn_fn", target_patterns="ffn_hc.fn"),
            WeightRenaming(source_patterns="hc_ffn_base", target_patterns="ffn_hc.base"),
            WeightRenaming(source_patterns="hc_ffn_scale", target_patterns="ffn_hc.scale"),
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
            WeightConverter(
                source_patterns=[
                    "self_attn.q_conv1d.weight",
                    "self_attn.k_conv1d.weight",
                    "self_attn.v_conv1d.weight",
                ],
                target_patterns="self_attn.conv1d.weight",
                operations=[Concatenate(dim=0)],
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
        "minimax_m3_vl": [
            WeightRenaming(source_patterns=r"^language_model\.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model\.model\.", target_patterns="model.language_model."),
            WeightRenaming(
                source_patterns=r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.",
                target_patterns="model.vision_tower.embeddings.proj.",
            ),
            WeightRenaming(
                source_patterns=r"^vision_tower\.vision_model\.encoder\.layers\.",
                target_patterns="model.vision_tower.layers.",
            ),
            WeightRenaming(
                source_patterns=r"^vision_tower\.vision_model\.pre_layrnorm\.",
                target_patterns="model.vision_tower.pre_layrnorm.",
            ),
            WeightRenaming(
                source_patterns=r"^multi_modal_projector\.linear_1\.",
                target_patterns="model.multi_modal_projector.linear_1.",
            ),
            WeightRenaming(
                source_patterns=r"^multi_modal_projector\.linear_2\.",
                target_patterns="model.multi_modal_projector.linear_2.",
            ),
            WeightRenaming(
                source_patterns=r"^patch_merge_mlp\.linear_1\.",
                target_patterns="model.multi_modal_projector.merge_linear_1.",
            ),
            WeightRenaming(
                source_patterns=r"^patch_merge_mlp\.linear_2\.",
                target_patterns="model.multi_modal_projector.merge_linear_2.",
            ),
            WeightRenaming(source_patterns=r"\.block_sparse_moe\.", target_patterns=r"\.mlp\."),
            WeightRenaming(
                source_patterns=r"\.e_score_correction_bias", target_patterns=r"\.gate\.e_score_correction_bias"
            ),
            WeightRenaming(
                source_patterns=r"\.self_attn\.index_q_proj\.", target_patterns=".self_attn.indexer.q_proj."
            ),
            WeightRenaming(
                source_patterns=r"\.self_attn\.index_k_proj\.", target_patterns=".self_attn.indexer.k_proj."
            ),
            WeightRenaming(
                source_patterns=r"\.self_attn\.index_q_norm\.", target_patterns=".self_attn.indexer.q_norm."
            ),
            WeightRenaming(
                source_patterns=r"\.self_attn\.index_k_norm\.", target_patterns=".self_attn.indexer.k_norm."
            ),
            WeightConverter(
                source_patterns=[
                    r"\.experts.*.w1.weight",
                    r"\.experts.*.w3.weight",
                ],
                target_patterns=r"\.experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns=r"\.experts.*.w2.weight",
                target_patterns=r"\.experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                source_patterns=[
                    r"\.gate_proj.weight",
                    r"(?<!vision_tower\.layers\.\d\.mlp)(?<!vision_tower\.layers\.\d\d\.mlp)"
                    r"(?<!vision_tower\.layers\.\d\d\d\.mlp)\.up_proj.weight",
                ],
                target_patterns=r"\.gate_up_proj.weight",
                operations=[Concatenate(dim=0)],
            ),
        ],
        "qwen2_audio": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^audio_tower", target_patterns="model.audio_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
        ],
        "Qwen2AudioModel": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="language_model"),
        ],
        "granite_speech": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^encoder", target_patterns="model.encoder"),
            WeightRenaming(source_patterns=r"^projector", target_patterns="model.projector"),
        ],
        # Legacy MoE weight names → standard ``@use_experts_implementation`` interface.
        "aria_text": [
            WeightRenaming(
                source_patterns=r"mlp\.experts\.fc1\.weight",
                target_patterns="mlp.experts.gate_up_proj",
            ),
            WeightRenaming(
                source_patterns=r"mlp\.experts\.fc2\.weight",
                target_patterns="mlp.experts.down_proj",
            ),
        ],
        "granitemoe": [
            WeightRenaming(
                source_patterns=r"(layers\.\d+)\.block_sparse_moe\.input_linear\.weight",
                target_patterns=r"\1.mlp.experts.gate_up_proj",
            ),
            WeightRenaming(
                source_patterns=r"(layers\.\d+)\.block_sparse_moe\.output_linear\.weight",
                target_patterns=r"\1.mlp.experts.down_proj",
            ),
            WeightRenaming(
                source_patterns=r"(layers\.\d+)\.block_sparse_moe\.router\.layer\.weight",
                target_patterns=r"\1.mlp.gate.weight",
            ),
        ],
        "GraniteSpeechModel": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="language_model"),
        ],
        # Mimi checkpoints use the original flat layout, whose SEANet stacks are `encoder.layers` / `decoder.layers`.
        # In `MimiModel`, its renaming to `encoder.encoder.layers` is followed by the halves' own, scoped to `encoder.`.
        # The checkpoint's `quantizer.` is `MimiModel.quantizer`, between the halves.
        "MimiEncoderModel": [
            WeightRenaming(source_patterns=r"^encoder\.layers", target_patterns="encoder_seanet.layers"),
        ],
        "MimiDecoderModel": [
            WeightRenaming(source_patterns=r"^decoder\.layers", target_patterns="decoder_seanet.layers"),
        ],
        "MimiModel": [
            WeightRenaming(source_patterns=r"^encoder\.layers", target_patterns="encoder.encoder.layers"),
            WeightRenaming(source_patterns=r"^encoder_transformer\.", target_patterns="encoder.encoder_transformer."),
            WeightRenaming(source_patterns=r"^downsample\.", target_patterns="encoder.downsample."),
            WeightRenaming(source_patterns=r"^upsample\.", target_patterns="decoder.upsample."),
            WeightRenaming(source_patterns=r"^decoder_transformer\.", target_patterns="decoder.decoder_transformer."),
            WeightRenaming(source_patterns=r"^decoder\.layers", target_patterns="decoder.decoder.layers"),
        ],
        "vibevoice_acoustic_tokenizer": [
            WeightRenaming(r"\.ffn\.linear1\.", ".ffn.up_proj."),
            WeightRenaming(r"\.ffn\.linear2\.", ".ffn.down_proj."),
        ],
        "vibevoice_asr": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(
                source_patterns=r"^acoustic_tokenizer_encoder", target_patterns="model.acoustic_tokenizer_encoder"
            ),
            WeightRenaming(
                source_patterns=r"^semantic_tokenizer_encoder", target_patterns="model.semantic_tokenizer_encoder"
            ),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
        ],
        "VibeVoiceAsrModel": [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="language_model"),
        ],
        "llava_next": [
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^language_model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^vision_tower", target_patterns="model.vision_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
            WeightRenaming(source_patterns=r"^image_newline", target_patterns="model.image_newline"),
        ],
        # Important to refer to classes by name, not model-type! Several classes share the same model type
        "CLIPVisionModel": [PrefixChange(prefix_to_remove="vision_model")],
        "CLIPTextModel": [PrefixChange(prefix_to_remove="text_model")],
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
        "molmo2": [
            WeightRenaming(
                source_patterns=r"(?<!image_vit\.)transformer\.(?!ln_f\.|blocks\.)", target_patterns="language_model."
            ),
            WeightRenaming(
                source_patterns=r"(?<!image_vit\.)transformer\.ln_f\.", target_patterns="language_model.norm."
            ),
            WeightRenaming(
                source_patterns=r"(?<!image_vit\.)transformer\.blocks\.", target_patterns="language_model.layers."
            ),
            WeightRenaming(
                source_patterns=r"vision_backbone\.image_vit\.(?!transformer\.resblocks\.)",
                target_patterns="vision_tower.",
            ),
            WeightRenaming(
                source_patterns=r"vision_backbone\.image_vit\.transformer\.resblocks\.",
                target_patterns="vision_tower.layers.",
            ),
            WeightRenaming(
                source_patterns=r"vision_backbone\.image_pooling_2d\.",
                target_patterns="multi_modal_projector.image_pooling_2d.",
            ),
            WeightRenaming(
                source_patterns=r"vision_backbone\.image_projector\.",
                target_patterns="multi_modal_projector.image_projector.",
            ),
            WeightRenaming(source_patterns=r"\.attention\.wq", target_patterns=".self_attn.q_proj"),
            WeightRenaming(source_patterns=r"\.attention\.wk", target_patterns=".self_attn.k_proj"),
            WeightRenaming(source_patterns=r"\.attention\.wv", target_patterns=".self_attn.v_proj"),
            WeightRenaming(source_patterns=r"\.attention\.wo", target_patterns=".self_attn.o_proj"),
            WeightRenaming(source_patterns=r"\.feed_forward\.w1", target_patterns=".mlp.up_proj"),
            WeightRenaming(source_patterns=r"\.feed_forward\.w2", target_patterns=".mlp.down_proj"),
            WeightRenaming(source_patterns=r"\.attention_norm", target_patterns=".input_layernorm"),
            WeightRenaming(source_patterns=r"\.ffn_norm", target_patterns=".post_attention_layernorm"),
            WeightRenaming(source_patterns=r"image_pooling_2d\.wq", target_patterns="image_pooling_2d.q_proj"),
            WeightRenaming(source_patterns=r"image_pooling_2d\.wk", target_patterns="image_pooling_2d.k_proj"),
            WeightRenaming(source_patterns=r"image_pooling_2d\.wv", target_patterns="image_pooling_2d.v_proj"),
            WeightRenaming(source_patterns=r"image_pooling_2d\.wo", target_patterns="image_pooling_2d.o_proj"),
            WeightRenaming(source_patterns=r"image_projector\.w1", target_patterns="image_projector.gate_proj"),
            WeightRenaming(source_patterns=r"image_projector\.w2", target_patterns="image_projector.down_proj"),
            WeightRenaming(source_patterns=r"image_projector\.w3", target_patterns="image_projector.up_proj"),
        ],
        # Scoped to the text model: its `self_attn.{q,k,v,o}_proj` targets also exist in the vision tower.
        "Molmo2TextModel": [
            WeightRenaming(source_patterns=r"\.attn_norm\.", target_patterns=".input_layernorm."),
            WeightRenaming(source_patterns=r"\.ff_norm\.", target_patterns=".post_attention_layernorm."),
            WeightConverter(
                source_patterns="self_attn.att_proj",
                target_patterns=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                operations=[Split(dim=0)],
            ),
            WeightRenaming(source_patterns=r"self_attn\.attn_out", target_patterns="self_attn.o_proj"),
            WeightRenaming(source_patterns=r"mlp\.ff_out", target_patterns="mlp.down_proj"),
            WeightConverter(
                source_patterns="mlp.ff_proj.weight",
                target_patterns=["mlp.up_proj.weight", "mlp.gate_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=["wte.embedding", "wte.new_embedding"],
                target_patterns="embed_tokens.weight",
                operations=[Concatenate(dim=0)],
            ),
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
        "Qwen2VLModel": [
            PrefixChange(prefix_to_add="language_model", model_prefix="model"),
        ],
        "Qwen2VLForConditionalGeneration": [
            WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
            WeightRenaming(
                source_patterns=r"^model(?!\.(language_model|visual))", target_patterns="model.language_model"
            ),
        ],
        "Qwen2_5OmniThinkerModel": [
            PrefixChange(prefix_to_remove="thinker"),
            PrefixChange(prefix_to_add="language_model", model_prefix="model"),
        ],
        "Qwen2_5OmniThinkerForConditionalGeneration": [
            PrefixChange(prefix_to_remove="thinker"),
            WeightRenaming(source_patterns=r"^audio_tower", target_patterns="model.audio_tower"),
            WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
            WeightRenaming(
                source_patterns=r"^model(?!\.(language_model|visual|audio_tower))",
                target_patterns="model.language_model",
            ),
        ],
        "deepseek_ocr2": [
            WeightRenaming(
                source_patterns=r"sam_model\.blocks\.(\d+)\.norm1\.",
                target_patterns=r"vision_tower.sam_encoder.layers.\1.input_layernorm.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.blocks\.(\d+)\.norm2\.",
                target_patterns=r"vision_tower.sam_encoder.layers.\1.post_attention_layernorm.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.blocks\.(\d+)\.attn\.",
                target_patterns=r"vision_tower.sam_encoder.layers.\1.attn.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.blocks\.(\d+)\.mlp\.",
                target_patterns=r"vision_tower.sam_encoder.layers.\1.mlp.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.patch_embed\.proj\.",
                target_patterns="vision_tower.sam_encoder.patch_embed.projection.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.pos_embed",
                target_patterns="vision_tower.sam_encoder.pos_embed",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.neck\.0\.",
                target_patterns="vision_tower.sam_encoder.neck.conv1.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.neck\.1\.",
                target_patterns="vision_tower.sam_encoder.neck.layer_norm1.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.neck\.2\.",
                target_patterns="vision_tower.sam_encoder.neck.conv2.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.neck\.3\.",
                target_patterns="vision_tower.sam_encoder.neck.layer_norm2.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.net_2\.",
                target_patterns="vision_tower.sam_encoder.proj.conv1.",
            ),
            WeightRenaming(
                source_patterns=r"sam_model\.net_3\.",
                target_patterns="vision_tower.sam_encoder.proj.conv2.",
            ),
            WeightRenaming(
                source_patterns=r"qwen2_model\.model\.model\.layers\.",
                target_patterns="vision_tower.vision_encoder.layers.",
            ),
            WeightRenaming(
                source_patterns=r"qwen2_model\.model\.model\.norm\.",
                target_patterns="vision_tower.vision_encoder.norm.",
            ),
            WeightRenaming(
                source_patterns=r"qwen2_model\.query_768\.",
                target_patterns="vision_tower.query_768_resolution.",
            ),
            WeightRenaming(
                source_patterns=r"qwen2_model\.query_1024\.",
                target_patterns="vision_tower.query_1024_resolution.",
            ),
            WeightRenaming(
                source_patterns=r"projector\.layers\.",
                target_patterns="multi_modal_projector.",
            ),
            WeightRenaming(source_patterns=r"view_seperator", target_patterns="view_separator"),
            WeightRenaming(
                source_patterns=r"(^|model\.)embed_tokens\.",
                target_patterns=r"\1language_model.embed_tokens.",
            ),
            WeightRenaming(source_patterns=r"(^|model\.)layers\.", target_patterns=r"\1language_model.layers."),
            WeightRenaming(source_patterns=r"(^|model\.)norm\.", target_patterns=r"\1language_model.norm."),
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
        "step3p7": [
            # ---- Pass 1: top-level prefix renames ----
            WeightRenaming(source_patterns=r"^vision_model\.", target_patterns="model.vision_model."),
            WeightRenaming(source_patterns=r"^vit_large_projector\.", target_patterns="model.multi_modal_projector."),
            # model.{embed_tokens,layers,norm}.* → model.language_model.* (explicit to avoid
            # negative lookaheads that break process_target_pattern during reverse mapping)
            WeightRenaming(
                source_patterns=r"^model\.embed_tokens\.", target_patterns="model.language_model.embed_tokens."
            ),
            WeightRenaming(source_patterns=r"^model\.layers\.", target_patterns="model.language_model.layers."),
            WeightRenaming(source_patterns=r"^model\.norm\.", target_patterns="model.language_model.norm."),
            # ---- Pass 2: MoE renames ----
            WeightRenaming(source_patterns=r"\.moe\.gate\.weight", target_patterns=".mlp.gate.weight"),
            WeightRenaming(
                source_patterns=r"\.moe\.router_bias",
                target_patterns=".mlp.gate.e_score_correction_bias",
            ),
            WeightRenaming(source_patterns=r"\.moe\.down_proj\.weight", target_patterns=".mlp.experts.down_proj"),
            WeightRenaming(
                source_patterns=r"\.moe\.down_proj\.weight_scale_inv",
                target_patterns=".mlp.experts.down_proj_scale_inv",
            ),
            WeightRenaming(source_patterns=r"\.share_expert\.", target_patterns=".mlp.shared_experts."),
            # ---- Tensor operations (run after all renames) ----
            # MoE gate_proj + up_proj → gate_up_proj (already stacked per layer, concat dim=1)
            WeightConverter(
                source_patterns=[r"moe.gate_proj.weight", r"moe.up_proj.weight"],
                target_patterns=r"mlp.experts.gate_up_proj",
                operations=[Concatenate(dim=1)],
            ),
        ],
        # Scoped separately from "step3p7" above: bare substring patterns like `\.attn\.` -> `.self_attn.`
        # would otherwise also match the text decoder's `language_model.layers.*.self_attn.*` keys.
        "step3p5_vision": [
            WeightRenaming(source_patterns=r"^conv1\.weight", target_patterns="embeddings.patch_embedding.weight"),
            WeightRenaming(
                source_patterns=r"^positional_embedding",
                target_patterns="embeddings.position_embedding.weight",
            ),
            WeightRenaming(source_patterns=r"^transformer\.resblocks\.", target_patterns="layers."),
            WeightRenaming(source_patterns=r"^ln_pre\.", target_patterns="pre_layernorm."),
            WeightRenaming(source_patterns=r"^vit_downsampler1\.", target_patterns="downsampler1."),
            WeightRenaming(source_patterns=r"^vit_downsampler2\.", target_patterns="downsampler2."),
            WeightRenaming(source_patterns=r"\.ls_1\.gamma", target_patterns=".lambda_1"),
            WeightRenaming(source_patterns=r"\.ls_2\.gamma", target_patterns=".lambda_2"),
            WeightRenaming(source_patterns=r"\.mlp\.c_fc\.", target_patterns=".mlp.up_proj."),
            WeightRenaming(source_patterns=r"\.mlp\.c_proj\.", target_patterns=".mlp.down_proj."),
            WeightRenaming(source_patterns=r"\.attn\.", target_patterns=".self_attn."),
            WeightRenaming(source_patterns=r"\.ln_1\.", target_patterns=".layernorm_before."),
            WeightRenaming(source_patterns=r"\.ln_2\.", target_patterns=".layernorm_after."),
            # Unfuse qkv and apply rope permutation.
            WeightConverter(
                source_patterns=r"attn\.in_proj_weight",
                target_patterns=[
                    r"attn.q_proj.weight",
                    r"attn.k_proj.weight",
                    r"attn.v_proj.weight",
                ],
                operations=[
                    Chunk(dim=0),
                    PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj", "k_proj"]),
                ],
            ),
            WeightConverter(
                source_patterns=r"attn\.in_proj_bias",
                target_patterns=[
                    r"attn.q_proj.bias",
                    r"attn.k_proj.bias",
                    r"attn.v_proj.bias",
                ],
                operations=[
                    Chunk(dim=0),
                    PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj", "k_proj"]),
                ],
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
        "yolos": [
            WeightRenaming(r"encoder.mid_position_embeddings", r"encoder.interpolation.mid_position_embeddings")
        ],
        "timesfm2_5": [
            WeightRenaming("ff0", "up_proj"),
            WeightRenaming("ff1", "down_proj"),
        ],
        "olmo_hybrid": [
            WeightRenaming("attention_layer_norm", "input_layernorm"),
            WeightRenaming("feedforward_layer_norm", "post_attention_layernorm"),
            WeightConverter(
                source_patterns=[
                    "linear_attn.q_conv1d.weight",
                    "linear_attn.k_conv1d.weight",
                    "linear_attn.v_conv1d.weight",
                ],
                target_patterns="linear_attn.conv1d.weight",
                operations=[Concatenate(dim=0)],
            ),
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
            WeightRenaming(
                r"(?<!vision_model.encoder\.)(?<!vision_tower.encoder\.)(?<!decoder\.)(?<!text_model\.)layers.",
                "text_model.layers.",
            ),
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
        "cosmos3_omni": [
            # Cosmos3 unified checkpoints store the Reasoner LLM with the old `model.` prefix
            # stripped off and the ViT under flat `blocks.*` / `merger.*` / `patch_embed.*` /
            # `pos_embed.*` / `deepstack_merger_list.*`. Re-target both to the nested Qwen3-VL
            # layout (`model.language_model.*` and `model.visual.*`). Newer checkpoints also
            # use Diffusers-style attention names; map them back to Qwen3-VL module names.
            WeightRenaming(
                source_patterns=r"^(layers\.|embed_tokens\.|norm\.)",
                target_patterns=r"model.language_model.\1",
            ),
            WeightRenaming(
                source_patterns=r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)",
                target_patterns=r"model.visual.\1",
            ),
            WeightRenaming(source_patterns=r"\.self_attn\.to_q\.", target_patterns=".self_attn.q_proj."),
            WeightRenaming(source_patterns=r"\.self_attn\.to_k\.", target_patterns=".self_attn.k_proj."),
            WeightRenaming(source_patterns=r"\.self_attn\.to_v\.", target_patterns=".self_attn.v_proj."),
            WeightRenaming(source_patterns=r"\.self_attn\.to_out\.", target_patterns=".self_attn.o_proj."),
            WeightRenaming(source_patterns=r"\.self_attn\.norm_q\.", target_patterns=".self_attn.q_norm."),
            WeightRenaming(source_patterns=r"\.self_attn\.norm_k\.", target_patterns=".self_attn.k_norm."),
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
        "lfm2": [
            WeightRenaming(source_patterns=r"\.feed_forward\.w1\.", target_patterns=r".feed_forward.gate_proj."),
            WeightRenaming(source_patterns=r"\.feed_forward\.w3\.", target_patterns=r".feed_forward.up_proj."),
            WeightRenaming(source_patterns=r"\.feed_forward\.w2\.", target_patterns=r".feed_forward.down_proj."),
        ],
        "lfm2_moe": [
            WeightRenaming(source_patterns=r"\.feed_forward\.w1\.", target_patterns=r".feed_forward.gate_proj."),
            WeightRenaming(source_patterns=r"\.feed_forward\.w3\.", target_patterns=r".feed_forward.up_proj."),
            WeightRenaming(source_patterns=r"\.feed_forward\.w2\.", target_patterns=r".feed_forward.down_proj."),
            WeightRenaming(r"\.feed_forward\.expert_bias", ".feed_forward.gate.e_score_correction_bias"),
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
            WeightRenaming("spatial_linear.0", "spatial_linear.up_proj"),
            WeightRenaming("spatial_linear.2", "spatial_linear.down_proj"),
            WeightRenaming("spatial_linear.3", "spatial_linear.ln"),
            WeightRenaming("temporal_linear.0", "temporal_linear.up_proj"),
            WeightRenaming("temporal_linear.2", "temporal_linear.down_proj"),
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
        # MaskFormer embeds both a Swin backbone (model_type="swin") and a DETR decoder
        # (model_type="detr") as submodules. Both get their mappings collected automatically, but
        # the Swin reverse mapping "mlp.up_proj → intermediate.dense" would corrupt DETR decoder keys
        # if it runs before the DETR reverse "layers.N.mlp.up_proj → layers.N.fc1".
        # By adding a "maskformer"-level mapping with the DETR fc1 rename, it is collected first
        # (model-level before submodule-level), so its reverse runs first and removes "mlp.up_proj"
        # from DETR decoder paths before the Swin reverse can match them.  # noqa: NC001, NC002
        "maskformer": [
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.up_proj"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.down_proj"),
        ],
        "DetrModel": [
            WeightRenaming("backbone.conv_encoder", "backbone"),
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.up_proj"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.down_proj"),
        ],
        "rt_detr": [
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.up_proj"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.down_proj"),
            WeightRenaming(r"encoder.encoder.(\d+).layers", r"encoder.aifi.\1.layers"),
        ],
        "pp_doclayout_v4": [
            WeightRenaming("decoder_roor_order_head.", "decoder.successor_order_head.proj."),
            WeightRenaming("decoder_roor_global_pointer.", "decoder.successor_order_head.global_pointer."),
            WeightRenaming("decoder_order_head.", "decoder.relative_order_head.proj."),
            WeightRenaming("decoder_global_pointer.", "decoder.relative_order_head.global_pointer."),
            WeightRenaming("s2r_fusion.a", "decoder.relative_order_head.s2r_fusion.closure_weight"),
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
            WeightRenaming(r"decoder.layers.(\d+).linear1", r"decoder.layers.\1.mlp.up_proj"),
            WeightRenaming(r"decoder.layers.(\d+).linear2", r"decoder.layers.\1.mlp.down_proj"),
            WeightRenaming(r"decoder.layers.(\d+).norm3", r"decoder.layers.\1.layer_norm"),
            WeightRenaming(r"decoder.norm", r"decoder.layernorm"),
            WeightRenaming(r"^transformer\.enc_output_norm", r"enc_output_norm"),
            WeightRenaming(r"^transformer\.enc_output", r"enc_output"),
            WeightRenaming(r"transformer.enc_out_class_embed", r"enc_out_class_embed"),
            WeightRenaming(r"transformer.enc_out_bbox_embed", r"enc_out_bbox_embed"),
            WeightRenaming(r"refpoint_embed\.weight", r"reference_point_embed.weight"),
            # RfDetrDinov2Backbone attention rename (legacy upstream uses split self-attention layout)
            WeightRenaming("attention.attention.query", "attention.q_proj"),
            WeightRenaming("attention.attention.key", "attention.k_proj"),
            WeightRenaming("attention.attention.value", "attention.v_proj"),
            WeightRenaming("attention.output.dense", "attention.o_proj"),
            # RfDetrAttention. The lookbehinds only apply on save (they are stripped from the load-time replacement):
            # they keep the reverse ops off the Dinov2 backbone's `layer.N.self_attn.*` keys
            WeightRenaming(r"self_attn.out_proj", rf"{_NOT_BACKBONE_LAYER}self_attn.o_proj"),
            WeightConverter(
                source_patterns=r"self_attn.in_proj_bias",
                target_patterns=[
                    rf"{_NOT_BACKBONE_LAYER}self_attn.q_proj.bias",
                    rf"{_NOT_BACKBONE_LAYER}self_attn.k_proj.bias",
                    rf"{_NOT_BACKBONE_LAYER}self_attn.v_proj.bias",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"self_attn.in_proj_weight",
                target_patterns=[
                    rf"{_NOT_BACKBONE_LAYER}self_attn.q_proj.weight",
                    rf"{_NOT_BACKBONE_LAYER}self_attn.k_proj.weight",
                    rf"{_NOT_BACKBONE_LAYER}self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
        ],
        "RfDetrForObjectDetection": [
            WeightRenaming(source_patterns="^", target_patterns="model."),
        ],
        "RfDetrForInstanceSegmentation": [
            WeightRenaming(source_patterns="^(?!segmentation_head)", target_patterns="model."),
            WeightRenaming(r"segmentation_head\.query_features_block\.layers\.0", r"query_features_block.mlp.up_proj"),
            WeightRenaming(
                r"segmentation_head\.query_features_block\.layers\.2", r"query_features_block.mlp.down_proj"
            ),
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
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.up_proj"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.down_proj"),
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
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.up_proj"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.down_proj"),
        ],
        "d_fine": [
            WeightRenaming("out_proj", "o_proj"),
            WeightRenaming(r"layers.(\d+).fc1", r"layers.\1.mlp.layers.0"),
            WeightRenaming(r"layers.(\d+).fc2", r"layers.\1.mlp.layers.1"),
            WeightRenaming(r"encoder.encoder.(\d+).layers", r"encoder.aifi.\1.layers"),
        ],
        "nemotron_h": [
            WeightRenaming("backbone.", "model."),
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
        "hyperclovax_vision_v2": [
            WeightRenaming(r"^model.language_model.lm_head", r"lm_head"),
            WeightRenaming(r"^model.vision_projector", r"model.projector"),
            PrefixChange(prefix_to_remove="model", model_prefix="model.language_model"),
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
        "gte": [
            PrefixChange(prefix_to_remove="new"),
            WeightRenaming(r"encoder.layer", r"layers"),
            WeightRenaming(r"attention.o_proj", r"self_attn.o_proj"),
            WeightRenaming(r"attn_ln", r"post_attention_layernorm"),
            WeightRenaming(r"mlp_ln", r"post_mlp_layernorm"),
            WeightRenaming(r"lm_head.norm", r"lm_head.layer_norm"),
            WeightConverter(
                source_patterns="attention.qkv_proj",
                target_patterns=[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="mlp.up_gate_proj",
                target_patterns=[
                    "mlp.up_proj",
                    "mlp.gate_proj",
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
            WeightRenaming(r"\.third_sub_layer\.dense_in", r".mlp.up_proj"),
            WeightRenaming(r"\.third_sub_layer\.dense_out", r".mlp.down_proj"),
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
            WeightRenaming(r"(layers\.\d+)\.attn\.proj\.", r"\1.self_attn.projection_layer."),
            WeightRenaming(r"\.embeddings\.class_embedding", r"\.embeddings\.cls_token"),
            WeightRenaming(r"\.embeddings\.position_embedding", r"\.embeddings\.position_embeddings"),
            WeightRenaming(r"\.embeddings\.patch_embedding\.", r"\.embeddings\.patch_embeddings\.projection\."),
            WeightRenaming(r"^language_model\.model\.", r"model\.language_model\."),
            WeightRenaming(r"^language_model\.lm_head\.", r"lm_head\."),
            WeightRenaming(r"^mlp1\.0\.", r"model\.multi_modal_projector\.layer_norm\."),
            WeightRenaming(r"^mlp1\.1\.", r"model\.multi_modal_projector\.linear_1\."),
            WeightRenaming(r"^mlp1\.3\.", r"model\.multi_modal_projector\.linear_2\."),
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
        "sapiens2": [
            WeightRenaming(r"^cls_token", r"embeddings.cls_token"),
            WeightRenaming(r"^storage_tokens", r"embeddings.register_tokens"),
            WeightRenaming(r"^patch_embed\.projection\.", r"embeddings.patch_embeddings."),
            WeightRenaming(r"blocks\.(\d+)\.", r"model.layer.\1."),
            WeightRenaming(r"attn\.proj\.", r"attention.o_proj."),
            WeightRenaming(r"attn\.wq\.", r"attention.q_proj."),
            WeightRenaming(r"attn\.wk\.", r"attention.k_proj."),
            WeightRenaming(r"attn\.wv\.", r"attention.v_proj."),
            WeightRenaming(r"attn\.q_norm\.", r"attention.q_norm."),
            WeightRenaming(r"attn\.k_norm\.", r"attention.k_norm."),
            WeightRenaming(r"attn\.gamma\.weight", r"layer_scale1.lambda1"),
            WeightRenaming(r"ffn\.w3\.", r"mlp.down_proj."),
            WeightRenaming(r"\.ln1\.", r".norm1."),
            WeightRenaming(r"\.ln2\.", r".norm2."),
            WeightRenaming(r"^ln1\.", r"norm."),
            WeightConverter(
                source_patterns=r"ffn\.w12\.weight",
                target_patterns=[r"mlp.gate_proj.weight", r"mlp.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"ffn\.w12\.bias",
                target_patterns=[r"mlp.gate_proj.bias", r"mlp.up_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
        ],
        "Sapiens2ForSemanticSegmentation": [
            WeightRenaming(r"^backbone\.", r"model."),
            WeightRenaming(
                r"decode_head\.deconv_layers\.0\.weight", r"decode_head.upsample_layers.0.convolution.weight"
            ),
            WeightRenaming(
                r"decode_head\.deconv_layers\.3\.weight", r"decode_head.upsample_layers.1.convolution.weight"
            ),
            WeightRenaming(
                r"decode_head\.deconv_layers\.6\.weight", r"decode_head.upsample_layers.2.convolution.weight"
            ),
            WeightRenaming(
                r"decode_head\.deconv_layers\.9\.weight", r"decode_head.upsample_layers.3.convolution.weight"
            ),
            WeightRenaming(r"decode_head\.conv_layers\.0\.(weight|bias)", r"decode_head.conv_layers.0.convolution.\1"),
            WeightRenaming(r"decode_head\.conv_layers\.3\.(weight|bias)", r"decode_head.conv_layers.1.convolution.\1"),
            WeightRenaming(r"decode_head\.conv_seg\.", r"decode_head.predictor."),
        ],
        "Sapiens2ForPoseEstimation": [
            WeightRenaming(r"^backbone\.", r"model."),
            WeightRenaming(
                r"decode_head\.deconv_layers\.0\.weight", r"decode_head.upsample_layers.0.convolution.weight"
            ),
            WeightRenaming(
                r"decode_head\.deconv_layers\.3\.weight", r"decode_head.upsample_layers.1.convolution.weight"
            ),
            WeightRenaming(r"decode_head\.conv_layers\.0\.(weight|bias)", r"decode_head.conv_layers.0.convolution.\1"),
            WeightRenaming(r"decode_head\.conv_layers\.3\.(weight|bias)", r"decode_head.conv_layers.1.convolution.\1"),
            WeightRenaming(r"decode_head\.conv_layers\.6\.(weight|bias)", r"decode_head.conv_layers.2.convolution.\1"),
            WeightRenaming(r"decode_head\.conv_pose\.", r"decode_head.predictor."),
        ],
        "Sapiens2ForNormalEstimation": [
            WeightRenaming(r"^backbone\.", r"model."),
            WeightRenaming(
                r"decode_head\.input_conv\.0\.(weight|bias)",
                r"decode_head.input_conv.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.weight",
                r"decode_head.upsample_layers.\1.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.bias",
                r"decode_head.upsample_layers.\1.convolution.bias",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.0\.(weight|bias)",
                r"decode_head.conv_layers.0.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.3\.(weight|bias)",
                r"decode_head.conv_layers.1.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.6\.(weight|bias)",
                r"decode_head.conv_layers.2.convolution.\1",
            ),
            WeightRenaming(r"decode_head\.conv_normal\.", r"decode_head.predictor."),
        ],
        "Sapiens2ForPointmapEstimation": [
            WeightRenaming(r"^backbone\.", r"model."),
            WeightRenaming(
                r"decode_head\.input_conv\.0\.(weight|bias)",
                r"decode_head.input_conv.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.weight",
                r"decode_head.upsample_layers.\1.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.bias",
                r"decode_head.upsample_layers.\1.convolution.bias",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.0\.(weight|bias)",
                r"decode_head.conv_layers.0.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.3\.(weight|bias)",
                r"decode_head.conv_layers.1.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.6\.(weight|bias)",
                r"decode_head.conv_layers.2.convolution.\1",
            ),
            WeightRenaming(r"decode_head\.conv_pointmap\.", r"decode_head.predictor."),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.0\.weight",
                r"scale_head.conv_layers.0.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.0\.bias",
                r"scale_head.conv_layers.0.convolution.bias",
            ),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.3\.weight",
                r"scale_head.conv_layers.1.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.3\.bias",
                r"scale_head.conv_layers.1.convolution.bias",
            ),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.6\.weight",
                r"scale_head.conv_layers.2.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.scale_conv_layers\.6\.bias",
                r"scale_head.conv_layers.2.convolution.bias",
            ),
            WeightRenaming(r"decode_head\.scale_final_layer\.1\.", r"scale_head.predictor.block1.layers.0."),
            WeightRenaming(r"decode_head\.scale_final_layer\.3\.", r"scale_head.predictor.block2.layers.0."),
            WeightRenaming(r"decode_head\.scale_final_layer\.5\.", r"scale_head.predictor.proj."),
        ],
        "Sapiens2ForImageMatting": [
            WeightRenaming(r"^backbone\.", r"model."),
            WeightRenaming(
                r"decode_head\.input_conv\.0\.(weight|bias)",
                r"decode_head.input_conv.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.weight",
                r"decode_head.upsample_layers.\1.convolution.weight",
            ),
            WeightRenaming(
                r"decode_head\.upsample_blocks\.(\d+)\.0\.bias",
                r"decode_head.upsample_layers.\1.convolution.bias",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.0\.(weight|bias)",
                r"decode_head.conv_layers.0.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.3\.(weight|bias)",
                r"decode_head.conv_layers.1.convolution.\1",
            ),
            WeightRenaming(
                r"decode_head\.conv_layers\.6\.(weight|bias)",
                r"decode_head.conv_layers.2.convolution.\1",
            ),
            WeightRenaming(r"decode_head\.conv_matting\.", r"decode_head.predictor."),
        ],
        "unlimited_ocr": [
            WeightRenaming(r"(^|model\.)sam_model", r"\1vision_tower.sam_encoder"),
            WeightRenaming(r"(^|model\.)vision_model", r"\1vision_tower.vision_encoder"),
            WeightRenaming(r"(^|model\.)view_seperator", r"\1view_separator"),
            WeightRenaming(r"(^|model\.)projector\.layers\.weight", r"\1multi_modal_projector.weight"),
            WeightRenaming(r"(^|model\.)projector\.layers\.bias", r"\1multi_modal_projector.bias"),
            WeightRenaming(r"(^|model\.)embed_tokens\.", r"\1language_model.embed_tokens."),
            WeightRenaming(r"(^|model\.)layers\.", r"\1language_model.layers."),
            WeightRenaming(r"(^|model\.)norm\.", r"\1language_model.norm."),
        ],
        "UnlimitedOcrSamVisionEncoder": [
            WeightRenaming(r"blocks\.(\d+)\.norm1\.", r"layers.\1.input_layernorm."),
            WeightRenaming(r"blocks\.(\d+)\.norm2\.", r"layers.\1.post_attention_layernorm."),
            WeightRenaming(r"blocks\.(\d+)\.attn\.", r"layers.\1.attn."),
            WeightRenaming(r"blocks\.(\d+)\.mlp\.", r"layers.\1.mlp."),
            WeightRenaming(r"patch_embed\.proj\.", "patch_embed.projection."),
            WeightRenaming(r"neck\.0\.", "neck.conv1."),
            WeightRenaming(r"neck\.1\.", "neck.layer_norm1."),
            WeightRenaming(r"neck\.2\.", "neck.conv2."),
            WeightRenaming(r"neck\.3\.", "neck.layer_norm2."),
            WeightRenaming(r"net_2\.", "proj.conv1."),
            WeightRenaming(r"net_3\.", "proj.conv2."),
        ],
        "UnlimitedOcrVisionEncoder": [
            WeightRenaming(r"transformer", "encoder"),
            WeightConverter(
                source_patterns="qkv_proj.weight",
                target_patterns=[
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns="qkv_proj.bias",
                target_patterns=[
                    "q_proj.bias",
                    "k_proj.bias",
                    "v_proj.bias",
                ],
                operations=[Chunk(dim=0)],
            ),
        ],
        "tipsv2": [
            WeightRenaming("text_encoder", "text_model"),
            WeightRenaming("vision_encoder", "vision_model"),
        ],
        "Tipsv2TextModel": [
            PrefixChange(prefix_to_remove="text_encoder"),
            WeightRenaming(r"ln_final\.", "final_layer_norm."),
            WeightRenaming(r"token_embedding\.", "embeddings.token_embedding."),
            WeightRenaming(r"transformer\.resblocks\.", r"encoder.layers."),
            WeightRenaming(r"\.ln_1\.", ".input_layernorm."),
            WeightRenaming(r"\.ln_2\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn\.out_proj\.", ".attn.o_proj."),
            WeightRenaming(r"\.attn\.", ".self_attn."),
            WeightRenaming(r"\.mlp\.c_fc\.", ".mlp.up_proj."),
            WeightRenaming(r"\.mlp\.c_proj\.", ".mlp.down_proj."),
            WeightConverter(
                source_patterns=r"\.in_proj_weight",
                target_patterns=[".q_proj.weight", ".k_proj.weight", ".v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"\.in_proj_bias",
                target_patterns=[".q_proj.bias", ".k_proj.bias", ".v_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
        ],
        "Tipsv2VisionModel": [
            PrefixChange(prefix_to_remove="vision_encoder"),
            WeightRenaming(r"patch_embed\.proj\.", "embeddings.patch_embeddings.projection."),
            WeightRenaming(r"cls_token", "embeddings.cls_token"),
            WeightRenaming(r"mask_token", "embeddings.mask_token"),
            WeightRenaming(r"register_tokens", "embeddings.register_tokens"),
            WeightRenaming(r"pos_embed", "embeddings.position_embeddings"),
            WeightRenaming(r"norm\.", "layernorm."),
            WeightRenaming(r"blocks\.", r"encoder.layer."),
            WeightRenaming(r"\.attn\.proj\.", ".attention.o_proj."),
            WeightRenaming(r"\.ls1\.gamma", ".layer_scale1.lambda1"),
            WeightRenaming(r"\.ls2\.gamma", ".layer_scale2.lambda1"),
            WeightRenaming(r"\.mlp\.c_fc\.", ".mlp.up_proj."),  # if config.use_swiglu_ffn=False
            WeightRenaming(r"\.mlp\.c_proj\.", ".mlp.down_proj."),  # if config.use_swiglu_ffn=False
            WeightRenaming(r"\.mlp\.w3\.", ".swiglu_ffn.down_proj."),  # if config.use_swiglu_ffn=True
            WeightConverter(
                source_patterns=r"\.mlp\.w12\.weight",  # if config.use_swiglu_ffn=True
                target_patterns=[".swiglu_ffn.gate_proj.weight", ".swiglu_ffn.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"\.mlp\.w12\.bias",  # if config.use_swiglu_ffn=True
                target_patterns=[".swiglu_ffn.gate_proj.bias", ".swiglu_ffn.up_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"\.attn\.qkv\.weight",
                target_patterns=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"\.attn\.qkv\.bias",
                target_patterns=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
        ],
        "tipsv2_dpt": [
            WeightRenaming("vision_encoder", "backbone"),
            WeightRenaming(
                r"head\.reassemble\.readout_projects\.(\d+)",
                r"neck.reassemble_stage.readout_projects.\1.layers.0",
            ),
            WeightRenaming(
                r"head\.reassemble\.out_projections\.(\d+)\.", r"neck.reassemble_stage.layers.\1.projection."
            ),
            WeightRenaming(r"head\.reassemble\.resize_layers\.(\d+)\.", r"neck.reassemble_stage.layers.\1.resize."),
            WeightRenaming(r"head\.fusion_blocks\.(\d+)\.out_conv\.", r"neck.fusion_stage.layers.\1.projection."),
            WeightRenaming(
                r"head\.fusion_blocks\.(\d+)\.main_unit\.conv1\.",
                r"neck.fusion_stage.layers.\1.residual_layer2.convolution1.",
            ),
            WeightRenaming(
                r"head\.fusion_blocks\.(\d+)\.main_unit\.conv2\.",
                r"neck.fusion_stage.layers.\1.residual_layer2.convolution2.",
            ),
            WeightRenaming(
                r"head\.fusion_blocks\.(\d+)\.residual_unit\.conv1\.",
                r"neck.fusion_stage.layers.\1.residual_layer1.convolution1.",
            ),
            WeightRenaming(
                r"head\.fusion_blocks\.(\d+)\.residual_unit\.conv2\.",
                r"neck.fusion_stage.layers.\1.residual_layer1.convolution2.",
            ),
            WeightRenaming(r"head\.convs", "neck.convs"),
            WeightRenaming(r"head\.project", "decoder.project"),
            WeightRenaming(r"depth_head\.depth_head\.", "depth_decoder.head."),
            WeightRenaming(r"normals_head\.normals_head\.", "normals_decoder.head."),
            WeightRenaming(r"segmentation_head\.segmentation_head\.", "segmentation_decoder.head."),
        ],
        "Tipsv2DptForDepthEstimation": [
            WeightRenaming("vision_encoder", "backbone"),
            WeightRenaming(
                r"depth_head\.reassemble\.readout_projects\.(\d+)",
                r"neck.reassemble_stage.readout_projects.\1.layers.0",
            ),
            WeightRenaming(
                r"depth_head\.reassemble\.out_projections\.(\d+)\.", r"neck.reassemble_stage.layers.\1.projection."
            ),
            WeightRenaming(
                r"depth_head\.reassemble\.resize_layers\.(\d+)\.", r"neck.reassemble_stage.layers.\1.resize."
            ),
            WeightRenaming(
                r"depth_head\.fusion_blocks\.(\d+)\.out_conv\.", r"neck.fusion_stage.layers.\1.projection."
            ),
            WeightRenaming(
                r"depth_head\.fusion_blocks\.(\d+)\.main_unit\.conv1\.",
                r"neck.fusion_stage.layers.\1.residual_layer2.convolution1.",
            ),
            WeightRenaming(
                r"depth_head\.fusion_blocks\.(\d+)\.main_unit\.conv2\.",
                r"neck.fusion_stage.layers.\1.residual_layer2.convolution2.",
            ),
            WeightRenaming(
                r"depth_head\.fusion_blocks\.(\d+)\.residual_unit\.conv1\.",
                r"neck.fusion_stage.layers.\1.residual_layer1.convolution1.",
            ),
            WeightRenaming(
                r"depth_head\.fusion_blocks\.(\d+)\.residual_unit\.conv2\.",
                r"neck.fusion_stage.layers.\1.residual_layer1.convolution2.",
            ),
            WeightRenaming(r"depth_head\.convs", "neck.convs"),
            WeightRenaming(r"depth_head\.project", "decoder.project"),
            WeightRenaming(r"depth_head\.depth_head\.", "decoder.head."),
        ],
        "UltrabertModel": [
            PrefixChange(prefix_to_remove="ultrabert"),
        ],
        "UltrabertForMaskedLM": [
            WeightRenaming(r"^ultrabert\.", "model."),
            WeightRenaming(r"^cls\.", "head.decoder."),
        ],
        "UltrabertForSequenceClassification": [
            WeightRenaming(r"^ultrabert\.", "model."),
        ],
        "UltrabertForTokenClassification": [
            WeightRenaming(r"^ultrabert\.", "model."),
        ],
        "UltrabertForQuestionAnswering": [
            WeightRenaming(r"^ultrabert\.", "model."),
        ],
        "UltraVBertForMaskedLM": [
            WeightRenaming(r"^head\.decoder\.", "lm_head."),
            WeightRenaming(r"^head\.", "projection_head."),
        ],
    }

    mapping["Tipsv2DptForNormalEstimation"] = [
        WeightRenaming(renaming.source_patterns[0].replace("depth", "normals"), renaming.target_patterns[0])
        for renaming in mapping["Tipsv2DptForDepthEstimation"]
    ]
    mapping["Tipsv2DptForSemanticSegmentation"] = [
        WeightRenaming(renaming.source_patterns[0].replace("depth", "segmentation"), renaming.target_patterns[0])
        for renaming in mapping["Tipsv2DptForDepthEstimation"]
    ]

    # The legacy mapping is added to the esm model here since the extra weight renaming do not apply to the esm model.
    mapping["esm"] += mapping["legacy"].copy()

    mapping["legacy"] += [
        WeightRenaming(
            source_patterns=".weight_g",
            target_patterns=".parametrizations.weight.original0",
        ),
        WeightRenaming(
            source_patterns=".weight_v",
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

    mapping["kimi_k25"] += mapping["qwen2_moe"].copy()

    # The pp_doclayout_v4-specific reading order renames are defined in the mapping literal above; it also
    # inherits the shared RT-DETR renames, like its PP-DocLayoutV2/V3 siblings.
    mapping["pp_doclayout_v4"] += mapping["rt_detr"].copy()

    mapping["bailing_hybrid"] = mapping["qwen2_moe"].copy()
    mapping["bailing_hybrid"] += [
        WeightRenaming(r"^model\.word_embeddings\.", "model.embed_tokens."),
        WeightRenaming(r"\.attention\.", ".self_attn."),
        WeightRenaming(r"self_attn\.f_proj\.", r"self_attn.forget_gate.f_proj."),
        WeightRenaming(r"self_attn\.dt_bias", r"self_attn.forget_gate.dt_bias"),
        WeightRenaming(r"self_attn\.A_log", r"self_attn.forget_gate.A_log"),
        # Suffix-only patterns keep this converter reversible across the attention -> self_attn rename.
        # Pack the official q/k/v kernels once at load time for a single runtime depthwise convolution.
        WeightConverter(
            source_patterns=[
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
            ],
            target_patterns="conv1d.weight",
            operations=[Concatenate(dim=0)],
        ),
        WeightRenaming(r"\.self_attn\.dense\.", ".self_attn.o_proj."),
        WeightRenaming(r"\.mlp\.gate\.expert_bias", ".mlp.gate.e_score_correction_bias"),
    ]

    mapping["ernie4_5_moe"] = mapping["qwen2_moe"].copy()
    mapping["ernie4_5_moe"] += [
        WeightRenaming("mlp.moe_statics.e_score_correction_bias", "mlp.gate.moe_statics.e_score_correction_bias")
    ]
    mapping["BeitModel"] = (
        mapping["ViTModel"][: -len(vit_layer_renames)]
        + [
            WeightRenaming("encoder.relative_position_bias", "shared_position_bias"),
            WeightRenaming(
                "attention.attention.relative_position_bias.relative_position_bias_table",
                "relative_position_bias.relative_position_bias_table",
            ),
        ]
        + vit_layer_renames
    )

    mapping["pixio"] = [
        WeightRenaming(r"encoder\.layer\.", "layers."),
        WeightRenaming("attention.query", "q_proj"),
        WeightRenaming("attention.key", "k_proj"),
        WeightRenaming("attention.value", "v_proj"),
        WeightRenaming("attention.output.dense", "attention.o_proj"),
        WeightRenaming("intermediate.dense", "mlp.up_proj"),
        WeightRenaming("output.dense", "mlp.down_proj"),
    ]
    mapping["qwen3_vl_vision"] = [
        WeightRenaming(r"\.mlp\.linear_fc1\.", ".mlp.up_proj."),
        WeightRenaming(r"\.mlp\.linear_fc2\.", ".mlp.down_proj."),
    ]
    gpt_neox_mlp_renames = [
        WeightRenaming(r"\.mlp\.dense_h_to_4h\.", ".mlp.up_proj."),
        WeightRenaming(r"\.mlp\.dense_4h_to_h\.", ".mlp.down_proj."),
    ]
    mapping["gpt_neox"] = gpt_neox_mlp_renames
    mapping["persimmon"] = gpt_neox_mlp_renames.copy()
    mapping["GPTNeoXForCausalLM"] = [
        WeightRenaming(source_patterns=r"^embed_out\.", target_patterns="lm_head."),
        *gpt_neox_mlp_renames,
    ]
    mapping["Qwen3OmniMoeTalkerForConditionalGeneration"] = [
        WeightRenaming(r"text_projection\.linear_fc1\.", "text_projection.up_proj."),
        WeightRenaming(r"text_projection\.linear_fc2\.", "text_projection.down_proj."),
        WeightRenaming(r"hidden_projection\.linear_fc1\.", "hidden_projection.up_proj."),
        WeightRenaming(r"hidden_projection\.linear_fc2\.", "hidden_projection.down_proj."),
    ]
    aria_projector_renames = [
        WeightRenaming(r"\.feed_forward\.linear_in\.", ".feed_forward.up_proj."),
        WeightRenaming(r"\.feed_forward\.linear_out\.", ".feed_forward.down_proj."),
    ]
    mapping["aria"] = mapping["llava"].copy() + aria_projector_renames
    mapping["AriaModel"] = mapping["LlavaModel"].copy() + aria_projector_renames
    granitemoe_shared_mlp_renames = [
        WeightRenaming(r"\.shared_mlp\.input_linear\.", ".shared_mlp.gate_up_proj."),
        WeightRenaming(r"\.shared_mlp\.output_linear\.", ".shared_mlp.down_proj."),
    ]
    mapping["granitemoeshared"] = mapping["granitemoe"].copy() + granitemoe_shared_mlp_renames
    mapping["granitemoehybrid"] = mapping["granitemoe"].copy() + granitemoe_shared_mlp_renames
    mapping["granitemoe_swa"] = granitemoe_shared_mlp_renames + granitemoe_block_renames
    modernbert_mlp_renames = [
        WeightRenaming(r"\.mlp\.Wi\.", ".mlp.gate_up_proj."),
        WeightRenaming(r"\.mlp\.Wo\.", ".mlp.down_proj."),
    ]
    mapping["modernbert"] = modernbert_mlp_renames
    mapping["modernbert-decoder"] = modernbert_mlp_renames.copy()
    mapping["starcoder2"] = [
        WeightRenaming(r"\.mlp\.c_fc\.", ".mlp.up_proj."),
        WeightRenaming(r"\.mlp\.c_proj\.", ".mlp.down_proj."),
    ]
    mapping["qwen2_5_omni_dit"] = [
        WeightRenaming(r"\.ff\.ff\.0\.", ".ff.up_proj."),
        WeightRenaming(r"\.ff\.ff\.3\.", ".ff.down_proj."),
    ]
    minicpmv4_6_merger_renames = [
        WeightRenaming(r"merger\.mlp\.(\d+)\.linear_1\.", r"merger.mlp.\1.up_proj."),
        WeightRenaming(r"merger\.mlp\.(\d+)\.linear_2\.", r"merger.mlp.\1.down_proj."),
    ]
    mapping["minicpmv4_6"] = minicpmv4_6_merger_renames
    mapping["minicpmv4_7"] = minicpmv4_6_merger_renames.copy()
    mapping["pixio"] += [
        WeightRenaming("norm1", "input_layernorm"),
        WeightRenaming("norm2", "post_attention_layernorm"),
        *vit_layer_renames[:1],
        WeightRenaming(r"^encoder\.", "pixio."),
    ]

    mapping["minimax_m2"] = [
        WeightRenaming(r"\.block_sparse_moe\.gate\.", ".mlp.gate."),
        WeightRenaming(r"\.block_sparse_moe\.experts\.", ".mlp.experts."),
        WeightRenaming(r"\.block_sparse_moe\.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"),
        *mapping["mixtral"][1:],
    ]
    mapping["hunyuan_v1_moe"] = mapping["qwen2_moe"].copy()
    mapping["hunyuan_v1_moe"] += [
        WeightRenaming(r"\.mlp\.shared_mlp\.", ".mlp.shared_experts."),
    ]
    mapping["exaone_moe"] = mapping["qwen2_moe"].copy()
    mapping["exaone_moe"] += [WeightRenaming("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")]

    mapping["kolibri1"] = mapping["qwen2_moe"].copy()
    mapping["kolibri1"] += [
        WeightRenaming("moe.router.expert_bias", "mlp.gate.e_score_correction_bias"),
        WeightRenaming("post_attn_norm", "post_self_attn_layernorm"),
        WeightRenaming("post_ffn_norm", "post_mlp_layernorm"),
    ]

    mapping["mimo_v2_flash"] = mapping["qwen2_moe"].copy()
    mapping["mimo_v2_flash"] += [
        WeightRenaming("self_attn.attention_sink_bias", "self_attn.sinks"),
    ]
    mapping["flint"] = mapping["qwen2_moe"].copy()
    mapping["flint"] += [
        WeightRenaming("self_attn.softmax_offset", "self_attn.sinks"),
    ]
    # HYV3: qwen2_moe expert fusion + attribute renames for MiniMaxM2-style inheritance
    mapping["hy_v3"] = mapping["qwen2_moe"].copy()
    mapping["hy_v3"] += [
        WeightRenaming(source_patterns=r"mlp\.router\.gate\.weight", target_patterns="mlp.gate.weight"),
        WeightRenaming(source_patterns=r"mlp\.expert_bias", target_patterns="mlp.gate.e_score_correction_bias"),
        WeightRenaming(source_patterns=r"mlp\.shared_mlp\.", target_patterns="mlp.shared_experts."),
    ]
    mapping["qwen3_5_moe_text"] = mapping["qwen3_5_text"].copy()
    mapping["qwen3_5_moe_text"] += mapping["qwen2_moe"].copy()

    mapping["laguna"] = mapping["qwen2_moe"].copy()
    mapping["laguna"] += [
        WeightRenaming("mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
        WeightRenaming("mlp.shared_expert.", "mlp.shared_experts."),
    ]

    # A.X-K1 checkpoints store the MoE output norm at the decoder-layer level.
    mapping["axk1"] = mapping["qwen2_moe"].copy()
    mapping["axk1"] += [
        WeightRenaming("post_mlp_layernorm", "mlp.post_mlp_layernorm"),
    ]

    mapping["qwen4_exp_text"] = mapping["qwen3_5_moe_text"].copy()
    mapping["qwen4_exp_text"] += [
        WeightConverter(
            source_patterns="ngram_embedding.shard_*.weight",
            target_patterns="ngram_embedding.weight",
            operations=[Concatenate(dim=0, num_shards_attribute="split_ngram_parts")],
            # The size of the embedding is ~95 GiB, so we cannot afford to perform the Cat on device, as it will need
            # a temporary memory buffer of the same size
            force_cpu=True,
        ),
    ]

    mapping["MtpModel"] = [
        PrefixChange(prefix_to_remove="model"),
        PrefixChange(prefix_to_remove="mtp"),
        WeightRenaming(source_patterns=".shared_head.norm.", target_patterns=".post_norm."),
        WeightRenaming(source_patterns=".mtp_block.enorm.", target_patterns=".enorm."),
        WeightRenaming(source_patterns=".mtp_block.hnorm.", target_patterns=".hnorm."),
        WeightRenaming(source_patterns=".mtp_block.eh_proj.", target_patterns=".eh_proj."),
        WeightRenaming(source_patterns=".mtp_block.post_norm.", target_patterns=".post_norm."),
        # Inkling checkpoint layout: per-depth extras sit at `layers.{k}.` while the decoder block
        # is nested under `layers.{k}.transformer_block.`; the main-model conversions (applied after)
        # rename the block internals
        WeightRenaming(source_patterns=r"\.hidden_norm\.", target_patterns=r".hnorm."),
        WeightRenaming(source_patterns=r"\.embed_norm\.", target_patterns=r".enorm."),
        WeightRenaming(source_patterns=r"\.input_proj\.", target_patterns=r".eh_proj."),
        WeightRenaming(source_patterns=r"^chain_norm\.", target_patterns=r"shared_post_norm."),
        WeightRenaming(source_patterns=r"layers\.(\d+)\.transformer_block\.", target_patterns=r"layers.\1.mtp_block."),
        # Mellum checkpoint layout: single MTP layer with `fc`, `pre_fc_norm_*` and `norm` stored one level up.
        WeightRenaming(
            source_patterns=r"^layers\.(\d+)\.(?!(?:mtp_block|enorm|hnorm|eh_proj|post_norm)\.)",
            target_patterns=r"layers.\1.mtp_block.",
        ),
        WeightRenaming(source_patterns=r"^fc\.", target_patterns="layers.0.eh_proj."),
        WeightRenaming(source_patterns=r"^pre_fc_norm_embedding\.", target_patterns="layers.0.enorm."),
        WeightRenaming(source_patterns=r"^pre_fc_norm_hidden\.", target_patterns="layers.0.hnorm."),
        WeightRenaming(source_patterns=r"^norm\.", target_patterns="layers.0.post_norm."),
    ]

    mapping["kimi_linear"] = [
        # Forget gate weights are attached to the forget gate module instead of the attention
        WeightRenaming(source_patterns=r"self_attn\.f_a_proj\.", target_patterns=r"self_attn.forget_gate.f_a_proj."),
        WeightRenaming(source_patterns=r"self_attn\.f_b_proj\.", target_patterns=r"self_attn.forget_gate.f_b_proj."),
        WeightRenaming(source_patterns=r"self_attn\.dt_bias", target_patterns=r"self_attn.forget_gate.dt_bias"),
        WeightRenaming(source_patterns=r"self_attn\.A_log", target_patterns=r"self_attn.forget_gate.A_log"),
        # Conv weights are stacked before runtime
        WeightConverter(
            source_patterns=[
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
            ],
            target_patterns="self_attn.conv1d.weight",
            operations=[Concatenate(dim=0)],
        ),
        # Rename MoEs so they have the same prefix as the MLPs
        WeightRenaming(source_patterns=r"\.block_sparse_moe\.", target_patterns=r"\.mlp\."),
        # Concatenate w1 (gate) and w3 (up) weights into a single weight and merge across experts
        WeightConverter(
            source_patterns=[
                r"\.experts.*.w1.weight",
                r"\.experts.*.w3.weight",
            ],
            target_patterns=r"\.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        # Merge w2 (down) weights across experts
        WeightConverter(
            source_patterns=r"\.experts.*.w2.weight",
            target_patterns=r"\.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ]

    # Gradium's runtime checkpoint (`model.safetensors` next to the runtime's files). Its codec (`mimi.`) and its
    # training-only tensors are left unexpected: the codec is the processor's audio tokenizer.
    conditioners = r"^flow_lm\.condition_provider\.conditioners\."
    flow_layers = r"^flow_lm\.transformer\.layers\.(\d+)\."
    flow_net = r"^flow_lm\.flow_net\."
    mapping["gradium_phonon_tts"] = [
        WeightRenaming(conditioners + r"transcript_in_segment\.embed\.", "model.embed_tokens."),
        WeightRenaming(conditioners + r"speaker_wavs\.output_proj\.", "model.audio_prompt_proj."),
        WeightRenaming(conditioners + r"num_speakers\.embed\.", "model.num_speakers_embed."),
        WeightRenaming(conditioners + r"num_speakers\.output_proj\.", "model.num_speakers_proj."),
        WeightRenaming(conditioners + r"duration_delta\.output_proj\.", "model.duration_delta_proj."),
        WeightRenaming(r"^flow_lm\.input_linear\.", "model.audio_latent_proj."),
        WeightRenaming(r"^flow_lm\.bos_emb", "model.bos_latent"),
        WeightRenaming(r"^flow_lm\.emb_mean", "latent_mean"),
        WeightRenaming(r"^flow_lm\.emb_std", "latent_std"),
        WeightRenaming(r"^flow_lm\.out_norm\.", "model.norm."),
        WeightRenaming(r"^flow_lm\.out_eos\.", "eos_head."),
        WeightRenaming(flow_layers + r"self_attn\.in_proj\.", r"model.layers.\1.self_attn.in_proj."),
        WeightRenaming(flow_layers + r"self_attn\.out_proj\.", r"model.layers.\1.self_attn.o_proj."),
        WeightRenaming(flow_layers + r"linear1\.", r"model.layers.\1.mlp.up_proj."),
        WeightRenaming(flow_layers + r"linear2\.", r"model.layers.\1.mlp.down_proj."),
        WeightRenaming(flow_layers + r"norm1\.", r"model.layers.\1.input_layernorm."),
        WeightRenaming(flow_layers + r"norm2\.", r"model.layers.\1.post_attention_layernorm."),
        WeightRenaming(flow_net + r"time_embed\.(\d+)\.freqs", r"flow_head.timestep_embeddings.\1.freqs"),
        WeightRenaming(flow_net + r"time_embed\.(\d+)\.mlp\.0\.", r"flow_head.timestep_embeddings.\1.fc1."),
        WeightRenaming(flow_net + r"time_embed\.(\d+)\.mlp\.2\.", r"flow_head.timestep_embeddings.\1.fc2."),
        WeightRenaming(
            flow_net + r"time_embed\.(\d+)\.mlp\.3\.alpha", r"flow_head.timestep_embeddings.\1.norm.weight"
        ),
        WeightRenaming(flow_net + r"cond_embed\.", "flow_head.cond_proj."),
        WeightRenaming(flow_net + r"input_proj\.", "flow_head.input_proj."),
        WeightRenaming(flow_net + r"res_blocks\.(\d+)\.in_ln\.", r"flow_head.layers.\1.norm."),
        WeightRenaming(flow_net + r"res_blocks\.(\d+)\.mlp\.0\.", r"flow_head.layers.\1.fc1."),
        WeightRenaming(flow_net + r"res_blocks\.(\d+)\.mlp\.2\.", r"flow_head.layers.\1.fc2."),
        WeightRenaming(flow_net + r"res_blocks\.(\d+)\.adaLN_modulation\.1\.", r"flow_head.layers.\1.adaln_proj."),
        WeightRenaming(flow_net + r"final_layer\.linear\.", "flow_head.final_layer.proj."),
        WeightRenaming(flow_net + r"final_layer\.adaLN_modulation\.1\.", "flow_head.final_layer.adaln_proj."),
        # The fused projection, its q/k rows permuted from interleaved RoPE pairs to the rotate-half layout (it also
        # splits the codec's, left unexpected)
        WeightConverter(
            source_patterns=r"self_attn\.in_proj\.weight",
            target_patterns=["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            operations=[Chunk(dim=0), PermuteForRope(permute_layer_names=["q_proj", "k_proj"])],
        ),
    ]

    for model_type, base_pattern in _MODEL_TO_CONVERSION_PATTERN.items():
        if model_type in mapping:
            continue
        mapping[model_type] = mapping[base_pattern].copy()

    def ungated_mlp_renames(source=r"\.mlp\.", target=".mlp."):
        return [
            WeightRenaming(source + r"fc1\.", target + "up_proj."),
            WeightRenaming(source + r"fc2\.", target + "down_proj."),
        ]

    for key in [
        "clip",
        "clip_text_model",
        "clip_vision_model",
        "CLIPTextModel",
        "CLIPVisionModel",
        "altclip",
        "altclip_vision_model",
        "AltCLIPVisionModel",
        "chinese_clip",
        "chinese_clip_vision_model",
        "ChineseCLIPVisionModel",
        "clipseg",
        "clipseg_text_model",
        "clipseg_vision_model",
        "CLIPSegTextModel",
        "CLIPSegVisionModel",
        "metaclip_2",
        "metaclip_2_text_model",
        "metaclip_2_vision_model",
        "MetaClip2TextModel",
        "MetaClip2VisionModel",
        "mlcd_vision_model",
        "siglip",
        "siglip_text_model",
        "siglip_vision_model",
        "SiglipTextModel",
        "SiglipVisionModel",
        "siglip2",
        "siglip2_text_model",
        "siglip2_vision_model",
        "Siglip2TextModel",
        "Siglip2VisionModel",
        "xclip",
        "xclip_text_model",
        "xclip_vision_model",
        "canary_decoder",
        "fun_asr_nano_encoder",
        "glm_image_vision",
        "glmasr_encoder",
        "internvl_vision",
        "jina_embeddings_v3",
        "minimax_m3_vl_vision",
        "mllama_vision_model",
        "moonshine_streaming_encoder",
        "muse_glimmer_vision",
        "muse_spark_vision",
        "nanochat",
        "nemotron3_diarization_audio",
        "neucodec",
        "paddleocr_vl_vision",
        "phi",
        "phi4_multimodal_vision",
        "qianfan_ocr_vision",
        "qwen2_vl_vision",
        "sam3_lite_text_text_model",
        "video_llama_3_vision",
        "videoprism_text_model",
        "videoprism_vision_model",
        "xcodec2",
        "kimi_k25_vision",
        "Dinov2Model",
        "Dinov2Backbone",
        "Dinov2WithRegistersModel",
        "Dinov2WithRegistersBackbone",
        "janus_vision_model",
        "blip_vision_model",
        "blip_2_vision_model",
        "instructblip_vision_model",
        "instructblipvideo_vision_model",
        "git_vision_model",
        "idefics2_vision",
        "idefics3_vision",
        "smolvlm_vision",
        "owlvit_text_model",
        "owlvit_vision_model",
        "owlv2_text_model",
        "owlv2_vision_model",
        "mimi",
        "MimiModel",
        "MimiEncoderModel",
        "MimiDecoderModel",
        "pocket_tts",
        "pocket_tts_audio_tokenizer",
        "gradium_phonon_tts",
        "dab-detr",
        "vitdet",
        "focalnet",
        "pp_ocrv5_server_rec",
        "pp_ocrv5_mobile_rec",
        "pp_ocrv6_small_rec",
        "mgp-str",
        "clvp_encoder",
        "patchtsmixer",
        "sam3_vit_model",
        "sam3_detr_encoder",
        "sam3_detr_decoder",
        "sam3_lite_text_detr_encoder",
        "sam3_lite_text_detr_decoder",
        "efficientloftr",
        "hiera",
        "vitpose_backbone",
        "vjepa2",
        "groupvit_text_model",
        "ernie4_5_vl_moe_vision",
        "minicpmv4_6_vision",
        "minicpmv4_7_vision",
        "cosmos3_edge_text",
        "cosmos3_edge_vision",
    ]:
        mapping[key] = mapping.get(key, []) + ungated_mlp_renames()
    mapping["UnlimitedOcrVisionEncoder"] = (
        mapping["UnlimitedOcrVisionEncoder"] + clip_layer_renames + ungated_mlp_renames()
    )
    for key in ["internvl_vision", "videoprism_vision_model", "videoprism_text_model"]:
        mapping[key] = mapping[key] + vit_layer_renames
    for key in [
        "AltCLIPVisionModel",
        "ChineseCLIPVisionModel",
        "CLIPTextModel",
        "CLIPVisionModel",
        "CLIPSegTextModel",
        "CLIPSegVisionModel",
        "clipseg",
        "MetaClip2TextModel",
        "MetaClip2VisionModel",
        "cosmos3_edge_vision",
        "git_vision_model",
        "groupvit",
        "groupvit_text_model",
        "groupvit_vision_model",
        "hunyuan_vl_vision",
        "janus_vision_model",
        "mlcd_vision_model",
        "minicpmv4_6_vision",
        "minicpmv4_7_vision",
        "minimax_m3_vl_vision",
        "nemotron3_diarization_audio",
        "owlv2_text_model",
        "owlv2_vision_model",
        "owlvit_text_model",
        "owlvit_vision_model",
        "paddleocr_vl_vision",
        "phi4_multimodal_vision",
        "pp_ocrv5_mobile_rec",
        "pp_ocrv5_server_rec",
        "pp_ocrv6_small_rec",
        "sam3_lite_text_text_model",
        "video_llama_3_vision",
        "xclip",
        "xclip_text_model",
        "xclip_vision_model",
    ]:
        mapping[key] = mapping.get(key, []) + clip_layer_renames
    for key in [
        "dinov3_vit",
        "eomt_dinov3",
        "radio",
        "rf_detr_dinov2",
        "sapiens2",
        "Tipsv2VisionModel",
        "Tipsv2VisionBackbone",
    ]:
        mapping[key] = mapping.get(key, []) + dino_layer_renames
    pre_norm_block_renames = [
        WeightRenaming(r"((?:layers|blocks)\.\d+)\.layer_norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"((?:layers|blocks)\.\d+)\.layer_norm2\.", r"\1.post_attention_layernorm."),
    ]
    # SAM-style models: only the vision encoder's layers; the two-way and memory-attention blocks keep `layer_norm1..4`
    for key in [
        "SiglipTextModel",
        "SiglipVisionModel",
        "Siglip2TextModel",
        "Siglip2VisionModel",
        "blip_vision_model",
        "blip_2_vision_model",
        "instructblip_vision_model",
        "instructblipvideo_vision_model",
        "idefics2_vision",
        "idefics3_vision",
        "smolvlm_vision",
        "SamVisionEncoder",
        "SamHQVisionEncoder",
        "GotOcr2VisionEncoder",
        "PPFormulaNetVisionModel",
        "SLANeXtVisionEncoder",
        "Sam2HieraDetModel",
        "sam3_vit_model",
    ]:
        mapping[key] = mapping.get(key, []) + pre_norm_block_renames
    mapping["oneformer"] = mapping.get("oneformer", []) + [
        WeightRenaming(r"(text_encoder\.transformer\.layers\.\d+)\.layer_norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"(text_encoder\.transformer\.layers\.\d+)\.layer_norm2\.", r"\1.post_attention_layernorm."),
    ]
    # only the classification head has the pooler; a class key replaces the `vjepa2` model_type rules
    mapping["VJEPA2ForVideoClassification"] = mapping.get("vjepa2", []) + [
        WeightRenaming(r"(pooler\.self_attention_layers\.\d+)\.layer_norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"(pooler\.self_attention_layers\.\d+)\.layer_norm2\.", r"\1.post_attention_layernorm."),
        WeightRenaming(
            r"pooler\.cross_attention_layer\.layer_norm1\.", "pooler.cross_attention_layer.input_layernorm."
        ),
        WeightRenaming(
            r"pooler\.cross_attention_layer\.layer_norm2\.", "pooler.cross_attention_layer.post_attention_layernorm."
        ),
    ]
    # the root also holds the Llama text decoder, whose norms already use these names
    mapping["idefics"] = mapping.get("idefics", []) + [
        WeightRenaming(r"(vision_model\.encoder\.layers\.\d+)\.layer_norm1\.", r"\1.input_layernorm."),
        WeightRenaming(r"(vision_model\.encoder\.layers\.\d+)\.layer_norm2\.", r"\1.post_attention_layernorm."),
    ]
    # scoped to the vision tower: unscoped, the reverse ops would also rename the language model's `self_attn`
    # q/k/v and `input_layernorm`/`post_attention_layernorm`
    mapping["qianfan_ocr_vision"] = mapping["qianfan_ocr_vision"] + [
        WeightRenaming(r"\.norm1\.", ".input_layernorm."),
        WeightRenaming(r"\.norm2\.", ".post_attention_layernorm."),
        WeightConverter(
            source_patterns=["attn.qkv.weight"],
            target_patterns=["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            operations=[Chunk(dim=0)],
        ),
        WeightConverter(
            source_patterns=["attn.qkv.bias"],
            target_patterns=["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            operations=[Chunk(dim=0)],
        ),
    ]
    mapping["moonshine"] = mapping.get("moonshine", []) + ungated_mlp_renames(
        r"encoder\.layers\.(\d+)\.mlp\.", r"encoder.layers.\1.mlp."
    )
    mapping["fun_asr_nano"] = mapping.get("fun_asr_nano", []) + ungated_mlp_renames(
        r"multi_modal_projector\.layers\.(\d+)\.mlp\.", r"multi_modal_projector.layers.\1.mlp."
    )
    mapping["lightglue"] = (
        mapping.get("lightglue", [])
        + ungated_mlp_renames(r"\.self_mlp\.", ".self_mlp.")
        + ungated_mlp_renames(r"\.cross_mlp\.", ".cross_mlp.")
    )
    mapping["vibevoice"] = mapping.get("vibevoice", []) + ungated_mlp_renames(r"\.timestep_proj\.", ".timestep_proj.")
    for key in ["eomt", "videomt", "radio", "rf_detr_dinov2"]:
        mapping[key] = (
            mapping.get(key, [])
            + [
                WeightRenaming(r"\.mlp\.gate_proj\.", ".swiglu_ffn.gate_proj."),
                WeightRenaming(r"\.mlp\.up_proj\.", ".swiglu_ffn.up_proj."),
                WeightRenaming(r"\.mlp\.down_proj\.", ".swiglu_ffn.down_proj."),
            ]
            + ungated_mlp_renames()
        )
    mapping["idefics"] = mapping.get("idefics", []) + ungated_mlp_renames(
        r"vision_model\.encoder\.layers\.(\d+)\.mlp\.", r"vision_model.encoder.layers.\1.mlp."
    )
    mapping["lw_detr"] = mapping["lw_detr"] + ungated_mlp_renames(
        r"decoder\.layers\.(\d+)\.mlp\.", r"decoder.layers.\1.mlp."
    )
    mapping["lw_detr_vit"] = mapping.get("lw_detr_vit", []) + ungated_mlp_renames(
        r"\.intermediate\.", ".intermediate."
    )
    mapping["swiftformer"] = mapping.get("swiftformer", []) + ungated_mlp_renames(r"\.linear\.", ".linear.")
    mapping["kosmos_2_5_text_model"] = mapping.get("kosmos_2_5_text_model", []) + ungated_mlp_renames(
        r"\.ffn\.", ".ffn."
    )
    for key in ["sam3", "sam3_lite_text"]:
        mapping[key] = mapping.get(key, []) + ungated_mlp_renames(
            r"geometry_encoder\.layers\.(\d+)\.mlp\.", r"geometry_encoder.layers.\1.mlp."
        )
    groupvit_vision_renames = []
    for attr in [r"mlp", r"mlp_channels", r"mlp_inter", r"group_projector\.\d+"]:
        groupvit_vision_renames += ungated_mlp_renames(rf"(vision_model\..*\.{attr})\.", r"\1.")
    for key in ["groupvit", "groupvit_vision_model"]:
        mapping[key] = mapping.get(key, []) + groupvit_vision_renames
    mapping["evolla"] = (
        mapping.get("evolla", [])
        + ungated_mlp_renames(r"\.adapter\.ff\.", ".adapter.ff.")
        + ungated_mlp_renames(
            r"sequence_compressor_resampler\.layers\.(\d+)\.1\.", r"sequence_compressor_resampler.layers.\1.1."
        )
    )
    mapping["llama4_vision_model"] = mapping.get("llama4_vision_model", []) + ungated_mlp_renames(
        r"model\.layers\.(\d+)\.mlp\.", r"model.layers.\1.mlp."
    )
    mapping["oneformer"] = mapping.get("oneformer", []) + ungated_mlp_renames(
        r"text_encoder\.transformer\.layers\.(\d+)\.mlp\.", r"text_encoder.transformer.layers.\1.mlp."
    )
    mimi_lookbehind = r"(?<!_transformer\.layers\.\d)(?<!_transformer\.layers\.\d\d)"
    for key in ["moshi", "kyutai_speech_to_text"]:
        mapping[key] = mapping.get(key, []) + [
            WeightRenaming(mimi_lookbehind + r"\.mlp\.fc1\.", ".mlp.gate_up_proj."),
            WeightRenaming(mimi_lookbehind + r"\.mlp\.fc2\.", ".mlp.down_proj."),
        ]
    for key in ["moonshine", "moonshine_streaming"]:
        mapping[key] = (
            mapping.get(key, [])
            + [
                WeightConverter(
                    source_patterns=rf"\.mlp\.fc1\.{name}",
                    target_patterns=f".mlp.gate_up_proj.{name}",
                    operations=[SwapHalves(dim=0)],
                )
                for name in ("weight", "bias")
            ]
            + [WeightRenaming(r"decoder\.layers\.(\d+)\.mlp\.fc2\.", r"decoder.layers.\1.mlp.down_proj.")]
        )
    mapping["florence_vision"] = mapping.get("florence_vision", []) + ungated_mlp_renames(r"\.ffn\.", ".ffn.")
    mapping["hunyuan_vl_vision"] = mapping.get("hunyuan_vl_vision", []) + [
        WeightRenaming(r"\.mlp\.dense_h_to_4h\.", ".mlp.up_proj."),
        WeightRenaming(r"\.mlp\.dense_4h_to_h\.", ".mlp.down_proj."),
    ]

    def attention_out_proj_renames(source=r"\.self_attn\.", target=".self_attn."):
        return [WeightRenaming(source + r"out_proj\.", target + "o_proj.")]

    for key in [
        "clip",
        "clip_text_model",
        "clip_vision_model",
        "CLIPTextModel",
        "CLIPVisionModel",
        "altclip",
        "altclip_vision_model",
        "AltCLIPVisionModel",
        "chinese_clip",
        "chinese_clip_vision_model",
        "ChineseCLIPVisionModel",
        "clipseg",
        "clipseg_text_model",
        "clipseg_vision_model",
        "CLIPSegTextModel",
        "CLIPSegVisionModel",
        "metaclip_2",
        "metaclip_2_text_model",
        "metaclip_2_vision_model",
        "MetaClip2TextModel",
        "MetaClip2VisionModel",
        "mlcd_vision_model",
        "minimax_m3_vl_vision",
        "step3p5_vision",
        "UnlimitedOcrVisionEncoder",
    ]:
        mapping[key] = mapping.get(key, []) + attention_out_proj_renames()
    for key in ["xclip", "xclip_text_model", "xclip_vision_model"]:
        mapping[key] = mapping.get(key, []) + attention_out_proj_renames()
    for key in ["xclip", "xclip_vision_model"]:
        mapping[key] = mapping[key] + attention_out_proj_renames(r"\.message_attn\.", ".message_attn.")
    mapping["sam3_lite_text_text_model"] = mapping["sam3_lite_text_text_model"] + attention_out_proj_renames()
    for key in ["eomt", "videomt", "aimv2", "aimv2_vision_model", "aimv2_text_model", "ovis2"]:
        mapping[key] = mapping.get(key, []) + attention_out_proj_renames(r"\.attention\.", ".attention.")
    for key in ["emu3_vqgan", "Emu3VQVAE"]:
        mapping[key] = mapping.get(key, []) + attention_out_proj_renames(r"\.", ".")
    # after their `attention.out_proj` rule, which matches the old attribute name
    for key in ["eomt", "videomt"]:
        mapping[key] = mapping[key] + dino_layer_renames

    return mapping


_checkpoint_conversion_mapping_cache = None


def get_checkpoint_conversion_mapping(model_type):
    global _checkpoint_conversion_mapping_cache
    if _checkpoint_conversion_mapping_cache is None:
        _checkpoint_conversion_mapping_cache = _build_checkpoint_conversion_mapping()
    return deepcopy(_checkpoint_conversion_mapping_cache.get(model_type))


USER_REGISTERED_MAPPINGS = set()


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
    # Keep track of what was added manually by the user
    USER_REGISTERED_MAPPINGS.add(model_type_or_class_name)


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
    from .modeling_utils import PreTrainedModel

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

    for module_name, submodule in model.named_modules():
        # Skip if it's not a submodel
        if not isinstance(submodule, PreTrainedModel):
            continue

        class_name = type(submodule).__name__
        model_type = submodule.config.model_type

        # Skip it if it's custom code and it was NOT registered by the user directly: it may have the same `model_type`/ClassName
        # as a native model inside the library, but it uses custom modeling so it should not share the conversions
        if (
            submodule.is_custom_code()
            and class_name not in USER_REGISTERED_MAPPINGS
            and model_type not in USER_REGISTERED_MAPPINGS
        ):
            continue

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
            # Scope each transform so it only matches keys under this sub-module's prefix - but we still allow to
            # arbitrary add/remove base_model_prefix to load ForXXX model from BaseModel and the opposite
            # Note that we need 2 removeprefix calls here, as only one level of nesting would not have the ending dot to module_name
            scope_prefix = module_name.removeprefix(model.base_model_prefix)
            scope_prefix = scope_prefix.removeprefix(".")
            for transform in conversions:
                transform.scope_prefix = scope_prefix
                transform.base_model_prefix = model.base_model_prefix
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

    if hf_quantizer is not None:
        weight_conversions = hf_quantizer.update_weight_conversions(weight_conversions, model)

    return weight_conversions
