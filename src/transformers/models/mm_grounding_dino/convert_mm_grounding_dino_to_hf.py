# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import argparse
import re

import requests
import torch
from PIL import Image

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
from transformers.models.mm_grounding_dino.configuration_mm_grounding_dino import MMGroundingDinoConfig
from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import MMGroundingDinoForObjectDetection
from transformers.models.swin.configuration_swin import SwinConfig


MODEL_NAME_TO_CHECKPOINT_URL_MAPPING = {
    "mm_grounding_dino_tiny_o365v1_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth",
    "mm_grounding_dino_base_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth",
    "mm_grounding_dino_base_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth",
    "mm_grounding_dino_large_o365v2_oiv6_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
    "mm_grounding_dino_large_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
    "llmdet_tiny": "https://huggingface.co/fushh7/LLMDet/resolve/main/tiny.pth?download=true",
    "llmdet_base": "https://huggingface.co/fushh7/LLMDet/resolve/main/base.pth?download=true",
    "llmdet_large": "https://huggingface.co/fushh7/LLMDet/resolve/main/large.pth?download=true",
}


MODEL_NAME_TO_EXPECTED_OUTPUT_MAPPING = {
    "mm_grounding_dino_tiny_o365v1_goldg": {
        "scores": torch.tensor([0.7722, 0.7584, 0.7984, 0.7163]),
        "boxes": torch.tensor(
            [
                [0.5212, 0.1594, 0.5792, 0.3895],
                [0.5424, 0.0513, 0.9996, 0.7757],
                [0.0629, 0.1526, 0.2746, 0.2447],
                [0.0091, 0.1127, 0.4945, 0.9911],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_grit": {
        "scores": torch.tensor([0.7865, 0.7180, 0.7665, 0.8177]),
        "boxes": torch.tensor(
            [
                [0.0084, 0.1129, 0.4940, 0.9895],
                [0.5214, 0.1597, 0.5786, 0.3875],
                [0.5413, 0.0507, 0.9998, 0.7768],
                [0.0631, 0.1527, 0.2740, 0.2449],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_v3det": {
        "scores": torch.tensor([0.5690, 0.5553, 0.6075, 0.5775]),
        "boxes": torch.tensor(
            [
                [0.5393, 0.0502, 0.9989, 0.7763],
                [0.0090, 0.1125, 0.4950, 0.9895],
                [0.5207, 0.1589, 0.5794, 0.3889],
                [0.0625, 0.1519, 0.2750, 0.2446],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_grit_v3det": {
        "scores": torch.tensor([0.8381, 0.8204, 0.7970, 0.7175]),
        "boxes": torch.tensor(
            [
                [0.0099, 0.1129, 0.4942, 0.9903],
                [0.5413, 0.0506, 0.9998, 0.7753],
                [0.0626, 0.1527, 0.2744, 0.2443],
                [0.5211, 0.1596, 0.5790, 0.3890],
            ]
        ),
    },
    "mm_grounding_dino_base_o365v1_goldg_v3det": {
        "scores": torch.tensor([0.8418, 0.8364, 0.8342, 0.7885]),
        "boxes": torch.tensor(
            [
                [0.5427, 0.0502, 0.9996, 0.7770],
                [0.0628, 0.1529, 0.2747, 0.2448],
                [0.0085, 0.1132, 0.4947, 0.9898],
                [0.5208, 0.1597, 0.5787, 0.3910],
            ]
        ),
    },
    "mm_grounding_dino_base_all": {
        "scores": torch.tensor([0.4713]),
        "boxes": torch.tensor([[0.5423, 0.0507, 0.9998, 0.7761]]),
    },
    "mm_grounding_dino_large_o365v2_oiv6_goldg": {
        "scores": torch.tensor([0.7824, 0.8275, 0.7715, 0.8211]),
        "boxes": torch.tensor(
            [
                [0.0082, 0.1133, 0.4945, 0.9889],
                [0.5410, 0.0508, 0.9998, 0.7771],
                [0.0632, 0.1526, 0.2740, 0.2439],
                [0.5205, 0.1599, 0.5787, 0.3906],
            ]
        ),
    },
    "mm_grounding_dino_large_all": {
        "scores": torch.tensor([0.7373, 0.6208, 0.6913, 0.4523]),
        "boxes": torch.tensor(
            [
                [0.5424, 0.0509, 0.9997, 0.7765],
                [0.0632, 0.1529, 0.2744, 0.2447],
                [0.0121, 0.1125, 0.4947, 0.9884],
                [0.5206, 0.1597, 0.5789, 0.3933],
            ]
        ),
    },
    "llmdet_tiny": {
        "scores": torch.tensor([0.7262, 0.7552, 0.7656, 0.8207]),
        "boxes": torch.tensor(
            [
                [0.0114, 0.1132, 0.4947, 0.9854],
                [0.5387, 0.0513, 0.9992, 0.7765],
                [0.5212, 0.1605, 0.5788, 0.3890],
                [0.0634, 0.1536, 0.2743, 0.2440],
            ]
        ),
    },
    "llmdet_base": {
        "scores": torch.tensor([0.8646, 0.7567, 0.6978, 0.8084]),
        "boxes": torch.tensor(
            [
                [0.0632, 0.1529, 0.2745, 0.2438],
                [0.5420, 0.0512, 0.9989, 0.7774],
                [0.0110, 0.1134, 0.4950, 0.9875],
                [0.5209, 0.1602, 0.5789, 0.3908],
            ]
        ),
    },
    "llmdet_large": {
        "scores": torch.tensor([0.7107, 0.8626, 0.7458, 0.8166]),
        "boxes": torch.tensor(
            [
                [0.0147, 0.1128, 0.4957, 0.9858],
                [0.0634, 0.1528, 0.2744, 0.2447],
                [0.5414, 0.0511, 0.9997, 0.7776],
                [0.5209, 0.1602, 0.5792, 0.3916],
            ]
        ),
    },
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # vision backbone
    r"backbone.patch_embed.projection.(weight|bias)":                                                               r"model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.\1",
    r"backbone.patch_embed.norm.(weight|bias)":                                                                     r"model.backbone.conv_encoder.model.embeddings.norm.\1",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.(relative_position_bias_table|relative_position_index)":        r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.self.\3",
    r"backbone.stages.(\d+).blocks.(\d+).norm1.(weight|bias)":                                                      r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.layernorm_before.\3",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.(query|key|value).(weight|bias)":                               r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.self.\3.\4",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.proj.(weight|bias)":                                            r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.output.dense.\3",
    r"backbone.stages.(\d+).blocks.(\d+).norm2.(weight|bias)":                                                      r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.layernorm_after.\3",
    r"backbone.stages.(\d+).blocks.(\d+).ffn.layers.0.0.(weight|bias)":                                             r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.intermediate.dense.\3",
    r"backbone.stages.(\d+).blocks.(\d+).ffn.layers.1.(weight|bias)":                                               r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.output.dense.\3",
    r"backbone.stages.(\d+).downsample.reduction.weight":                                                           r"model.backbone.conv_encoder.model.encoder.layers.\1.downsample.reduction.weight",
    r"backbone.stages.(\d+).downsample.norm.(weight|bias)":                                                         r"model.backbone.conv_encoder.model.encoder.layers.\1.downsample.norm.\2",
    r"backbone.norms.(\d+).(weight|bias)":                                                                            r"model.backbone.conv_encoder.model.hidden_states_norms.stage\1.\2",
    r"neck.convs.(\d+).conv.(weight|bias)":                                                                         r"model.input_proj_vision.\1.0.\2",
    r"neck.convs.(\d+).gn.(weight|bias)":                                                                           r"model.input_proj_vision.\1.1.\2",
    r"neck.extra_convs.(\d+).conv.(weight|bias)":                                                                   r"model.input_proj_vision.\1.0.\2",
    r"neck.extra_convs.(\d+).gn.(weight|bias)":                                                                     r"model.input_proj_vision.\1.1.\2",
    # text backbone
    r"language_model.language_backbone.body.model.(.*)":                                                            r"model.text_backbone.\1",
    r"text_feat_map.(weight|bias)":                                                                                 r"model.text_projection.\1",
    # encoder
    r"encoder.fusion_layers.(\d+).gamma_v":                                                                         r"model.encoder.layers.\1.fusion_layer.vision_param",
    r"encoder.fusion_layers.(\d+).gamma_l":                                                                         r"model.encoder.layers.\1.fusion_layer.text_param",
    r"encoder.fusion_layers.(\d+).layer_norm_v.(weight|bias)":                                                      r"model.encoder.layers.\1.fusion_layer.layer_norm_vision.\2",
    r"encoder.fusion_layers.(\d+).attn.v_proj.(weight|bias)":                                                       r"model.encoder.layers.\1.fusion_layer.attn.vision_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.values_v_proj.(weight|bias)":                                                r"model.encoder.layers.\1.fusion_layer.attn.values_vision_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.out_v_proj.(weight|bias)":                                                   r"model.encoder.layers.\1.fusion_layer.attn.out_vision_proj.\2",
    r"encoder.fusion_layers.(\d+).layer_norm_l.(weight|bias)":                                                      r"model.encoder.layers.\1.fusion_layer.layer_norm_text.\2",
    r"encoder.fusion_layers.(\d+).attn.l_proj.(weight|bias)":                                                       r"model.encoder.layers.\1.fusion_layer.attn.text_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.values_l_proj.(weight|bias)":                                                r"model.encoder.layers.\1.fusion_layer.attn.values_text_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.out_l_proj.(weight|bias)":                                                   r"model.encoder.layers.\1.fusion_layer.attn.out_text_proj.\2",
    r"encoder.layers.(\d+).self_attn.(sampling_offsets|attention_weights|value_proj|output_proj).(weight|bias)":    r"model.encoder.layers.\1.deformable_layer.self_attn.\2.\3",
    r"encoder.layers.(\d+).norms.0.(weight|bias)":                                                                  r"model.encoder.layers.\1.deformable_layer.self_attn_layer_norm.\2",
    r"encoder.layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                           r"model.encoder.layers.\1.deformable_layer.fc1.\2",
    r"encoder.layers.(\d+).ffn.layers.1.(weight|bias)":                                                             r"model.encoder.layers.\1.deformable_layer.fc2.\2",
    r"encoder.layers.(\d+).norms.1.(weight|bias)":                                                                  r"model.encoder.layers.\1.deformable_layer.final_layer_norm.\2",
    r"encoder.text_layers.(\d+).self_attn.attn.(query|key|value)_proj_(weight|bias)":                               r"model.encoder.layers.\1.text_enhancer_layer.self_attn.\2.\3",
    r"encoder.text_layers.(\d+).self_attn.attn.out_proj.(weight|bias)":                                             r"model.encoder.layers.\1.text_enhancer_layer.self_attn.out_proj.\2",
    r"encoder.text_layers.(\d+).norms.0.(weight|bias)":                                                             r"model.encoder.layers.\1.text_enhancer_layer.layer_norm_before.\2",
    r"encoder.text_layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                      r"model.encoder.layers.\1.text_enhancer_layer.fc1.\2",
    r"encoder.text_layers.(\d+).ffn.layers.1.(weight|bias)":                                                        r"model.encoder.layers.\1.text_enhancer_layer.fc2.\2",
    r"encoder.text_layers.(\d+).norms.1.(weight|bias)":                                                             r"model.encoder.layers.\1.text_enhancer_layer.layer_norm_after.\2",
    r"encoder.bbox_head.cls_branch.bias":                                                                           r"model.encoder_output_class_embed.bias",
    r"encoder.bbox_head.reg_branch.0.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.0.\1",
    r"encoder.bbox_head.reg_branch.2.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.1.\1",
    r"encoder.bbox_head.reg_branch.4.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.2.\1",
    # decoder
    r"decoder.norm.(weight|bias)":                                                                                  r"model.decoder.layer_norm.\1",
    r"decoder.ref_point_head.layers.(\d+).(weight|bias)":                                                           r"model.decoder.reference_points_head.layers.\1.\2",
    r"decoder.layers.(\d+).self_attn.attn.(query|key|value)_proj_(weight|bias)":                                    r"model.decoder.layers.\1.self_attn.\2.\3",
    r"decoder.layers.(\d+).self_attn.attn.out_proj.(weight|bias)":                                                  r"model.decoder.layers.\1.self_attn.out_proj.\2",
    r"decoder.layers.(\d+).norms.0.(weight|bias)":                                                                  r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"decoder.layers.(\d+).cross_attn_text.attn.(query|key|value)_proj_(weight|bias)":                              r"model.decoder.layers.\1.encoder_attn_text.\2.\3",
    r"decoder.layers.(\d+).cross_attn_text.attn.out_proj.(weight|bias)":                                            r"model.decoder.layers.\1.encoder_attn_text.out_proj.\2",
    r"decoder.layers.(\d+).norms.1.(weight|bias)":                                                                  r"model.decoder.layers.\1.encoder_attn_text_layer_norm.\2",
    r"decoder.layers.(\d+).cross_attn.(sampling_offsets|attention_weights|value_proj|output_proj).(weight|bias)":   r"model.decoder.layers.\1.encoder_attn.\2.\3",
    r"decoder.layers.(\d+).norms.2.(weight|bias)":                                                                  r"model.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"decoder.layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                           r"model.decoder.layers.\1.fc1.\2",
    r"decoder.layers.(\d+).ffn.layers.1.(weight|bias)":                                                             r"model.decoder.layers.\1.fc2.\2",
    r"decoder.layers.(\d+).norms.3.(weight|bias)":                                                                  r"model.decoder.layers.\1.final_layer_norm.\2",
    r"decoder.bbox_head.cls_branches.(\d+).bias":                                                                   r"model.decoder.class_embed.\1.bias",
    r"decoder.bbox_head.reg_branches.(\d+).0.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.0.\2",
    r"decoder.bbox_head.reg_branches.(\d+).2.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.1.\2",
    r"decoder.bbox_head.reg_branches.(\d+).4.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.2.\2",
    # other
    r"level_embed":                                                                                                 r"model.level_embed",
    r"query_embedding.weight":                                                                                      r"model.query_position_embeddings.weight",
    r"memory_trans_fc.(weight|bias)":                                                                               r"model.enc_output.\1",
    r"memory_trans_norm.(weight|bias)":                                                                             r"model.enc_output_norm.\1",
    r"bbox_head.cls_branches.(\d+).bias":                                                                           r"class_embed.\1.bias",
    r"bbox_head.reg_branches.(\d+).0.(weight|bias)":                                                                r"bbox_embed.\1.layers.0.\2",
    r"bbox_head.reg_branches.(\d+).2.(weight|bias)":                                                                r"bbox_embed.\1.layers.1.\2",
    r"bbox_head.reg_branches.(\d+).4.(weight|bias)":                                                                r"bbox_embed.\1.layers.2.\2",
}
# fmt: on


def get_mm_grounding_dino_config(model_name: str) -> MMGroundingDinoConfig:
    if "tiny" in model_name:
        swin_image_size = 224
        swin_window_size = 7
        swin_embed_dim = 96
        swin_depths = (2, 2, 6, 2)
        swin_num_heads = (3, 6, 12, 24)
        swin_out_features = ["stage2", "stage3", "stage4"]
        num_feature_levels = 4
    elif "base" in model_name:
        swin_image_size = 384
        swin_window_size = 12
        swin_embed_dim = 128
        swin_depths = (2, 2, 18, 2)
        swin_num_heads = (4, 8, 16, 32)
        swin_out_features = ["stage2", "stage3", "stage4"]
        num_feature_levels = 4
    elif "large" in model_name:
        swin_image_size = 384
        swin_window_size = 12
        swin_embed_dim = 192
        swin_depths = (2, 2, 18, 2)
        swin_num_heads = (6, 12, 24, 48)
        swin_out_features = ["stage1", "stage2", "stage3", "stage4"]
        num_feature_levels = 5
    else:
        raise ValueError(
            f"Model name: {model_name} is not supported. Only `tiny`, `base` and `large` models are currently supported."
        )

    backbone_config = SwinConfig(
        image_size=swin_image_size,
        window_size=swin_window_size,
        embed_dim=swin_embed_dim,
        depths=swin_depths,
        num_heads=swin_num_heads,
        out_features=swin_out_features,
    )

    model_config = MMGroundingDinoConfig(
        backbone_config=backbone_config,
        num_feature_levels=num_feature_levels,
    )

    return model_config


def get_mm_grounding_dino_processor() -> GroundingDinoProcessor:
    img_processor = GroundingDinoImageProcessor()
    txt_processor = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = GroundingDinoProcessor(img_processor, txt_processor)
    return processor


# Copied from: https://github.com/iSEE-Laboratory/LLMDet/blob/96ec8c82a9d97b170db759e043afd5b81445d0f1/hf_model/mmdet2groundingdino_swint.py#L8C1-L13C13
def correct_unfold_reduction_order(x: torch.Tensor) -> torch.Tensor:
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4).transpose(1, 2)
    x = x[:, [0, 2, 1, 3], :]
    x = x.reshape(out_channel, in_channel)
    return x


# Copied from: https://github.com/iSEE-Laboratory/LLMDet/blob/96ec8c82a9d97b170db759e043afd5b81445d0f1/hf_model/mmdet2groundingdino_swint.py#L15C1-L20C13
def correct_unfold_norm_order(x: torch.Tensor) -> torch.Tensor:
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4).transpose(0, 1)
    x = x[[0, 2, 1, 3], :]
    x = x.reshape(in_channel)
    return x


def preprocess_old_state(state_dict: dict, config: MMGroundingDinoConfig) -> dict:
    """
    Preprocesses old state dict to enable 1-1 mapping:
        - split qkv projections in Swin backbone
        - reorder reduction and norm parameters in Swin backbone
        - shift output norm indices in Swin backbone
        - shift output proj indices in neck
        - split q,k,v projections in text self and cross attentions in encoder and decoder
        - duplicate detection head parameters for decoder and encoder
    """
    new_state_dict = state_dict.copy()
    for k in state_dict:
        if k.startswith("backbone"):
            if "downsample.reduction" in k:
                new_state_dict[k] = correct_unfold_reduction_order(new_state_dict.pop(k))
            elif "downsample.norm" in k:
                new_state_dict[k] = correct_unfold_norm_order(new_state_dict.pop(k))
            elif "w_msa.qkv" in k:
                q_param, k_param, v_param = new_state_dict.pop(k).chunk(3)
                new_state_dict[k.replace("qkv", "query")] = q_param
                new_state_dict[k.replace("qkv", "key")] = k_param
                new_state_dict[k.replace("qkv", "value")] = v_param
            elif "backbone.norm" in k:
                match = re.match(r"backbone.norm(\d+).(weight|bias)", k)
                new_state_dict[f"backbone.norms.{int(match.group(1)) + 1}.{match.group(2)}"] = new_state_dict.pop(k)
        elif k.startswith("neck.extra_convs"):
            num_normal_convs = len(config.backbone_config.out_indices)
            if "gn" in k:
                match = re.match(r"neck.extra_convs.(\d+).gn.(weight|bias)", k)
                new_state_dict[f"neck.extra_convs.{num_normal_convs + int(match.group(1))}.gn.{match.group(2)}"] = (
                    new_state_dict.pop(k)
                )
            elif "conv" in k:
                match = re.match(r"neck.extra_convs.(\d+).conv.(weight|bias)", k)
                new_state_dict[f"neck.extra_convs.{num_normal_convs + int(match.group(1))}.conv.{match.group(2)}"] = (
                    new_state_dict.pop(k)
                )
        elif k.startswith("encoder"):
            if "self_attn.attn.in_proj" in k:
                q_param, k_param, v_param = new_state_dict.pop(k).chunk(3)
                new_state_dict[k.replace("in", "query")] = q_param
                new_state_dict[k.replace("in", "key")] = k_param
                new_state_dict[k.replace("in", "value")] = v_param
        elif k.startswith("decoder"):
            if "self_attn.attn.in_proj" in k or "cross_attn_text.attn.in_proj" in k:
                q_param, k_param, v_param = new_state_dict.pop(k).chunk(3)
                new_state_dict[k.replace("in", "query")] = q_param
                new_state_dict[k.replace("in", "key")] = k_param
                new_state_dict[k.replace("in", "value")] = v_param
        elif k.startswith("bbox_head"):
            num_decoder_layers = config.decoder_layers
            match = re.match(r"bbox_head.(cls|reg)_branches.(\d+).(.*)", k)
            cls_or_reg = match.group(1)
            layer_idx = int(match.group(2))
            suffix = match.group(3)
            if layer_idx < num_decoder_layers:
                new_key = f"decoder.bbox_head.{cls_or_reg}_branches.{layer_idx}.{suffix}"
                new_state_dict[new_key] = new_state_dict[k]  # copy
            else:
                new_key = f"encoder.bbox_head.{cls_or_reg}_branch.{suffix}"
                new_state_dict[new_key] = new_state_dict.pop(k)  # move

        # remove unused params
        if (
            k == "dn_query_generator.label_embedding.weight"
            or k == "language_model.language_backbone.body.model.embeddings.position_ids"
            or k == "image_seperate.weight"
            or k.startswith("lmm")
            or k.startswith("connector")
            or k.startswith("region_connector")
            or k.startswith("ref_point_head")
        ):
            new_state_dict.pop(k)

    return new_state_dict


# Copied from transformers/models/siglip2/convert_siglip2_to_hf.py
def convert_old_keys_to_new_keys(state_dict_keys: list) -> dict:
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def convert_mm_to_hf_state(original_state: dict, hf_cfg: MMGroundingDinoConfig) -> dict:
    original_state = preprocess_old_state(original_state, hf_cfg)
    original_state_keys = list(original_state.keys())
    original_to_hf_key_map = convert_old_keys_to_new_keys(original_state_keys)

    hf_state = {}
    for original_key in original_state_keys:
        hf_key = original_to_hf_key_map[original_key]
        hf_state[hf_key] = original_state.pop(original_key)

    return hf_state


def prepare_test_inputs():
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    text = [["cat", "remote"]]
    return image, text


@torch.no_grad()
def convert_mm_grounding_dino_checkpoint(
    model_name: str,
    verify_outputs: bool,
    push_to_hub: bool,
    hub_user_name: str,
) -> tuple[MMGroundingDinoConfig, dict]:
    # Load original state
    checkpoint_url = MODEL_NAME_TO_CHECKPOINT_URL_MAPPING[model_name]
    print(f"Loading checkpoint from: {checkpoint_url}")
    ckpt = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    mm_state = ckpt["state_dict"]

    # Create hf model and processor
    print("Creating model...")
    hf_cfg = get_mm_grounding_dino_config(model_name)
    hf_state = convert_mm_to_hf_state(mm_state, hf_cfg)
    hf_model = MMGroundingDinoForObjectDetection(hf_cfg).eval()
    hf_model.load_state_dict(hf_state)
    hf_processor = get_mm_grounding_dino_processor()

    # Verify outputs if needed
    if verify_outputs:
        print("Running inference to verify outputs...")
        image, text = prepare_test_inputs()
        model_inputs = hf_processor(images=image, text=text, return_tensors="pt")
        model_outputs = hf_model(**model_inputs)
        results = hf_processor.post_process_grounded_object_detection(
            model_outputs,
            model_inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
        )
        result = results[0]
        print(result)
        expected = MODEL_NAME_TO_EXPECTED_OUTPUT_MAPPING[model_name]
        for key in expected:
            torch.testing.assert_close(result[key], expected[key], atol=1e-3, rtol=1e-3)
        print("Outputs match.")

    # Push to hub if needed
    if push_to_hub:
        print("Pushing to hub...")
        hub_url = f"{hub_user_name}/{model_name}"
        hf_model.push_to_hub(hub_url)
        hf_processor.push_to_hub(hub_url)
        print(f"Pushed to huggingface hub at: {hub_url}.")

    return hf_cfg, hf_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        choices=list(MODEL_NAME_TO_CHECKPOINT_URL_MAPPING.keys()),
        help="URL to the original mm grounding dino checkpoint.",
    )
    parser.add_argument("--hub-user-name", type=str, help="User name on the huggingface hub.")
    parser.add_argument("--push-to-hub", action="store_true", help="Whether to push model to hub or not.")
    parser.add_argument(
        "--verify-outputs", action="store_true", help="Whether to verify that model output is correct or not."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_mm_grounding_dino_checkpoint(
        args.model_name,
        args.verify_outputs,
        args.push_to_hub,
        args.hub_user_name,
    )
