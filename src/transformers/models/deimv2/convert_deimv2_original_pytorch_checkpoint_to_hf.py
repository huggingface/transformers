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
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from transformers import Deimv2Config, Deimv2ForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


MODEL_NAME_TO_HUB_REPO = {
    "deimv2_hgnetv2_n_coco": "Intellindust/DEIMv2_HGNetv2_N_COCO",
    "deimv2_hgnetv2_pico_coco": "Intellindust/DEIMv2_HGNetv2_PICO_COCO",
    "deimv2_hgnetv2_femto_coco": "Intellindust/DEIMv2_HGNetv2_FEMTO_COCO",
    "deimv2_hgnetv2_atto_coco": "Intellindust/DEIMv2_HGNetv2_ATTO_COCO",
    "deimv2_dinov3_s_coco": "Intellindust/DEIMv2_DINOv3_S_COCO",
    "deimv2_dinov3_m_coco": "Intellindust/DEIMv2_DINOv3_M_COCO",
    "deimv2_dinov3_l_coco": "Intellindust/DEIMv2_DINOv3_L_COCO",
    "deimv2_dinov3_x_coco": "Intellindust/DEIMv2_DINOv3_X_COCO",
}


def get_deimv2_config(model_name: str) -> Deimv2Config:
    repo_id = MODEL_NAME_TO_HUB_REPO[model_name]
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        orig_config = json.load(f)

    # COCO labels
    id2label = json.load(
        open(hf_hub_download("huggingface/label-files", "coco-detection-mmdet-id2label.json", repo_type="dataset"))
    )
    id2label = {int(k): v for k, v in id2label.items()}

    decoder_cfg = orig_config["DEIMTransformer"]
    if "HybridEncoder" in orig_config:
        encoder_cfg = orig_config["HybridEncoder"]
    elif "LiteEncoder" in orig_config:
        raise ValueError(
            "LiteEncoder variants (pico/femto/atto) are not yet supported. "
            "The LiteEncoder uses a different architecture (AvgPool downsampling, GAP fusion, "
            "RepNCSPELAN4 blocks) that requires a dedicated Deimv2LiteEncoder implementation. "
            "Supported variants: deimv2_hgnetv2_n_coco and DINOv3 variants."
        )
    else:
        raise ValueError(f"No encoder config found. Available keys: {list(orig_config.keys())}")

    config = Deimv2Config()
    config.num_labels = 80
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # Encoder settings
    config.encoder_hidden_dim = encoder_cfg["hidden_dim"]
    config.encoder_in_channels = encoder_cfg["in_channels"]
    config.feat_strides = encoder_cfg["feat_strides"]
    config.activation_function = encoder_cfg.get("act", "silu")
    config.depth_mult = encoder_cfg.get("depth_mult", 1.0)
    config.hidden_expansion = encoder_cfg.get("expansion", 1.0)
    config.encoder_fuse_op = encoder_cfg.get("fuse_op", "sum")
    config.encoder_ffn_dim = encoder_cfg["dim_feedforward"]
    config.encoder_attention_heads = encoder_cfg["nhead"]
    config.dropout = encoder_cfg.get("dropout", 0.0)
    config.encode_proj_layers = encoder_cfg["use_encoder_idx"]
    config.encoder_activation_function = encoder_cfg.get("enc_act", "gelu")

    # Decoder settings
    config.d_model = decoder_cfg["hidden_dim"]
    config.decoder_ffn_dim = decoder_cfg["dim_feedforward"]
    config.decoder_layers = decoder_cfg["num_layers"]
    config.num_feature_levels = decoder_cfg["num_levels"]
    config.decoder_n_points = decoder_cfg["num_points"]
    config.num_queries = decoder_cfg["num_queries"]
    config.num_denoising = decoder_cfg.get("num_denoising", 100)
    config.label_noise_ratio = decoder_cfg.get("label_noise_ratio", 0.5)
    config.box_noise_scale = decoder_cfg.get("box_noise_scale", 1.0)
    config.max_num_bins = decoder_cfg.get("reg_max", 32)
    config.reg_scale = decoder_cfg.get("reg_scale", 4.0)
    config.eval_idx = decoder_cfg.get("eval_idx", -1)
    config.layer_scale = decoder_cfg.get("layer_scale", 1)
    config.decoder_in_channels = decoder_cfg["feat_channels"]
    config.eval_size = tuple(decoder_cfg["eval_spatial_size"]) if "eval_spatial_size" in decoder_cfg else None
    config.decoder_activation_function = decoder_cfg.get("activation", "silu")

    # Backbone settings
    if "HGNetv2" in orig_config:
        backbone_cfg = orig_config["HGNetv2"]
        backbone_name = backbone_cfg.get("name", "B0")
        return_idx = backbone_cfg.get("return_idx", [2, 3])
        config.backbone_config.out_indices = [i + 1 for i in return_idx]
        config.backbone_config.use_learnable_affine_block = backbone_cfg.get("use_lab", True)

        # Set backbone sizes based on the model variant
        if backbone_name == "B0":
            config.backbone_config.hidden_sizes = [128, 256, 512, 1024]
            config.backbone_config.stem_channels = [3, 16, 16]
            config.backbone_config.stage_in_channels = [16, 64, 256, 512]
            config.backbone_config.stage_mid_channels = [16, 32, 64, 128]
            config.backbone_config.stage_out_channels = [64, 256, 512, 1024]
            config.backbone_config.stage_num_blocks = [1, 1, 2, 1]
            config.backbone_config.stage_downsample = [False, True, True, True]
            config.backbone_config.stage_light_block = [False, False, True, True]
            config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
            config.backbone_config.stage_numb_of_layers = [3, 3, 3, 3]
        elif backbone_name in ["B1", "B2"]:
            config.backbone_config.hidden_sizes = [128, 256, 512, 1024]
            config.backbone_config.stem_channels = [3, 16, 16]
            config.backbone_config.stage_in_channels = [16, 64, 256, 512]
            config.backbone_config.stage_mid_channels = [16, 32, 64, 128]
            config.backbone_config.stage_out_channels = [64, 256, 512, 1024]
            config.backbone_config.stage_num_blocks = [1, 1, 2, 1]
            config.backbone_config.stage_downsample = [False, True, True, True]
            config.backbone_config.stage_light_block = [False, False, True, True]
            config.backbone_config.stage_kernel_size = [3, 3, 5, 5]
            config.backbone_config.stage_numb_of_layers = [3, 3, 3, 3]
        else:
            raise ValueError(f"Unknown HGNetv2 variant: {backbone_name}")

        config.use_spatial_tuning_adapter = False
    elif "DINOv3STAs" in orig_config:
        raise ValueError(
            "DINOv3 backbone variants are not yet supported. "
            "The DINOv3+STA architecture requires ViT backbone key mappings and "
            "STA adapter integration that are not yet implemented in the conversion script. "
            "Supported variants: deimv2_hgnetv2_n_coco."
        )
    else:
        raise ValueError(f"Unknown backbone in config: {list(orig_config.keys())}")

    return config


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Backbone stem mappings
    r"backbone\.stem\.(stem\w+)\.conv\.weight": r"model.backbone.model.embedder.\1.convolution.weight",
    # Stem normalization
    r"backbone\.stem\.(stem\w+)\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.\1.normalization.\2",
    # Stem lab parameters
    r"backbone\.stem\.(stem\w+)\.lab\.(scale|bias)": r"model.backbone.model.embedder.\1.lab.\2",
    # Backbone stages mappings
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.lab\.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    # Conv1/Conv2 layers
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.conv\.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.lab\.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.lab.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.conv\.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.lab\.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.lab.\4",
    # Backbone stages aggregation
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.conv\.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.lab\.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.lab.\4",
    # Downsample
    r"backbone\.stages\.(\d+)\.downsample\.conv\.weight": r"model.backbone.model.encoder.stages.\1.downsample.convolution.weight",
    r"backbone\.stages\.(\d+)\.downsample\.bn\.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.downsample.normalization.\2",
    r"backbone\.stages\.(\d+)\.downsample\.lab\.(scale|bias)": r"model.backbone.model.encoder.stages.\1.downsample.lab.\2",
    # Encoder mappings
    # Input projections
    r"encoder\.input_proj\.(\d+)\.conv\.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder\.input_proj\.(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder_input_proj.\1.1.\2",
    # AIFI transformer encoder layers
    r"encoder\.encoder\.(\d+)\.layers\.0\.self_attn\.out_proj\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.self_attn.o_proj.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.linear1\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.mlp.layers.0.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.linear2\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.mlp.layers.1.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.norm1\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.self_attn_layer_norm.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.norm2\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.final_layer_norm.\2",
    # Encoder projections and convolutions
    r"encoder\.lateral_convs\.(\d+)\.conv\.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder\.lateral_convs\.(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.lateral_convs.\1.norm.\2",
    # FPN blocks - complete structure
    # Basic convolutions
    r"encoder\.fpn_blocks\.(\d+)\.cv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv4\.conv\.weight": r"model.encoder.fpn_blocks.\1.conv4.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv4.norm.\2",
    # CSP Rep1 path
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    # CSP Rep2 path
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    # FPN trailing convs
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv3.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv3.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv3.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv3.norm.\2",
    # PAN blocks - complete structure
    r"encoder\.pan_blocks\.(\d+)\.cv1\.conv\.weight": r"model.encoder.pan_blocks.\1.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv4\.conv\.weight": r"model.encoder.pan_blocks.\1.conv4.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv4.norm.\2",
    # CSP Rep1 path
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    # CSP Rep2 path
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    # PAN trailing convs
    r"encoder\.pan_blocks\.(\d+)\.cv2\.1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv3.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv3.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv3.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv3.norm.\2",
    # Downsample convolutions
    r"encoder\.downsample_convs\.(\d+)\.0\.cv(\d+)\.conv\.weight": r"model.encoder.downsample_convs.\1.conv\2.conv.weight",
    r"encoder\.downsample_convs\.(\d+)\.0\.cv(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.conv\2.norm.\3",
    # Decoder layers
    r"decoder\.input_proj\.(\d+)\.0\.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder\.input_proj\.(\d+)\.1\.(weight|bias|running_mean|running_var)": r"model.decoder_input_proj.\1.1.\2",
    r"decoder\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)": r"model.decoder.layers.\1.self_attn.o_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.sampling_offsets\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.attention_weights\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.attention_weights.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.value_proj\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.value_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.output_proj\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.output_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.num_points_scale": r"model.decoder.layers.\1.encoder_attn.num_points_scale",
    r"decoder\.decoder\.layers\.(\d+)\.norm1\.scale": r"model.decoder.layers.\1.self_attn_layer_norm.scale",
    r"decoder\.decoder\.layers\.(\d+)\.norm3\.scale": r"model.decoder.layers.\1.final_layer_norm.scale",
    r"decoder\.decoder\.layers\.(\d+)\.swish_ffn\.w12\.(weight|bias)": r"model.decoder.layers.\1.mlp.w12.\2",
    r"decoder\.decoder\.layers\.(\d+)\.swish_ffn\.w3\.(weight|bias)": r"model.decoder.layers.\1.mlp.w3.\2",
    r"decoder\.decoder\.layers\.(\d+)\.gateway\.gate\.(weight|bias)": r"model.decoder.layers.\1.gateway.gate.\2",
    r"decoder\.decoder\.layers\.(\d+)\.gateway\.norm\.scale": r"model.decoder.layers.\1.gateway.norm.scale",
    # LQE layers
    r"decoder\.decoder\.lqe_layers\.(\d+)\.reg_conf\.layers\.(\d+)\.(weight|bias)": r"model.decoder.lqe_layers.\1.reg_conf.layers.\2.\3",
    # Decoder heads and projections
    r"decoder\.dec_score_head\.(\d+)\.(weight|bias)": r"model.decoder.class_embed.\1.\2",
    r"decoder\.dec_bbox_head\.(\d+)\.layers\.(\d+)\.(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder\.pre_bbox_head\.layers\.(\d+)\.(weight|bias)": r"model.decoder.pre_bbox_head.layers.\1.\2",
    r"decoder\.query_pos_head\.layers\.(\d+)\.(weight|bias)": r"model.decoder.query_pos_head.layers.\1.\2",
    # Encoder output and score heads
    r"decoder\.enc_output\.proj\.(weight|bias)": r"model.enc_output.0.\1",
    r"decoder\.enc_output\.norm\.(weight|bias)": r"model.enc_output.1.\1",
    r"decoder\.enc_score_head\.(weight|bias)": r"model.enc_score_head.\1",
    r"decoder\.enc_bbox_head\.layers\.(\d+)\.(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
    # Denoising class embed
    r"decoder\.denoising_class_embed\.weight": r"model.denoising_class_embed.weight",
    # Decoder parameters
    r"decoder\.decoder\.up": r"model.decoder.up",
    r"decoder\.decoder\.reg_scale": r"model.decoder.reg_scale",
}


def convert_old_keys_to_new_keys(state_dict):
    # Use the mapping to rename keys
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict.keys()):
            new_key = re.sub(f"^{original_key}$", converted_key, key)
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
    return state_dict


def read_in_q_k_v(state_dict, config):
    encoder_hidden_dim = config.encoder_hidden_dim
    d_model = config.d_model

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"encoder.encoder.{i}.layers.0.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"encoder.encoder.{i}.layers.0.self_attn.in_proj_bias", None)
        if in_proj_weight is not None:
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
                :encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
                encoder_hidden_dim : 2 * encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
                -encoder_hidden_dim:
            ]
        if in_proj_bias is not None:
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
                encoder_hidden_dim : 2 * encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]

    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"decoder.decoder.layers.{i}.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"decoder.decoder.layers.{i}.self_attn.in_proj_bias", None)
        if in_proj_weight is not None:
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[d_model : 2 * d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[d_model : 2 * d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-d_model:]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-d_model:]


def load_original_state_dict(repo_id):
    filepath = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    return load_file(filepath)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_deimv2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our Deimv2 structure.
    """
    hub_repo = MODEL_NAME_TO_HUB_REPO[model_name]
    config = get_deimv2_config(model_name)
    state_dict = load_original_state_dict(hub_repo)

    logger.info(f"Converting model {model_name} from {hub_repo}...")
    logger.info(f"Original state dict has {len(state_dict)} keys")

    state_dict.pop("decoder.valid_mask", None)
    state_dict.pop("decoder.anchors", None)

    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)

    state_dict = convert_old_keys_to_new_keys(state_dict)

    if "model.enc_output.0.weight" not in state_dict:
        d_model = config.d_model
        state_dict["model.enc_output.0.weight"] = torch.eye(d_model)
        state_dict["model.enc_output.0.bias"] = torch.zeros(d_model)
        state_dict["model.enc_output.1.weight"] = torch.ones(d_model)
        state_dict["model.enc_output.1.bias"] = torch.zeros(d_model)

    # for two_stage
    for key in list(state_dict.keys()):
        if "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            new_key = key.split("model.decoder.")[-1]
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]

    model = Deimv2ForObjectDetection(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    model.eval()

    image_processor = RTDetrImageProcessor()
    img = prepare_img()

    transformations = transforms.Compose(
        [
            transforms.Resize([640, 640], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    assert torch.allclose(original_pixel_values, pixel_values), "Image preprocessing mismatch!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    outputs = model(pixel_values)
    logger.info(f"Logits shape: {outputs.logits.shape}")
    logger.info(f"Boxes shape: {outputs.pred_boxes.shape}")
    logger.info(f"Logits sample: {outputs.logits[0, :3, :3]}")
    logger.info(f"Boxes sample: {outputs.pred_boxes[0, :3, :3]}")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        logger.info(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        logger.info(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        push_repo = repo_id or f"deimv2-{model_name}"
        logger.info(f"Pushing to hub: {push_repo}")
        config.push_to_hub(repo_id=push_repo)
        model.push_to_hub(repo_id=push_repo)
        image_processor.push_to_hub(repo_id=push_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="deimv2_hgnetv2_n_coco",
        type=str,
        choices=list(MODEL_NAME_TO_HUB_REPO.keys()),
        help="Model name to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push to the hub.")
    parser.add_argument("--repo_id", type=str, default=None, help="Hub repo_id to push to.")
    args = parser.parse_args()
    convert_deimv2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
