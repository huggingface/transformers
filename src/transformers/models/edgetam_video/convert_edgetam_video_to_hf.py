# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert SAM checkpoints from the original repository.

URL: https://github.com/facebookresearch/segment-anything-2.
"""

import argparse
import re
from io import BytesIO

import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    EdgeTamVideoConfig,
    EdgeTamVideoMaskDecoderConfig,
    EdgeTamVideoModel,
    EdgeTamVideoPromptEncoderConfig,
    EdgeTamVisionConfig,
    Sam2ImageProcessorFast,
    Sam2VideoProcessor,
    Sam2VideoVideoProcessor,
    TimmWrapperConfig,
)


def get_config(model_name):
    backbone_config = TimmWrapperConfig.from_pretrained(
        "timm/repvit_m1.dist_in1k",
        model_args={"in_chans": 3, "features_only": True, "out_indices": (0, 1, 2, 3)},
    )
    vision_config = EdgeTamVisionConfig(backbone_config=backbone_config)

    prompt_encoder_config = EdgeTamVideoPromptEncoderConfig()
    mask_decoder_config = EdgeTamVideoMaskDecoderConfig()
    enable_temporal_pos_encoding_for_object_pointers = False
    enable_occlusion_spatial_embedding = False

    config = EdgeTamVideoConfig(
        vision_config=vision_config,
        prompt_encoder_config=prompt_encoder_config,
        mask_decoder_config=mask_decoder_config,
        enable_temporal_pos_encoding_for_object_pointers=enable_temporal_pos_encoding_for_object_pointers,
        enable_occlusion_spatial_embedding=enable_occlusion_spatial_embedding,
    )

    return config


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
    "body.": "timm_model.",
    "ff.0": "mlp.layer_norm",
    "ff.1": "mlp.up_proj",
    "ff.3": "mlp.down_proj",
}


def replace_keys(state_dict):
    model_state_dict = {}
    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"
    output_mask_decoder_mlps_pattern = r"mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*"
    output_mask_decoder_score_head_pattern = r"mask_decoder.pred_obj_score_head.layers.(\d+).*"
    output_vision_encoder_mlps_pattern = r"vision_encoder.backbone.blocks.(\d+).mlp.layers.(\d+).*"
    output_vision_encoder_neck_pattern = r"vision_encoder.neck.convs.(\d+).conv"
    output_memory_encoder_projection_pattern = r"memory_encoder.o_proj.*"
    memory_attention_pattern = r"memory_attention.*"
    output_object_pointer_proj_pattern = r"object_pointer_proj.layers.(\d+).*"
    output_memory_encoder_mask_downsampler_pattern = r"memory_encoder.mask_downsampler.encoder.(\d+).*"
    perceiver_resampler_patterns = {
        r"spatial_perceiver.latents": r"spatial_perceiver.latents_1d",
        r"spatial_perceiver.latents_1d_2d": r"spatial_perceiver.latents_2d",
        r"spatial_perceiver.layers.(\d+).attn.layer_norm_x": r"spatial_perceiver.layers.\1.layer_norm_input",
        r"spatial_perceiver.layers.(\d+).attn.layer_norm_latents": r"spatial_perceiver.layers.\1.layer_norm_latents",
        r"spatial_perceiver.layers.(\d+).self_attn.layer_norm": r"spatial_perceiver.layers.\1.layer_norm_self",
        r"spatial_perceiver.layers.(\d+).attn.to_q": r"spatial_perceiver.layers.\1.cross_attention.q_proj",
        r"spatial_perceiver.layers.(\d+).attn.to_kv": r"spatial_perceiver.layers.\1.cross_attention.kv_proj_combined",
        r"spatial_perceiver.layers.(\d+).attn.to_out": r"spatial_perceiver.layers.\1.cross_attention.o_proj",
        r"spatial_perceiver.layers.(\d+).self_attn.to_q": r"spatial_perceiver.layers.\1.self_attention.q_proj",
        r"spatial_perceiver.layers.(\d+).self_attn.to_kv": r"spatial_perceiver.layers.\1.self_attention.kv_proj_combined",
        r"spatial_perceiver.layers.(\d+).self_attn.to_out": r"spatial_perceiver.layers.\1.self_attention.o_proj",
        r"spatial_perceiver.layers.(\d+).attn": r"spatial_perceiver.layers.\1.cross_attention",
        r"spatial_perceiver.layers.(\d+).self_attn": r"spatial_perceiver.layers.\1.self_attention",
    }

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        for pattern, replacement in perceiver_resampler_patterns.items():
            if re.match(pattern, key):
                key = re.sub(pattern, replacement, key)

        # vision_encoder.blocks.0.mlp.layers.1.weight -> vision_encoder.blocks.0.mlp.proj_out.weight
        if re.match(output_vision_encoder_mlps_pattern, key):
            layer_nb = int(re.match(output_vision_encoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "proj_out")

        if re.match(memory_attention_pattern, key):
            key = key.replace("linear1", "mlp.up_proj")
            key = key.replace("linear2", "mlp.down_proj")

        # mask_decoder.transformer.layers.0.mlp.layers.1.weight -> mask_decoder.transformer.layers.1.mlp.proj_out.weight
        if re.match(output_mask_decoder_mlps_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("mlp.layers.0", "mlp.proj_in")
            elif layer_nb == 1:
                key = key.replace("mlp.layers.1", "mlp.proj_out")

        # mask_decoder.pred_obj_score_head.layers.1.weight -> mask_decoder.pred_obj_score_head.proj_in.weight
        if re.match(output_mask_decoder_score_head_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_score_head_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        # vision_encoder.neck.convs.1.conv.bias -> vision_encoder.neck.convs.1.bias
        if re.match(output_vision_encoder_neck_pattern, key):
            key = key.replace(".conv.", ".")

        # memory_encoder.o_proj.weight -> memory_encoder.projection.weight
        if re.match(output_memory_encoder_projection_pattern, key):
            key = key.replace(".o_proj.", ".projection.")

        if re.match(output_object_pointer_proj_pattern, key):
            layer_nb = int(re.match(output_object_pointer_proj_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

                key = key.replace("layers.2", "proj_out")

        if re.match(output_memory_encoder_mask_downsampler_pattern, key):
            layer_nb = int(re.match(output_memory_encoder_mask_downsampler_pattern, key).group(1))
            if layer_nb == 12:
                key = key.replace(f"encoder.{layer_nb}", "final_conv")
            elif layer_nb % 3 == 0:
                key = key.replace(f"encoder.{layer_nb}", f"layers.{layer_nb // 3}.conv")
            elif layer_nb % 3 == 1:
                key = key.replace(f"encoder.{layer_nb}", f"layers.{layer_nb // 3}.layer_norm")
        if "kv_proj_combined" in key:
            # Split the weight tensor in half along dimension 0 (output dimension)
            k_weight, v_weight = torch.chunk(value, 2, dim=0)
            # Create the k_proj and v_proj keys
            k_key = key.replace("kv_proj_combined", "k_proj")
            v_key = key.replace("kv_proj_combined", "v_proj")
            model_state_dict[k_key] = k_weight
            model_state_dict[v_key] = v_weight
            continue

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]
    model_state_dict["prompt_encoder.point_embed.weight"] = torch.cat(
        [model_state_dict.pop(f"prompt_encoder.point_embed.{i}.weight") for i in range(4)],
        dim=0,
    )

    return model_state_dict


def convert_edgetam_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub, run_sanity_check):
    config = get_config(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    state_dict = replace_keys(state_dict)

    image_processor = Sam2ImageProcessorFast()
    video_processor = Sam2VideoVideoProcessor()
    processor = Sam2VideoProcessor(image_processor=image_processor, video_processor=video_processor)
    hf_model = EdgeTamVideoModel(config)
    hf_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=True)
    hf_model = hf_model.to(device)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    if run_sanity_check:
        url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        with httpx.stream("GET", url) as response:
            raw_image = Image.open(BytesIO(response.read())).convert("RGB")

        input_points = [[[[1000, 600]]]]
        input_labels = [[[1]]]

        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = hf_model._single_frame_forward(**inputs)
        scores = output.iou_scores.squeeze()

        assert torch.allclose(scores, torch.tensor([0.0356, 0.2141, 0.9707]).cuda(), atol=1e-3)

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        repo_id = f"yonigozlan/{pytorch_dump_folder.split('/')[-1]}"
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["EdgeTAM"]
    parser.add_argument(
        "--model_name",
        default="EdgeTAM",
        choices=choices,
        type=str,
        help="Name of the original model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    parser.add_argument("--pytorch_dump_folder_path", default="", type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )
    parser.add_argument(
        "--run_sanity_check",
        action="store_true",
        help="Whether to run the sanity check after converting",
    )

    args = parser.parse_args()

    hf_model_name = args.model_name.replace("_", "-")
    checkpoint_path = (
        hf_hub_download(f"facebook/{hf_model_name}", f"{args.model_name.lower()}.pt")
        if args.checkpoint_path is None
        else args.checkpoint_path
    )

    convert_edgetam_checkpoint(
        args.model_name, checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.run_sanity_check
    )
