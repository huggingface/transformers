# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert RT Detr checkpoints from the original repository: https://github.com/lyuwenyu/RT-DETR/issues/42"""

import argparse

import requests
import torch
from PIL import Image
from pathlib import Path
from transformers import RTDetrConfig, RtDetrImageProcessor, RTDetrModel

# TODO: Rafael Convert all these weights (?)
# rtdetr_r18vd_5x_coco_objects365_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
# rtdetr_r18vd_1x_objects365_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_1x_objects365_from_paddle.pth

# rtdetr_r50vd_6x_coco_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
# rtdetr_r50vd_2x_coco_objects365_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth
# rtdetr_r50vd_1x_objects365_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_1x_objects365_from_paddle.pth

# rtdetr_r101vd_6x_coco_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth
# rtdetr_r101vd_2x_coco_objects365_from_paddle.pth -> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth
# rtdetr_r101vd_1x_objects365_from_paddle.pth-> https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_1x_objects365_from_paddle.pth

# Weights downloaded from: https://github.com/lyuwenyu/RT-DETR/issues/42
#########################

def get_sample_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw)

def update_config_values(config, checkpoint_name):
    # Real values for rtdetr_r50vd_6x_coco_from_paddle.pth
    # backbone = {"depth": 50, "variant": "d", "freeze_at": 0, "return_idx": [1, 2, 3], "num_stages": 4, "freeze_norm": True, "pretrained": True}
    # encoder = {"in_channels": [512, 1024, 2048], "feat_strides": [8, 16, 32], "hidden_dim": 256, "use_encoder_idx": [2], "num_encoder_layers": 1, "num_head": 8, "dim_feedforward": 1024, "dropout": 0.0, "enc_act": "gelu", "pe_temperature": 10000, "expansion": 1.0, "depth_mult": 1, "act_encoder": "silu", "eval_spatial_size": [640, 640]}
    # decoder = {"feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "hidden_dim": 256, "num_levels": 3, "num_queries": 300, "num_decoder_layers": 6, "num_denoising": 100, "eval_idx": -1, "eval_spatial_size": [640, 640]}
    if checkpoint_name == "rtdetr_r50vd_6x_coco_from_paddle.pth":
        config.freeze_at = 0
        config.return_idx = [1, 2, 3]
        config.eval_spatial_size = [640, 640]
        config.feat_channels = [256, 256, 256]
    else:
        raise ValueError("ERRORS")
        
 
def convert_rt_detr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, repo_id):
    config = RTDetrConfig()

    checkpoint_name = Path(checkpoint_url).name
    version = Path(checkpoint_url).parts[-2]
    
    if version != "v0.1":
        raise ValueError("Given checkpoint version is not supported.")
    
    # Update config values basaed on the checkpoint    
    update_config_values(config, checkpoint_name)
    
    # Load model with the updated config
    model = RTDetrModel(config)

    # Load checkpoints from url
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["ema"]["module"]
    # Load model with checkpoints
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare image
    img = get_sample_img()
    image_processor = RtDetrImageProcessor()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    
    # Pass image by the model
    outputs = model(pixel_values)
    
    d = {name: param for name, param in model.named_parameters() if name.startswith("backbone")}
    ds = {name: param.shape for name, param in model.named_parameters() if name.startswith("backbone")}

# PResNet
# "{'depth': 50, 'variant': 'd', 'freeze_at': 0, 'return_idx': [1, 2, 3], 'num_stages': 4, 'freeze_norm': True, 'pretrained': True}"
# HybridEncoder
# "{'in_channels': [512, 1024, 2048], 'feat_strides': [8, 16, 32], 'hidden_dim': 256, 'use_encoder_idx': [2], 'num_encoder_layers': 1, 'nhead': 8, 'dim_feedforward': 1024, 'dropout': 0.0, 'enc_act': 'gelu', 'pe_temperature': 10000, 'expansion': 1.0, 'depth_mult': 1, 'act': 'silu', 'eval_spatial_size': [640, 640]}"
# RTDETRTransformer
# "{'feat_channels': [256, 256, 256], 'feat_strides': [8, 16, 32], 'hidden_dim': 256, 'num_levels': 3, 'num_queries': 300, 'num_decoder_layers': 6, 'num_denoising': 100, 'eval_idx': -1, 'eval_spatial_size': [640, 640]}"



    # image_processor = ViTMAEImageProcessor(size=config.image_size)

    # new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTMAEImageProcessor(size=config.image_size)
    inputs = image_processor(images=image, return_tensors="pt")

    # # forward pass
    # torch.manual_seed(2)
    # outputs = model(**inputs)
    # logits = outputs.logits

    # if "large" in checkpoint_url:
    #     expected_slice = torch.tensor(
    #         [[-0.7309, -0.7128, -1.0169], [-1.0161, -0.9058, -1.1878], [-1.0478, -0.9411, -1.1911]]
    #     )
    # elif "huge" in checkpoint_url:
    #     expected_slice = torch.tensor(
    #         [[-1.1599, -0.9199, -1.2221], [-1.1952, -0.9269, -1.2307], [-1.2143, -0.9337, -1.2262]]
    #     )
    # else:
    #     expected_slice = torch.tensor(
    #         [[-0.9192, -0.8481, -1.1259], [-1.1349, -1.0034, -1.2599], [-1.1757, -1.0429, -1.2726]]
    #     )

    # verify logits
    assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)

    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)
    
    if push_to_hub:
        model.push_to_hub(repo_id=repo_id, organization="DepuMeng", commit_message="Add model")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )

    args = parser.parse_args()
    # TODO: Rafael remove it from here
    args.checkpoint_url = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
    args.repo_id = "rafaelpadilla/porting_rt_detr"
    ####
    
    convert_rt_detr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
    