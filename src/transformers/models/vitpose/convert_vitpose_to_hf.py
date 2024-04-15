# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert ViTPose checkpoints from the original repository.

URL: https://github.com/vitae-transformer/vitpose
"""


import argparse
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ViTPoseConfig, ViTPoseForPoseEstimation, ViTPoseImageProcessor


def get_config(model_name):
    config = ViTPoseConfig()
    # size of the architecture
    if "small" in model_name:
        config.hidden_size = 768
        config.intermediate_size = 2304
        config.num_hidden_layers = 8
        config.num_attention_heads = 8
    elif "large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "huge" in model_name:
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16

    return config


def rename_key(name):
    if "backbone" in name:
        name = name.replace("backbone", "vit")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "last_norm" in name:
        name = name.replace("last_norm", "layernorm")
    if "final_layer." in name:
        name = name.replace("final_layer.", "")
    if "keypoint_head" in name:
        name = name.replace("keypoint_head", "head.conv")

    return name


def convert_state_dict(orig_state_dict, dim):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            if "weight" in key:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vitpose_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ViTPose structure.
    """

    # define default ViTPose configuration
    config = get_config(model_name)

    # load HuggingFace model
    model = ViTPoseForPoseEstimation(config)
    model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    # for name, param in state_dict.items():
    #     print(name, param.shape)

    new_state_dict = convert_state_dict(state_dict, dim=config.hidden_size)
    model.load_state_dict(new_state_dict)

    # TODO verify image processor
    image_processor = ViTPoseImageProcessor()
    # encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # pixel_values = encoding["pixel_values"]

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="vitpose_batch_data.pt", repo_type="dataset")
    pixel_values = torch.load(filepath, map_location="cpu")["img"]
    img_metas = torch.load(filepath, map_location="cpu")["img_metas"]

    print("Shape of pixel values:", pixel_values.shape)
    with torch.no_grad():
        # first forward pass
        output_heatmap = model(pixel_values).logits

        # TODO assert logits (output heatmap)
        print("Shape of heatmap:", output_heatmap.shape)
        print("Mean value of heatmap:", output_heatmap.numpy().mean())

        print("----------------")

        # second forward pass (flipped)
        pixel_values_flipped = torch.flip(pixel_values, [3])
        print("Mean of pixel_values_flipped:", pixel_values_flipped.mean())
        output_flipped_heatmap = model(
            pixel_values_flipped, flip_pairs=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        ).logits

        print("Shape of flipped heatmap:", output_flipped_heatmap.shape)
        print("Mean value of flipped heatmap:", output_flipped_heatmap.mean())

    output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

    print("Mean of final output_heatmap:", output_heatmap.mean())

    # TODO verify postprocessing
    batch_size = pixel_values.shape[0]
    heatmaps = output_heatmap.cpu().numpy()

    if "bbox_id" in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]["center"]
        s[i, :] = img_metas[i]["scale"]
        image_paths.append(img_metas[i]["image_file"])

        if "bbox_score" in img_metas[i]:
            score[i] = np.array(img_metas[i]["bbox_score"]).reshape(-1)
        if bbox_ids is not None:
            bbox_ids.append(img_metas[i]["bbox_id"])

    preds, maxvals = image_processor.keypoints_from_heatmaps(heatmaps, center=c, scale=s, use_udp=True)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}

    result["preds"] = all_preds
    result["boxes"] = all_boxes
    result["image_paths"] = image_paths
    result["bbox_ids"] = bbox_ids

    # print(result)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and image processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and image processor for {model_name} to hub")
        model.push_to_hub(f"nielsr/{model_name}")
        image_processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitpose_base",
        type=str,
        help="Name of the ViTPose model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/ViTPose/vitpose-b-simple.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pt file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_vitpose_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
