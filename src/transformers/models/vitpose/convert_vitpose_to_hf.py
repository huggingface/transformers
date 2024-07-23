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

from transformers import ViTPoseBackboneConfig, ViTPoseConfig, ViTPoseForPoseEstimation, ViTPoseImageProcessor
from transformers.models.vitpose.image_processing_vitpose import coco_to_pascal_voc


def get_original_pose_results(pixel_values, img_metas, output_heatmap, image_processor):
    batch_size = pixel_values.shape[0]

    centers = np.zeros((batch_size, 2), dtype=np.float32)
    scales = np.zeros((batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        centers[i, :] = img_metas[i]["center"]
        scales[i, :] = img_metas[i]["scale"]

    preds, scores = image_processor.keypoints_from_heatmaps(output_heatmap, center=centers, scale=scales)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = scores
    all_boxes[:, 0:2] = centers[:, 0:2]
    all_boxes[:, 2:4] = scales[:, 0:2]
    all_boxes[:, 4] = np.prod(scales * 200.0, axis=1)

    poses = all_preds

    # create final results by adding person bbox information
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="vitpose_person_results.pt", repo_type="dataset")
    person_results = torch.load(filepath, map_location="cpu")
    bboxes = np.array([box["bbox"] for box in person_results])
    bboxes_xyxy = coco_to_pascal_voc(bboxes)

    pose_results = []
    for pose, person_result, bbox_xyxy in zip(poses, person_results, bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result["keypoints"] = pose
        pose_result["bbox"] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results


def get_config(model_name):
    num_experts = 6 if "+" in model_name else 1
    part_features = 192 if "+" in model_name else 0

    backbone_config = ViTPoseBackboneConfig(out_indices=[12], num_experts=num_experts, part_features=part_features)
    # size of the architecture
    if "small" in model_name:
        backbone_config.hidden_size = 768
        backbone_config.intermediate_size = 2304
        backbone_config.num_hidden_layers = 8
        backbone_config.num_attention_heads = 8
    elif "large" in model_name:
        backbone_config.hidden_size = 1024
        backbone_config.intermediate_size = 4096
        backbone_config.num_hidden_layers = 24
        backbone_config.num_attention_heads = 16
    elif "huge" in model_name:
        backbone_config.hidden_size = 1280
        backbone_config.intermediate_size = 5120
        backbone_config.num_hidden_layers = 32
        backbone_config.num_attention_heads = 16

    use_simple_decoder = "simple" in model_name

    config = ViTPoseConfig(
        backbone_config=backbone_config,
        num_labels=17,
        use_simple_decoder=use_simple_decoder,
    )

    return config


def rename_key(name, config):
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
    if "last_norm" in name:
        name = name.replace("last_norm", "layernorm")

    # keypoint head
    if "keypoint_head" in name and config.use_simple_decoder:
        name = name.replace("final_layer.", "")
        name = name.replace("keypoint_head", "head.conv")
    elif "keypoint_head" in name and not config.use_simple_decoder:
        name = name.replace("keypoint_head", "head")
        name = name.replace("deconv_layers.0.weight", "deconv1.weight")
        name = name.replace("deconv_layers.1.weight", "batchnorm1.weight")
        name = name.replace("deconv_layers.1.bias", "batchnorm1.bias")
        name = name.replace("deconv_layers.1.running_mean", "batchnorm1.running_mean")
        name = name.replace("deconv_layers.1.running_var", "batchnorm1.running_var")
        name = name.replace("deconv_layers.1.num_batches_tracked", "batchnorm1.num_batches_tracked")
        name = name.replace("deconv_layers.3.weight", "deconv2.weight")
        name = name.replace("deconv_layers.4.weight", "batchnorm2.weight")
        name = name.replace("deconv_layers.4.bias", "batchnorm2.bias")
        name = name.replace("deconv_layers.4.running_mean", "batchnorm2.running_mean")
        name = name.replace("deconv_layers.4.running_var", "batchnorm2.running_var")
        name = name.replace("deconv_layers.4.num_batches_tracked", "batchnorm2.num_batches_tracked")

        name = name.replace("final_layer.weight", "conv.weight")
        name = name.replace("final_layer.bias", "conv.bias")

    return name


def convert_state_dict(orig_state_dict, dim, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            if "weight" in key:
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key, config)] = val

    return orig_state_dict


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


model_name_to_file_name = {
    "vitpose-base-simple": "vitpose-b-simple.pth",
    "vitpose-base": "vitpose-b.pth",
    "vitpose-base-coco-aic-mpii": "vitpose_base_coco_aic_mpii.pth",
    "vitpose+-base": "vitpose+_base.pth",
}


@torch.no_grad()
def convert_vitpose_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ViTPose structure.
    """

    # define default ViTPose configuration
    config = get_config(model_name)

    # load HuggingFace model
    model = ViTPoseForPoseEstimation(config)
    model.eval()

    # load original state_dict
    filename = model_name_to_file_name[model_name]
    checkpoint_path = hf_hub_download(
        repo_id="nielsr/vitpose-original-checkpoints", filename=filename, repo_type="model"
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    # rename some keys
    new_state_dict = convert_state_dict(state_dict, dim=config.backbone_config.hidden_size, config=config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # TODO add associate_heads to the MoE models
    if model_name in ["vitpose-base", "vitpose-base-simple"]:
        assert missing_keys == []
        assert unexpected_keys == []
    elif model_name == "vitpose-base-coco-aic-mpii":
        for key in unexpected_keys:
            if key != "backbone.cls_token":
                assert "associate_heads" in key

    # create image processor
    image_processor = ViTPoseImageProcessor()

    # verify image processor
    image = prepare_img()
    boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]
    pixel_values = image_processor(images=image, boxes=boxes, return_tensors="pt").pixel_values

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="vitpose_batch_data.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")["img"]
    assert torch.allclose(pixel_values, original_pixel_values, atol=1e-1)

    img_metas = torch.load(filepath, map_location="cpu")["img_metas"]
    dataset_index = torch.tensor([0])

    print("Shape of pixel values:", pixel_values.shape)
    with torch.no_grad():
        # first forward pass
        outputs = model(pixel_values, dataset_index=dataset_index)
        output_heatmap = outputs.heatmaps

        # second forward pass (flipped)
        # this is done since the model uses `flip_test=True` in its test config
        pixel_values_flipped = torch.flip(pixel_values, [3])
        outputs_flipped = model(
            pixel_values_flipped,
            dataset_index=dataset_index,
            flip_pairs=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]),
        )
        output_flipped_heatmap = outputs_flipped.heatmaps

    output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

    # Verify pose_results
    pose_results = get_original_pose_results(pixel_values, img_metas, output_heatmap, image_processor)
    # This is a list of dictionaries, containing the bounding box and keypoints per detected person
    assert torch.allclose(
        torch.from_numpy(pose_results[0]["bbox"]).float(), torch.tensor([412.8, 157.61, 464.85, 294.62])
    )
    assert torch.allclose(
        torch.from_numpy(pose_results[1]["bbox"]).float(), torch.tensor([384.43, 172.21, 398.55, 206.95])
    )

    if model_name == "vitpose-base-simple":
        assert torch.allclose(
            torch.from_numpy(pose_results[1]["keypoints"][0, :3]),
            torch.tensor([3.98180511e02, 1.81808380e02, 8.66642594e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose-base":
        assert torch.allclose(
            torch.from_numpy(pose_results[1]["keypoints"][0, :3]),
            torch.tensor([3.9807913e02, 1.8182812e02, 8.8235235e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose-base-coco-aic-mpii":
        assert torch.allclose(
            torch.from_numpy(pose_results[1]["keypoints"][0, :3]),
            torch.tensor([3.98305542e02, 1.81741592e02, 8.69966745e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose+-base":
        assert torch.allclose(
            torch.from_numpy(pose_results[1]["keypoints"][0, :3]),
            torch.tensor([3.98201294e02, 1.81728302e02, 8.75046968e-01]),
            atol=5e-2,
        )
    else:
        raise ValueError("Model not supported")
    print("Looks ok!")

    # test post_process_pose_estimation
    # results are slightly different due to no flip augmentation
    hf_pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes[0])
    if model_name == "vitpose-base-simple":
        assert torch.allclose(
            torch.tensor(hf_pose_results[1]["keypoints"][0, :3]),
            torch.tensor([3.9813846e02, 1.8180725e02, 8.7446749e-01]),
            atol=5e-2,
        )
        assert hf_pose_results[0]["keypoints"].shape == (17, 3)
        assert hf_pose_results[1]["keypoints"].shape == (17, 3)

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
        default="vitpose-base-simple",
        choices=model_name_to_file_name.keys(),
        type=str,
        help="Name of the ViTPose model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_vitpose_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
