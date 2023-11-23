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
"""Convert RT Detr checkpoints with Timm backbone"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import RTDetrConfig, RTDetrImageProcessor, RTDetrModel


replaces = {}

# Mapping layers
conv_layers = [("conv1_1", "conv1.0"), ("conv1_2", "conv1.3")]
layers_to_map = ["conv.weight", "norm.weight", "norm.bias", "norm.running_mean", "norm.running_var"]

for origin, dest in conv_layers:
    dest_order = int(dest[-1])
    versions = [dest_order if i == 0 else dest_order + 1 for i in range(len(layers_to_map))]
    for idx, layer in enumerate(layers_to_map):
        layer_name, layer_attribute = layer.split(".")
        key = f"backbone.conv1.{origin}.{layer}"
        val = f"model.backbone._backbone.{dest[:-1]}{versions[idx]}.{layer_attribute}"
        replaces[key] = val

# Mapping layers with batch norm
conv_layers = ["conv{}.weight", "bn{}.weight", "bn{}.bias", "bn{}.running_mean", "bn{}.running_var"]
for res_layer in [0, 1, 2, 3]:
    if res_layer in (0, 3):
        blocks = [0, 1, 2]
    elif res_layer == 1:
        blocks = [0, 1, 2, 3]
    elif res_layer == 2:
        blocks = [0, 1, 2, 3, 4, 5]
    for block in blocks:
        for id_branch, branch in enumerate(["2a", "2b", "2c"]):
            for conv_layer, layer in zip(conv_layers, layers_to_map):
                key = f"backbone.res_layers.{res_layer}.blocks.{block}.branch{branch}.{layer}"
                val_end = conv_layer.format(id_branch + 1)
                val = f"model.backbone._backbone.layer{res_layer+1}.{block}.{val_end}"
                replaces[key] = val

# Mapping conv1_3
for origin, dest in zip(layers_to_map, conv_layers):
    key = f"backbone.conv1.conv1_3.{origin}"
    version = "1.6" if dest.startswith("conv") else "1"
    val = f"model.backbone._backbone.{dest.format(version)}"
    replaces[key] = val

# Mapping downsample layers
for layer_number in range(0, 4):
    extra_conv = ".conv" if layer_number != 0 else ""
    for id, layer_name in enumerate(layers_to_map):
        downsample = "1" if id == 0 else "2"
        end_layer = layer_name[layer_name.find(".") + 1 :]
        key = f"backbone.res_layers.{layer_number}.blocks.0.short{extra_conv}.{layer_name}"
        val = f"model.backbone._backbone.layer{layer_number+1}.0.downsample.{downsample}.{end_layer}"
        replaces[key] = val

# This mapping is different from models trained in COCO (e.g. Detr)
id2label = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
}

expected_logits = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.tensor(
        [-4.159348487854004, -4.703853607177734, -5.946484565734863, -5.562824249267578, -4.7707929611206055]
    ),
}

expected_logits_shape = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.Size([1, 300, 80]),
}

expected_boxes = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.tensor(
        [
            [0.1688060760498047, 0.19992263615131378, 0.21225441992282867, 0.09384090453386307],
            [0.768376350402832, 0.41226309537887573, 0.4636859893798828, 0.7233726978302002],
            [0.25953856110572815, 0.5483334064483643, 0.4777486026287079, 0.8709195256233215],
        ]
    )
}

expected_boxes_shape = {
    "rtdetr_r50vd_6x_coco_from_paddle.pth": torch.Size([1, 300, 4]),
}


def get_sample_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw)


def update_config_values(config, checkpoint_name):
    config.num_labels = 91
    config.id2label = id2label
    config.label2id = {v: int(k) for k, v in id2label.items()}

    # Real values for rtdetr_r50vd_6x_coco_from_paddle.pth
    if checkpoint_name == "rtdetr_r50vd_6x_coco_from_paddle.pth":
        config.eval_spatial_size = [640, 640]
        config.feat_channels = [256, 256, 256]
    else:
        raise ValueError(f"Checkpoint {checkpoint_name} is not valid")


def convert_rt_detr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, repo_id):
    config = RTDetrConfig()

    checkpoint_name = Path(checkpoint_url).name
    version = Path(checkpoint_url).parts[-2]

    if version != "v0.1":
        raise ValueError(f"Given checkpoint version ({version}) is not supported.")

    # Update config values based on the checkpoint
    update_config_values(config, checkpoint_name)

    # Load model with the updated config
    model = RTDetrModel(config)

    # Load checkpoints from url
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["ema"]["module"]
    # For RTDetrObjectDetection:
    # state_dict_for_object_detection = {f"model.{k}": v for k, v in state_dict.items()}
    # For RTDetrModel
    state_dict_for_object_detection = {f"{k}": v for k, v in state_dict.items()}

    # Apply mapping
    for old_key, new_key in replaces.items():
        new_val = state_dict_for_object_detection.pop(old_key)
        new_key = new_key.replace("model.", "")
        state_dict_for_object_detection[new_key] = new_val

    # model contains an object criterion (DetrLoss) which has a buffer that needs to be mapped
    state_dict_for_object_detection["criterion.empty_weight"] = model.criterion.empty_weight

    # Transfer mapped weights
    model.load_state_dict(state_dict_for_object_detection)
    model.eval()

    # Prepare image
    img = get_sample_img()
    image_processor = RTDetrImageProcessor()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    # Verify boxes
    output_boxes = outputs.pred_boxes
    assert (
        output_boxes.shape == expected_boxes_shape[checkpoint_name]
    ), f"Shapes of output boxes do not match {checkpoint_name} {version}"
    expected = expected_boxes[checkpoint_name].to(device)
    assert torch.allclose(
        output_boxes[0, :3, :], expected, atol=1e-5
    ), f"Output boxes do not match for {checkpoint_name} {version}"

    # Verify logits
    output_logits = outputs.logits.cpu()
    original_logits = torch.tensor(
        [
            [
                [-4.64763879776001, -5.001153945922852, -4.978509902954102],
                [-4.159348487854004, -4.703853607177734, -5.946484565734863],
                [-4.437461853027344, -4.65836238861084, -6.235235691070557],
            ]
        ]
    )
    assert torch.allclose(output_logits[0, :3, :3], original_logits[0, :3, :3], atol=1e-4)

    if push_to_hub:
        config.push_to_hub(
            repo_id=repo_id, commit_message="Add config from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py",
        )


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
    convert_rt_detr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
