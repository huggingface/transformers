# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert SegFormer checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        if "patch_embed" in key:
            # replace for example patch_embed1 by patch_embeddings.0
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if "segformer.encoder.layer_norm" in key:
            # replace for example layer_norm1 by layer_norm.0
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "block" in key:
            # replace for example block1 by block.0
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        if "linear_c" in key:
            # replace for example linear_c4 by linear_c.3
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        new_state_dict[key] = value

    return new_state_dict


def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.bias")
            # next, add keys and values (in that order) to the state dict
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_segformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our SegFormer structure.
    """

    # load default SegFormer configuration
    config = SegformerConfig()
    encoder_only = False

    # set attributes based on model_name
    repo_id = "huggingface/label-files"
    if "segformer" in model_name:
        size = model_name[len("segformer.") : len("segformer.") + 2]
        if "ade" in model_name:
            config.num_labels = 150
            filename = "ade20k-id2label.json"
            expected_shape = (1, 150, 128, 128)
        elif "city" in model_name:
            config.num_labels = 19
            filename = "cityscapes-id2label.json"
            expected_shape = (1, 19, 128, 128)
        else:
            raise ValueError(f"Model {model_name} not supported")
    elif "mit" in model_name:
        encoder_only = True
        size = model_name[4:6]
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # set config attributes
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "b0":
        pass
    elif size == "b1":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 256
    elif size == "b2":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 6, 3]
    elif size == "b3":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 18, 3]
    elif size == "b4":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 8, 27, 3]
    elif size == "b5":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 6, 40, 3]
    else:
        raise ValueError(f"Size {size} not supported")

    # load feature extractor (only resize + normalize)
    feature_extractor = SegformerFeatureExtractor(
        image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
    )

    # prepare image
    image = prepare_img()
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    if encoder_only:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]

    # rename keys
    state_dict = rename_keys(state_dict, encoder_only=encoder_only)
    if not encoder_only:
        del state_dict["decode_head.conv_seg.weight"]
        del state_dict["decode_head.conv_seg.bias"]

    # key and value matrices need special treatment
    read_in_k_v(state_dict, config)

    # create HuggingFace model and load state dict
    if encoder_only:
        config.reshape_last_stage = False
        model = SegformerForImageClassification(config)
    else:
        model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # forward pass
    outputs = model(pixel_values)
    logits = outputs.logits

    # set expected_slice based on model name
    # ADE20k checkpoints
    if model_name == "segformer.b0.512x512.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-4.6310, -5.5232, -6.2356], [-5.1921, -6.1444, -6.5996], [-5.4424, -6.2790, -6.7574]],
                [[-12.1391, -13.3122, -13.9554], [-12.8732, -13.9352, -14.3563], [-12.9438, -13.8226, -14.2513]],
                [[-12.5134, -13.4686, -14.4915], [-12.8669, -14.4343, -14.7758], [-13.2523, -14.5819, -15.0694]],
            ]
        )
    elif model_name == "segformer.b1.512x512.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-7.5820, -8.7231, -8.3215], [-8.0600, -10.3529, -10.0304], [-7.5208, -9.4103, -9.6239]],
                [[-12.6918, -13.8994, -13.7137], [-13.3196, -15.7523, -15.4789], [-12.9343, -14.8757, -14.9689]],
                [[-11.1911, -11.9421, -11.3243], [-11.3342, -13.6839, -13.3581], [-10.3909, -12.1832, -12.4858]],
            ]
        )
    elif model_name == "segformer.b2.512x512.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-11.8173, -14.3850, -16.3128], [-14.5648, -16.5804, -18.6568], [-14.7223, -15.7387, -18.4218]],
                [[-15.7290, -17.9171, -19.4423], [-18.3105, -19.9448, -21.4661], [-17.9296, -18.6497, -20.7910]],
                [[-15.0783, -17.0336, -18.2789], [-16.8771, -18.6870, -20.1612], [-16.2454, -17.1426, -19.5055]],
            ]
        )
    elif model_name == "segformer.b3.512x512.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-9.0878, -10.2081, -10.1891], [-9.3144, -10.7941, -10.9843], [-9.2294, -10.3855, -10.5704]],
                [[-12.2316, -13.9068, -13.6102], [-12.9161, -14.3702, -14.3235], [-12.5233, -13.7174, -13.7932]],
                [[-14.6275, -15.2490, -14.9727], [-14.3400, -15.9687, -16.2827], [-14.1484, -15.4033, -15.8937]],
            ]
        )
    elif model_name == "segformer.b4.512x512.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-12.3144, -13.2447, -14.0802], [-13.3614, -14.5816, -15.6117], [-13.3340, -14.4433, -16.2219]],
                [[-19.2781, -20.4128, -20.7506], [-20.6153, -21.6566, -22.0998], [-19.9800, -21.0430, -22.1494]],
                [[-18.8739, -19.7804, -21.1834], [-20.1233, -21.6765, -23.2944], [-20.0315, -21.2641, -23.6944]],
            ]
        )
    elif model_name == "segformer.b5.640x640.ade.160k":
        expected_slice = torch.tensor(
            [
                [[-9.5524, -12.0835, -11.7348], [-10.5229, -13.6446, -14.5662], [-9.5842, -12.8851, -13.9414]],
                [[-15.3432, -17.5323, -17.0818], [-16.3330, -18.9255, -19.2101], [-15.1340, -17.7848, -18.3971]],
                [[-12.6072, -14.9486, -14.6631], [-13.7629, -17.0907, -17.7745], [-12.7899, -16.1695, -17.1671]],
            ]
        )
    # Cityscapes checkpoints
    elif model_name == "segformer.b0.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-11.9295, -13.4057, -14.8106], [-13.3431, -14.8179, -15.3781], [-14.2836, -15.5942, -16.1588]],
                [[-11.4906, -12.8067, -13.6564], [-13.1189, -14.0500, -14.1543], [-13.8748, -14.5136, -14.8789]],
                [[0.5374, 0.1067, -0.4742], [0.1141, -0.2255, -0.7099], [-0.3000, -0.5924, -1.3105]],
            ]
        )
    elif model_name == "segformer.b0.512x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-7.8217, -9.8767, -10.1717], [-9.4438, -10.9058, -11.4047], [-9.7939, -12.3495, -12.1079]],
                [[-7.1514, -9.5336, -10.0860], [-9.7776, -11.6822, -11.8439], [-10.1411, -12.7655, -12.8972]],
                [[0.3021, 0.0805, -0.2310], [-0.0328, -0.1605, -0.2714], [-0.1408, -0.5477, -0.6976]],
            ]
        )
    elif model_name == "segformer.b0.640x1280.city.160k":
        expected_slice = torch.tensor(
            [
                [
                    [-1.1372e01, -1.2787e01, -1.3477e01],
                    [-1.2536e01, -1.4194e01, -1.4409e01],
                    [-1.3217e01, -1.4888e01, -1.5327e01],
                ],
                [
                    [-1.4791e01, -1.7122e01, -1.8277e01],
                    [-1.7163e01, -1.9192e01, -1.9533e01],
                    [-1.7897e01, -1.9991e01, -2.0315e01],
                ],
                [
                    [7.6723e-01, 4.1921e-01, -7.7878e-02],
                    [4.7772e-01, 9.5557e-03, -2.8082e-01],
                    [3.6032e-01, -2.4826e-01, -5.1168e-01],
                ],
            ]
        )
    elif model_name == "segformer.b0.768x768.city.160k":
        expected_slice = torch.tensor(
            [
                [[-9.4959, -11.3087, -11.7479], [-11.0025, -12.6540, -12.3319], [-11.4064, -13.0487, -12.9905]],
                [[-9.8905, -11.3084, -12.0854], [-11.1726, -12.7698, -12.9583], [-11.5985, -13.3278, -14.1774]],
                [[0.2213, 0.0192, -0.2466], [-0.1731, -0.4213, -0.4874], [-0.3126, -0.6541, -1.1389]],
            ]
        )
    elif model_name == "segformer.b1.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-13.5748, -13.9111, -12.6500], [-14.3500, -15.3683, -14.2328], [-14.7532, -16.0424, -15.6087]],
                [[-17.1651, -15.8725, -12.9653], [-17.2580, -17.3718, -14.8223], [-16.6058, -16.8783, -16.7452]],
                [[-3.6456, -3.0209, -1.4203], [-3.0797, -3.1959, -2.0000], [-1.8757, -1.9217, -1.6997]],
            ]
        )
    elif model_name == "segformer.b2.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-16.0976, -16.4856, -17.3962], [-16.6234, -19.0342, -19.7685], [-16.0900, -18.0661, -19.1180]],
                [[-18.4750, -18.8488, -19.5074], [-19.4030, -22.1570, -22.5977], [-19.1191, -20.8486, -22.3783]],
                [[-4.5178, -5.5037, -6.5109], [-5.0884, -7.2174, -8.0334], [-4.4156, -5.8117, -7.2970]],
            ]
        )
    elif model_name == "segformer.b3.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-14.2081, -14.4732, -14.1977], [-14.5867, -16.4423, -16.6356], [-13.4441, -14.9685, -16.8696]],
                [[-14.4576, -14.7073, -15.0451], [-15.0816, -17.6237, -17.9873], [-14.4213, -16.0199, -18.5992]],
                [[-4.7349, -4.9588, -5.0966], [-4.3210, -6.9325, -7.2591], [-3.4312, -4.7484, -7.1917]],
            ]
        )
    elif model_name == "segformer.b4.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-11.7737, -11.9526, -11.3273], [-13.6692, -14.4574, -13.8878], [-13.8937, -14.6924, -15.9345]],
                [[-14.6706, -14.5330, -14.1306], [-16.1502, -16.8180, -16.4269], [-16.8338, -17.8939, -20.1746]],
                [[1.0491, 0.8289, 1.0310], [1.1044, 0.5219, 0.8055], [1.0899, 0.6926, 0.5590]],
            ]
        )
    elif model_name == "segformer.b5.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                [[-12.5641, -13.4777, -13.0684], [-13.9587, -15.8983, -16.6557], [-13.3109, -15.7350, -16.3141]],
                [[-14.7074, -15.4352, -14.5944], [-16.6353, -18.1663, -18.6120], [-15.1702, -18.0329, -18.1547]],
                [[-1.7990, -2.0951, -1.7784], [-2.6397, -3.8245, -3.9686], [-1.5264, -2.8126, -2.9316]],
            ]
        )
    else:
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])

    # verify logits
    if not encoder_only:
        assert logits.shape == expected_shape
        assert torch.allclose(logits[0, :3, :3, :3], expected_slice, atol=1e-2)

    # finally, save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="segformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_segformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
