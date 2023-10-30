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
"""Convert ViTPose from the timm library."""


import argparse
import json
from pathlib import Path

from torchvision.transforms.functional import to_pil_image
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ViTPoseConfig, ViTPoseModel,ViTPoseImageProcessor, ViTPoseForPoseEstimation
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


__models__ = []# add all the model names 

# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []

    rename_keys.extend(
        [
            ("backbone.cls_token","vitpose.embeddings.cls_token"),
            ("backbone.pos_embed","vitpose.embeddings.position_embeddings"),
            ("backbone.patch_embed.proj.weight","vitpose.embeddings.patch_embeddings.projection.weight"),
            ("backbone.patch_embed.proj.bias","vitpose.embeddings.patch_embeddings.projection.bias"),
        ]
    )

    for i in range(config.depth):
       # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
       rename_keys.append((f"backbone.blocks.{i}.norm1.weight", f"vitpose.encoder.blocks.{i}.norm1.weight"))
       rename_keys.append((f"backbone.blocks.{i}.norm1.bias", f"vitpose.encoder.blocks.{i}.norm1.bias"))
       rename_keys.append((f"backbone.blocks.{i}.attn.qkv.weight", f"vitpose.encoder.blocks.{i}.attn.qkv.weight"))
       rename_keys.append((f"backbone.blocks.{i}.attn.qkv.bias", f"vitpose.encoder.blocks.{i}.attn.qkv.bias"))
       rename_keys.append((f"backbone.blocks.{i}.attn.proj.weight", f"vitpose.encoder.blocks.{i}.attn.proj.weight"))
       rename_keys.append((f"backbone.blocks.{i}.attn.proj.bias", f"vitpose.encoder.blocks.{i}.attn.proj.bias"))
       rename_keys.append((f"backbone.blocks.{i}.norm2.weight", f"vitpose.encoder.blocks.{i}.norm2.weight"))
       rename_keys.append((f"backbone.blocks.{i}.norm2.bias", f"vitpose.encoder.blocks.{i}.norm2.bias"))
       rename_keys.append((f"backbone.blocks.{i}.mlp.fc1.weight", f"vitpose.encoder.blocks.{i}.mlp.fc1.weight"))
       rename_keys.append((f"backbone.blocks.{i}.mlp.fc1.bias", f"vitpose.encoder.blocks.{i}.mlp.fc1.bias"))
       rename_keys.append((f"backbone.blocks.{i}.mlp.fc2.weight", f"vitpose.encoder.blocks.{i}.mlp.fc2.weight"))
       rename_keys.append((f"backbone.blocks.{i}.mlp.fc2.bias", f"vitpose.encoder.blocks.{i}.mlp.fc2.bias"))


    for i in range(5):
        rename_keys.extend(
            [
                (f"keypoint_head.deconv_layers.{i}.weight",f"vitpose.keypoint_head.deconv_layers.{i}.weight"),
                (f"keypoint_head.deconv_layers.{i}.bias",f"vitpose.keypoint_head.deconv_layers.{i}.bias"),
                (f"keypoint_head.deconv_layers.{i}.running_mean",f"vitpose.keypoint_head.deconv_layers.{i}.running_mean"),
                (f"keypoint_head.deconv_layers.{i}.running_var",f"vitpose.keypoint_head.deconv_layers.{i}.running_var"),
                (f"keypoint_head.deconv_layers.{i}.num_batches_tracked",f"vitpose.keypoint_head.deconv_layers.{i}.num_batches_tracked"),
            ]
        )

    rename_keys.extend(
        [
            ("keypoint_head.final_layer.weight", "vitpose.keypoint_head.final_layer.weight"),
            ("keypoint_head.final_layer.bias", "vitpose.keypoint_head.final_layer.bias"),
            ("backbone.last_norm.weight", "vitpose.layernorm.weight"),
            ("backbone.last_norm.bias", "vitpose.layernorm.bias"),
        ]
    )

    #if base_model:
    #    # layernorm + pooler
    #    rename_keys.extend(
    #        [
    #            ("norm.weight", "layernorm.weight"),
    #            ("norm.bias", "layernorm.bias"),
    #            ("pre_logits.fc.weight", "pooler.dense.weight"),
    #            ("pre_logits.fc.bias", "pooler.dense.bias"),
    #        ]
    #    )

    #    # if just the base model, we should remove "vit" from all keys that start with "vit"
    #    rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    #else:
    #    # layernorm + classification head
    #    rename_keys.extend(
    #        [
    #            ("norm.weight", "vit.layernorm.weight"),
    #            ("norm.bias", "vit.layernorm.bias"),
    #            ("head.weight", "classifier.weight"),
    #            ("head.bias", "classifier.bias"),
    #        ]
    #    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


## change it to a simple for loop 
def rename_key(state_dict, old, new):
    if old in state_dict:
        val = state_dict.pop(old)
        state_dict[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://images.pexels.com/photos/4045762/pexels-photo-4045762.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vitpose_checkpoint(pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViTPose structure.
    """

    # define default ViTPose configuration
    config = ViTPoseConfig()
    base_model = False
    # dataset (ImageNet-21k only or also fine-tuned on ImageNet 2012), patch_size and image_size
    ## change or remove 
    vitpose_name = hf_hub_download(repo_id="shauray/VitPose", filename="vitpose_small.pth", repo_type="model")

#    if vitpose_name[-5:] == "in21k":
#        base_model = True
#        config.patch_size = int(vit_name[-12:-10])
#        config.image_size = int(vit_name[-9:-6])
#    else:
#        config.num_labels = 1000
#        repo_id = "huggingface/label-files"
#        filename = "imagenet-1k-id2label.json"
#        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
#        id2label = {int(k): v for k, v in id2label.items()}
#        config.id2label = id2label
#        config.label2id = {v: k for k, v in id2label.items()}
#        config.patch_size = int(vit_name[-6:-4])
#        config.image_size = int(vit_name[-3:])
#    # size of the architecture
#    if "deit" in vit_name:
#        if vit_name[9:].startswith("tiny"):
#            config.hidden_size = 192
#            config.intermediate_size = 768
#            config.num_hidden_layers = 12
#            config.num_attention_heads = 3
#        elif vit_name[9:].startswith("small"):
#            config.hidden_size = 384
#            config.intermediate_size = 1536
#            config.num_hidden_layers = 12
#            config.num_attention_heads = 6
#        else:
#            pass
#    else:
#        if vit_name[4:].startswith("small"):
#            config.hidden_size = 768
#            config.intermediate_size = 2304
#            config.num_hidden_layers = 8
#            config.num_attention_heads = 8
#        elif vit_name[4:].startswith("base"):
#            pass
#        elif vit_name[4:].startswith("large"):
#            config.hidden_size = 1024
#            config.intermediate_size = 4096
#            config.num_hidden_layers = 24
#            config.num_attention_heads = 16
#        elif vit_name[4:].startswith("huge"):
#            config.hidden_size = 1280
#            config.intermediate_size = 5120
#            config.num_hidden_layers = 32
#            config.num_attention_heads = 16

    # load original model from timm
    state_dict = torch.load(vitpose_name, map_location="cpu")
    state_dict = state_dict["state_dict"]
    # load state_dict of original model, remove and rename some keys
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    #read_in_q_k_v(state_dict, config, base_model)


    # load HuggingFace model
   # if vit_name[-5:] == "in21k":
   #     model = ViTModel(config).eval()
   # else:
   #     model = ViTForImageClassification(config).eval()
    # Load DeTR for detection pipeling
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=prepare_img(), return_tensors="pt",size={"height":256, "width":192})

    outputs = detr_model(**inputs)
    target_sizes = torch.tensor([inputs.pixel_values.shape[2:]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    bboxes = results["boxes"]

    model = ViTPoseForPoseEstimation(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTImageProcessor/DeiTImageProcessor
    image_processor = ViTPoseImageProcessor(size=config.img_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values, bboxes)

    show = image_processor.post_processing(prepare_img(), outputs)
    to_pil_image(show).save("pred.jpg")

   # if base_model:
   #     timm_pooled_output = timm_model.forward_features(pixel_values)
   #     assert timm_pooled_output.shape == outputs.pooler_output.shape
   #     assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=1e-3)
   # else:
   #     timm_logits = timm_model(pixel_values)
   #     assert timm_logits.shape == outputs.logits.shape
   #     assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {vitpose_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    # setting it to hard values rather then argparse, set up argparse for all the models
    parser.add_argument(
        "--pytorch_dump_folder_path", default="./", type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vitpose_checkpoint(args.pytorch_dump_folder_path)
