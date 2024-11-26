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
"""Convert DepthPro checkpoints from the original repository.

URL: https://huggingface.co/apple/DepthPro/tree/main
"""

import argparse
from pathlib import Path
import re

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers.image_utils import PILImageResampling
from transformers.utils import logging

# from transformers import DepthProConfig, DepthProImageProcessorFast, DepthProForDepthEstimation
# TODO: import directly from transformers
from transformers.models.depth_pro.configuration_depth_pro import DepthProConfig
from transformers.models.depth_pro.modeling_depth_pro import DepthProForDepthEstimation
from transformers.models.depth_pro.image_processing_depth_pro_fast import DepthProImageProcessorFast


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_vit_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))

    for i in range(config.num_hidden_layers):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        # MLP
        if config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    state_dict_keys = state_dict.keys()
    for key in list(state_dict_keys):
        if "qkv" in key:
            in_proj = state_dict.pop(key)
            q, k, v = torch.split(in_proj, config.hidden_size, dim=0)

            if "fov" in key:
                key = key.replace('fov.encoder.0', 'fov_model.encoder')
            else:
                key = "depth_pro." + key

            key = key.replace("blocks", "encoder.layer")
            state_dict[key.replace("attn.qkv", "attention.attention.query")] = q
            state_dict[key.replace("attn.qkv", "attention.attention.key")] = k
            state_dict[key.replace("attn.qkv", "attention.attention.value")] = v
    return state_dict


# hard coded upsample keys
def update_hard_coded_keys(state_dict):
    mapping = [
        # upsamples
        ('encoder.upsample_latent0.0.weight', 'depth_pro.encoder.upsample_intermediate.1.proj.weight'),
        ('encoder.upsample_latent0.1.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.0.weight'),
        ('encoder.upsample_latent0.2.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.1.weight'),
        ('encoder.upsample_latent0.3.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.2.weight'),
        ('encoder.upsample_latent1.0.weight', 'depth_pro.encoder.upsample_intermediate.0.proj.weight'),
        ('encoder.upsample_latent1.1.weight', 'depth_pro.encoder.upsample_intermediate.0.upsample_blocks.0.weight'),
        ('encoder.upsample_latent1.2.weight', 'depth_pro.encoder.upsample_intermediate.0.upsample_blocks.1.weight'),
        ('encoder.upsample0.0.weight', 'depth_pro.encoder.upsample_scaled_images.2.proj.weight'),
        ('encoder.upsample0.1.weight', 'depth_pro.encoder.upsample_scaled_images.2.upsample_blocks.0.weight'),
        ('encoder.upsample1.0.weight', 'depth_pro.encoder.upsample_scaled_images.1.proj.weight'),
        ('encoder.upsample1.1.weight', 'depth_pro.encoder.upsample_scaled_images.1.upsample_blocks.0.weight'),
        ('encoder.upsample2.0.weight', 'depth_pro.encoder.upsample_scaled_images.0.proj.weight'),
        ('encoder.upsample2.1.weight', 'depth_pro.encoder.upsample_scaled_images.0.upsample_blocks.0.weight'),
        ('encoder.upsample_lowres.weight', 'depth_pro.encoder.upsample_image.upsample_blocks.0.weight'),
        ('encoder.upsample_lowres.bias', 'depth_pro.encoder.upsample_image.upsample_blocks.0.bias'),

        # neck
        ("fov.downsample.0.weight", "fov_model.global_neck.0.weight"),
        ("fov.downsample.0.bias", "fov_model.global_neck.0.bias"),
        ("fov.encoder.1.weight", "fov_model.encoder_neck.weight"),
        ("fov.encoder.1.bias", "fov_model.encoder_neck.bias"),
    ]
    for src, dest in mapping:
        state_dict[dest] = state_dict.pop(src)
    
    return state_dict


# We will verify our results on an image of cute cats
def inference_test(processor, model):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    inputs = processor(image)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    fov = outputs.fov
    target_sizes = [[image.height, image.width]] * len(predicted_depth)

    outputs = processor.post_process_depth_estimation(
        predicted_depths=predicted_depth,
        fovs=fov,
        target_sizes=target_sizes,
    )
    predicted_depth = outputs['predicted_depth']
    fov = outputs['fov']

    print("\nInference ...")
    print("predicted_depth:", predicted_depth)
    print("predicted_depth[0].shape:", predicted_depth[0].shape)
    print("fov:", fov)
    print("Inference was Successfull!\n")


@torch.no_grad()
def convert_depth_pro_checkpoint(repo_id, filename, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DepthPro structure.
    """

    # define default DepthPro configuration
    config = DepthProConfig(use_fov_model=True)

    # load original weights from huggingface hub
    file_path = hf_hub_download(repo_id, filename)
    # file_path = "/home/geetu/work/hf/depth_pro/depth_pro.pt"
    state_dict = torch.load(file_path, weights_only=True)

    # enumerate fusion layers
    n_scaled_images = len(config.scaled_images_ratios)       # 3
    n_intermediate_hooks = len(config.intermediate_hook_ids) # 2
    n_fusion_layers = n_scaled_images + n_intermediate_hooks # 5

    # 1. keys for vit encoders
    vit_rename_keys = create_vit_rename_keys(config)
    for src_prefix, dest_prefix in [
        ("encoder.patch_encoder", "depth_pro.encoder.patch_encoder"),
        ("encoder.image_encoder", "depth_pro.encoder.image_encoder"),
        ("fov.encoder.0", "fov_model.encoder"),
    ]:
        for src, dest in vit_rename_keys:
            src = src_prefix + "." + src
            dest = dest_prefix + "." + dest
            state_dict[dest] = state_dict.pop(src)

    # 2. qkv keys for vit encoders
    state_dict = read_in_q_k_v(state_dict, config)

    # 3. hard coded mapping
    state_dict = update_hard_coded_keys(state_dict)


    for key in list(state_dict.keys()):

        # 4. final depth estimation head
        if key.startswith("head."):
            new_key = "head." + key

        # 5. fov model head
        elif key.startswith("fov.head."):
            new_key = key.replace("fov", 'fov_model')

        # 6. projections between encoder and fusion
        elif "decoder.convs." in key:
            n = re.findall(r'\d+', key)[0] # find digit inside string
            n = n_fusion_layers - int(n) - 1
            new_key = f"projections.{n}.weight"

        # 7. fuse low res with image features
        elif "encoder.fuse_lowres." in key:
            new_key = key.replace("encoder.fuse_lowres", "depth_pro.encoder.fuse_image_with_low_res")

        # 8. fusion stage (decoder)
        elif key.startswith("decoder.fusions."):
            new_key = key.replace("decoder.fusions.", "fusion_stage.layers.")
            new_key = new_key.replace("resnet1", "residual_layer1")
            new_key = new_key.replace("resnet2", "residual_layer2")
            new_key = new_key.replace("residual.1", "convolution1")
            new_key = new_key.replace("residual.3", "convolution2")
            new_key = new_key.replace("out_conv", "projection")

            n_with_dots = re.findall(r'.\d+.', new_key)[0] # find digit inside string followed by .
            n = n_with_dots[1:-1]
            n = n_fusion_layers - int(n) - 1
            new_key = new_key.replace(n_with_dots, f".{n}.")

        else:
            continue

        state_dict[new_key] = state_dict.pop(key)        

    model = DepthProForDepthEstimation(config, use_fov_model=True).eval()
    model.load_state_dict(state_dict)

    processor = DepthProImageProcessorFast(
        do_resize = True,
        size = {"height": 1536, "width": 1536},
        resample = PILImageResampling.BILINEAR,
        antialias = False,
        do_rescale = True,
        rescale_factor = 1 / 255,
        do_normalize = True,
        image_mean = 0.5,
        image_std = 0.5,
        return_tensors = "pt",
    )
    inference_test(processor, model)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        hub_path = "geetu040/DepthPro"
        model.push_to_hub(hub_path)
        processor.push_to_hub(hub_path)


"""
- create files locally using function
```py
convert_depth_pro_checkpoint(
    "apple/DepthPro",
    "depth_pro.pt",
    "my_local_depth_pro_dump",
    True,
)
```

- create files locally using command line args
```cmd
python transformers/src/transformers/models/depth_pro/convert_depth_pro_to_hf.py \
    --repo_id "apple/DepthPro" \
    --filename "depth_pro.pt" \
    --pytorch_dump_folder_path "my_local_depth_pro_dump" \
    --push_to_hub
```
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--repo_id", default="apple/DepthPro", type=str, help="Name of the repo from huggingface you'd like to convert."
    )
    parser.add_argument(
        "--filename", default="depth_pro.pt", type=str, help="Name of the file from repo you'd like to convert."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_depth_pro_checkpoint(
        args.repo_id,
        args.filename,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
