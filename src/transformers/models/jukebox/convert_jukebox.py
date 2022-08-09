# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ViT checkpoints trained with the DINO method."""


import argparse
from pathlib import Path

import torch

import requests
from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging
import os

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


PREFIX = "https://openaipublic.azureedge.net/jukebox/models/"
MODEL_MAPPING = {
    "jukebox-1b-lyrics": [
        "5b/vqvae.pth.tar",
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "1b_lyrics/prior_level_2.pth.tar",
    ],
    "jukebox-5b": [
        "5b/vqvae.pth.tar5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b/prior_level_2.pth.tar",
    ],
    "jukebox-5b-lyrics": [
        "5b/vqvae.pth.tar5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b_lyrics/prior_level_2.pth.tar",
    ],
}

def replace_key(key) : 
    if ".k." in key:   # replace vqvae.X.k with vqvae.X.codebook
        return key.replace(".k.", ".codebook.")
    elif ".y_emb." in key:
            key = key.replace(".y_emb.", ".metadata_embedding.")
      
          
# TODO right a clean conversion code using regex or replace 
# depending on the most appropriate choice            
def fix_jukebox_keys(state_dict, model_state_dict):
    new_dict = {}
    model_unformatted_keys = {".".join(k.split('.')[2:]) for k in model_state_dict.keys()}
    import re 
    model_to_conv = {1:"conv1d_1", 3:"conv1d_2"}
    re_cond_block = re.compile("conditioner_blocks.([\d]).cond.model.([\d]).([\d]).model.([\d])")
    groups = re_cond_block.match(original_key).groups()
    block_index = int(groups[0]) * 2 + int(groups[1])
    re_new_key = f"conditioner_blocks.{groups[0]}.upsampler.upsample_block.{block_index}.resnet_block.{model_to_conv[groups[-1]]}"
    
    re_cond_block.sub(re_new_key,original_key)
    
    for original_key, value in state_dict.items():
        key = original_key
        
        if ".k." in key:
            key = key.replace(".k.", ".codebook.")
        
        elif ".y_emb." in key:
            key = key.replace(".y_emb.", ".metadata_embedding.")
        else:
            wo_model = key.split("model")
            if len(wo_model) == 2 and "encoders" in key:
                if len(wo_model[1].split(".")) <= 3:
                    key = wo_model[0] + "proj_out." + wo_model[1].split(".")[-1]
                else:
                    block_index = str(
                        int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2])
                    )
                    key = (
                        wo_model[0]
                        + "downsample_block."
                        + block_index
                        + "."
                        + wo_model[1].split(".")[-1]
                    )
            elif len(wo_model) == 2 and "decoders" in key:
                if len(wo_model[1].split(".")) <= 3:
                    key = wo_model[0] + "proj_in." + wo_model[1].split(".")[-1]
                else:
                    block_index = str(
                        int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2]) - 2
                    )
                    key = (
                        wo_model[0]
                        + "upsample_block."
                        + block_index
                        + "."
                        + wo_model[1].split(".")[-1]
                    )
            elif len(wo_model) == 2 and "cond.model." in key:
                if len(wo_model[1].split(".")) <= 3:
                    key = wo_model[0] + "proj_in." + wo_model[1].split(".")[-1]
                else:
                    block_index = str(
                        int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2]) - 2
                    )
                    key = (
                        wo_model[0]
                        + "upsample_block."
                        + block_index
                        + "."
                        + wo_model[1].split(".")[-1]
                    )
            elif len(wo_model) == 3 and "priors" in key:
                # should also rename cond to low_lvl_conditioner
                block_index = str(
                    int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2]) - 2
                )
                key = (
                    wo_model[0]
                    + "upsample_block."
                    + block_index
                    + ".resnet_block."
                    + wo_model[1].split(".")[-2]
                    + ".model"
                    + wo_model[2]
                )
            elif len(wo_model) == 4 and "decoders" in key:
                # convert from
                # model.1.0 is the first upsample block's resnet layer. Then this
                # layer has resnet_blocks (1 to 3) which has a sequential (last model). 3 is the 3nd conv
                # vqvae.decoders.0.level_blocks.0.model.1.0.model.1.model.3.bias
                # to
                # vqvae.decoders.1.level_blocks.0.upsample_block.1.resnet_blocks.2.conv1d_2.weight
                block_index = str(
                    int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2]) - 2
                )
                key = (
                    wo_model[0]
                    + "upsample_block."
                    + block_index
                    + ".resnet_block."
                    + wo_model[2].split(".")[1]
                    + ".model"
                    + wo_model[3]
                )
            elif len(wo_model) == 4 and "encoders" in key:
                block_index = str(int(wo_model[1].split(".")[1]) * 2 + int(wo_model[1].split(".")[2]))
                key = (
                    wo_model[0]
                    + "downsample_block."
                    + block_index
                    + ".resnet_block."
                    + wo_model[2].split(".")[1]
                    + ".model"
                    + wo_model[3]
                )

            if key.endswith(".model.1.bias") and len(key.split(".")) > 10:
                key = key.replace(".model.1.bias", ".conv1d_1.bias")
            elif key.endswith(".model.1.weight") and len(key.split(".")) > 10:
                key = key.replace(".model.1.weight", ".conv1d_1.weight")
            elif key.endswith(".model.3.bias") and len(key.split(".")) > 10:
                key = key.replace(".model.3.bias", ".conv1d_2.bias")
            elif key.endswith(".model.3.weight") and len(key.split(".")) > 10:
                key = key.replace(".model.3.weight", ".conv1d_2.weight")
                
        if ".cond." in key : 
            key = key.replace(".cond.", ".upsampler.")
        if ".ln" in key : 
            key = key.replace(".ln", ".layer_norm")
        if "_ln" in key : 
            key = key.replace("_ln", "_layer_norm")
        if "prime_prior" in key:
            key = key.replace("prime_prior","lyric_encoder")
        if "prime_x_out" in key:
            key = key.replace("prime_x_out","lyric_enc_proj_out")
        # if "x_emb" in key:
        #     key = key.replace("x_emb","lyric_enc_proj_out")
        if not "conditioner_blocks" in key and "x_emb" in key:
            key = key.replace("x_emb","lyric_enc.proj_out")
        if key not in model_unformatted_keys:
            print(f"failed converting {original_key} to {key}, does not match")
        
        # elif value.shape != model_state_dict[key].shape:
        #     print(
        #         f"{original_key}-> {key} : \nshape {model_unformatted_keys[key].shape} and"
        #         f" { value.shape}, do not match"
        #     )
        #     key = original_key
        new_dict[key] = value
    return new_dict


@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    for file in MODEL_MAPPING[model_name]:
        if not os.path.isfile(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}"):
            r = requests.get(f"{PREFIX}{file}", allow_redirects=True)
            os.makedirs(f"{pytorch_dump_folder_path}/",exist_ok=True)
            open(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}", "wb").write(r.content)

    vqvae, *priors = MODEL_MAPPING[model_name.split("/")[-1]]
    vqvae_dic = torch.load(f"{pytorch_dump_folder_path}/{vqvae.split('/')[-1]}", map_location=torch.device("cpu"))[
        "model"
    ]

    config = JukeboxConfig.from_pretrained("ArthurZ/"+model_name)
    model = JukeboxModel(config)

    weight_dict = []
    for dict_name in priors:
        old_dic = torch.load(f"{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}")["model"]

        new_dic = {}
        for k in old_dic.keys():
            if k.endswith(".b"):
                new_dic[k.replace("b", "bias")] = old_dic[k]
            elif k.endswith(".w"):
                new_dic[k.replace("w", "weight")] = old_dic[k]
            elif "level_2" not in dict_name and "cond.model." in k:
                new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]

        new_dic = fix_jukebox_keys(new_dic, model.state_dict())
        weight_dict.append(new_dic)

    model.vqvae.load_state_dict(vqvae_dic)
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[i])

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path, save_config=False)

    return weight_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="jukebox-1b-lyrics",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="converted_model", type=str, help="Path to the output PyTorch model directory."
    )
    args = parser.parse_args()
    convert_openai_checkpoint(args.model_name, args.pytorch_dump_folder_path)


# previous code to convert dummy :
# weight_dict = []
# vqvae_dic = torch.load("/Users/arthur/Work/HuggingFace/jukebox/porting/vqvae.pth")
# weight_dict.append(vqvae_dic)

# for dict_name in ["up0", "up1", "up2"]:
#     old_dic = torch.load(f"/Users/arthur/Work/HuggingFace/jukebox/porting/{dict_name}.pth")
#     new_dic = {}
#     for k in old_dic.keys():
#         if k.endswith(".b"):
#             new_dic[k.replace("b", "bias")] = old_dic[k]
#         elif k.endswith(".w"):
#             new_dic[k.replace("w", "weight")] = old_dic[k]
#         elif dict_name != "up2" and "cond.model." in k:
#             new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
#         else:
#             new_dic[k] = old_dic[k]
#     weight_dict.append(new_dic)

# return weight_dict
