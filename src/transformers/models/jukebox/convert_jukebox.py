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
import requests
import torch

from transformers import JukeboxConfig, JukeboxModel
from transformers.models import jukebox
from transformers.utils import logging


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
    "jukebox-5b":[
        "5b/vqvae.pth.tar"
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b/prior_level_2.pth.tar",
    ],
    "jukebox-5b-lyrics":[
        "5b/vqvae.pth.tar"
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b_lyrics/prior_level_2.pth.tar",
    ]
} 



@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    for file in MODEL_MAPPING[model_name]:
        r = requests.get(f"{PREFIX}{file}", allow_redirects=True)
        open(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}", 'wb').write(r.content)
    
    vqvae, *priors =  MODEL_MAPPING[model_name.split('/')[-1]]
    vqvae_dic = torch.load(f"{pytorch_dump_folder_path}/{vqvae.split('/')[-1]}",map_location=torch.device('cpu'))['model']
    
    weight_dict = []
    for dict_name in priors:
        old_dic = torch.load(f"{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}")['model']
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
        weight_dict.append(new_dic)
        
    config = JukeboxConfig.from_pretrained(model_name)
    model = JukeboxModel(config)
    
    model.vqvae.load_state_dict(vqvae_dic)
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[i])

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path,save_config = False)

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
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
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