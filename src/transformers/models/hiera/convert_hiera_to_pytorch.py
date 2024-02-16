import argparse

import requests
import torch
from PIL import Image



def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "patch_embed.projection")
    return name


def e(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
           pass
        else:
            new_name = rename_key(key)
            orig_state_dict[new_name] = val

    return orig_state_dict


