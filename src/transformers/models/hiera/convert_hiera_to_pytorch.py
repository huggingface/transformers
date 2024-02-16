import argparse

import requests
import torch
from PIL import Image



def rename_key(name):
    # if "patch_embed.proj" in name:
    #     name = name.replace("patch_embed.proj", "patch_embed.projection")
    # # elif "block.proj" in name:
    # #     name = name.replace("block.proj", "block.projection")
    # elif "attn.proj" in name:
    #     name = name.replace("attn.proj", "attn.projection")
    if ".proj." in name:
        name = name.replace(".proj.", ".projection.")
    if "attn" in name:
        name = name.replace("attn", "attention")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "patch_embedding")
    return name


def convert_state_dict(orig_state_dict, config):
    updated_model_state = {rename_key(k): v for k, v in orig_state_dict.items()}
    return updated_model_state


