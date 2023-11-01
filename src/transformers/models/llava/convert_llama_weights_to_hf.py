import json
import os
from collections import OrderedDict

import requests
import torch
from PIL import Image

from transformers import LlaVaConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def rename_key(state_dict, old, new):
    if old in state_dict:
        val = state_dict.pop(old)
        state_dict[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_llava_checkpoint(
    checkpoint_path_llava_1, checkpoint_path_llava_2, checkpoint_path_clip, pytorch_dump_folder_path
):
    """
    Copy/paste/tweak model's weights to our ProPainter structure.
    """

    # define default ViT configuration
    config = LlaVaConfig()

    # load original models
    llava_state_dict = OrderedDict()
    llava_state_dict = torch.load(checkpoint_path_llava_1, map_location="cpu")
    rename_keys = []
    for i in llava_state_dict:
        rename_keys.append((i, "model.text_model." + i))

    for src, dest in rename_keys:
        rename_key(llava_state_dict, src, dest)

    index_dict = {"weight_map": {}}
    param_count = 0
    filename = "pytorch_model-00001-of-00002.bin"
    for k, v in llava_state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(llava_state_dict, os.path.join("./temp", filename))

    llava_state_dict = None
    rename_keys = None

    llava_state_dict = OrderedDict()
    llava_state_dict = torch.load(checkpoint_path_llava_2, map_location="cpu")
    rename_keys = []
    for i in llava_state_dict:
        rename_keys.append((i, "model.text_model." + i))

    for src, dest in rename_keys:
        rename_key(llava_state_dict, src, dest)
    print("here")
    llava_state_dict.update(torch.load(checkpoint_path_clip, map_location="cpu"))
    print("another")
    rename_keys = []
    for i in llava_state_dict:
        if "text_model" not in i:
            rename_keys.append((i, "model.vision_model." + i))

    for src, dest in rename_keys:
        rename_key(llava_state_dict, src, dest)

    index_dict = {"weight_map": {}}
    filename = "pytorch_model-00002-of-00002.bin"
    for k, v in llava_state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(llava_state_dict, os.path.join("./temp", filename))

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join("./temp", "pytorch_model.bin.index.json"))

    llava_state_dict = None
    rename_keys = None

    config = LlaVaConfig()
    config.save_pretrained("./temp")

    # print("Loading the checkpoint in a Llama model.")
    model = LlavaForCausalLM.from_pretrained(
        "./temp", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda"
    )
    # Avoid saving this as part of the config.
    # del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print("Saving in the Transformers format.")
    model.save_pretrained("./new", safe_serialization=True)


if __name__ == "__main__":
    checkpoint_path_llava_1 = "./pytorch_model-00001-of-00002.bin"
    checkpoint_path_llava_2 = "./pytorch_model-00002-of-00002.bin"
    checkpoint_path_clip = "./pytorch_model.bin"
    pytorch_dump_folder_path = "./"

    convert_llava_checkpoint(
        checkpoint_path_llava_1,
        checkpoint_path_llava_2,
        checkpoint_path_clip,
        pytorch_dump_folder_path,
    )
