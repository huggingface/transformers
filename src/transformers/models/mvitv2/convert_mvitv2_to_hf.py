import argparse
import urllib.request
import re
import requests

import numpy as np
import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import MViTV2ForImageClassification, MViTV2Config, AutoImageProcessor

def get_mvitv2_config(model_name):
    config = MViTV2Config()
    config.num_labels = 1000
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(url).text.strip().split("\n")
    config.id2label = {i: label for i, label in enumerate(labels)}
    config.label2id = {label: i for i, label in enumerate(labels)}
    if "tiny" in model_name:
        config.depths = (1, 2, 5, 2)
    if "small" in model_name:
        config.depths = (1, 2, 11, 2)
    if "large" in model_name:
        config.depths = (2, 6, 36, 4)
        config.hidden_size = 144
        config.num_heads = 2
        config.expand_feature_dimension_in_attention = False
    return config

def rename_key(name):
    if name.startswith("patch_embed"):
        name = name.replace("patch_embed", "patch_embeddings")
        return "mvitv2.embeddings." + name
    if name.startswith("head.projection."):
        return name.replace("head.projection.", "classifier.")

    name = "mvitv2.encoder." + name

    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")

    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if "attn.qkv" in name:
        name = name.replace("attn.qkv", "attention.attention.qkv")

    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")

    if "attn" in name:
        name = name.replace("attn", "attention.attention")

    if ".proj." in name:
        name = name.replace(".proj.", ".shortcut_project_feature_dim.")

    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")

    if ".norm." in name:
        name = name.replace(".norm.", ".layer_norm.")

    replacements = {
        "rel_pos_h": "relative_positional_embeddings_height",
        "rel_pos_w": "relative_positional_embeddings_width",
        "pool_q": "pool_query",
        "pool_k": "pool_key",
        "pool_v": "pool_value",
        "norm_q": "norm_query",
        "norm_k": "norm_key",
        "norm_v": "norm_value",
    }

    for key, value in replacements.items():
        if key in name:
            name = name.replace(key, value)

    return name

def convert_state_dict(orig_state_dict, config):
    depths_arr = np.array(config.depths)
    depths_cum = depths_arr.cumsum()
    depths_cum = np.insert(depths_cum, 0, 0)
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        if "blocks" in key:
            block_num = int(key.split(".")[1])
            new_stage_num = np.sum(block_num >= depths_cum) - 1
            new_block_num = block_num - depths_cum[new_stage_num]
            key = re.sub(r"blocks\.\d+", f"stages.{new_stage_num}.blocks.{new_block_num}", key)
        orig_state_dict[rename_key(key)] = val
    return orig_state_dict

def prepare_image():
    file = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_nlvr2", filename="image2.jpeg", repo_type="dataset"
    )
    image = Image.open(file)
    return image

def convert_timesformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    config = get_mvitv2_config(model_name)

    model = MViTV2ForImageClassification(config)

    original_file = "original.pyth"
    urllib.request.urlretrieve(checkpoint_url, original_file)
    original_checkpoint = torch.load(original_file, map_location="cpu", weights_only=True)
    state_dict = original_checkpoint["model_state"]
    state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(state_dict)
    model.eval()

    image = prepare_image()
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", image_mean=IMAGENET_DEFAULT_MEAN, 
                                                   image_std=IMAGENET_DEFAULT_STD)
    
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs) 

    # Logits and predicted class
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    if(model_name == "mvitv2-base"):
        expected_class = 273
        expected_shape = torch.Size([1, 1000])
        expected_slice = torch.Tensor([-2.4503e-01, 3.0206e-01, 1.0540e-01])
    elif(model_name == "mvitv2-small"):
        expected_class = 273
        expected_shape = torch.Size([1, 1000])
        expected_slice = torch.Tensor([2.2804e-01, 3.6781e-01, -4.1515e-01])
    elif(model_name == "mvitv2-tiny"):
        expected_class = 273
        expected_shape = torch.Size([1, 1000])
        expected_slice = torch.Tensor([2.5125e-02, 1.3908e-01, -2.4903e-01])
    elif(model_name == "mvitv2-large"):
        expected_class = 273
        expected_shape = torch.Size([1, 1000])
        expected_slice = torch.Tensor([8.2464e-02, -3.0645e-01, 1.8997e-02])

    else:
        print("Model not supported! Should be mvitv2-base, mvitv2-small, mvitv2-tiny or mvitv2-large")

    assert predicted_class == expected_class
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    assert logits.shape == expected_shape

    print("Logits ok!")

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"KamilaMila/{model_name}")
        processor.push_to_hub(f"KamilaMila/{model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="mvitv2-base", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_timesformer_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub
    )
