import argparse
import json

import torch
from PIL import Image

import requests
import timm
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, FocalNetConfig, FocalNetForImageClassification


def get_focalnet_config(focalnet_name):
    config = FocalNetConfig()
    name_split = focalnet_name.split("_")

    model_size = name_split[1]
    img_size = int(name_split[4])

    if model_size == "tiny":
        embed_dim = 96
        depths = (2, 2, 6, 2)
    elif model_size == "small":
        embed_dim = 96
        depths = (2, 2, 18, 2)
    elif model_size == "base":
        embed_dim = 128
        depths = (2, 2, 18, 2)
    else:
        embed_dim = 192
        depths = (2, 2, 18, 2)

    # TODO id2label

    config.image_size = img_size
    config.embed_dim = embed_dim
    config.depths = depths

    return config


def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "focalnet." + name

    return name


def convert_focalnet_checkpoint(focalnet_name, pytorch_dump_folder_path):
    timm_model = timm.create_model(focalnet_name, pretrained=True)
    timm_model.eval()

    config = get_focalnet_config(focalnet_name)
    model = FocalNetForImageClassification(config)
    model.eval()

    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    model.load_state_dict(new_state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(focalnet_name.replace("_", "-")))
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    timm_outs = timm_model(inputs["pixel_values"])
    hf_outs = model(**inputs).logits

    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--focalnet_name",
        default="focalnet_tiny_patch4_window7_224",
        type=str,
        help="Name of the FocalNet timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_focalnet_checkpoint(args.focalnet_name, args.pytorch_dump_folder_path)
