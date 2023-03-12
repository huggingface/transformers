import argparse
import json

import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import AutoFeatureExtractor, SwinConfig, SwinForImageClassification


def get_swin_config(swin_name):
    config = SwinConfig()
    name_split = swin_name.split("_")

    model_size = name_split[1]
    img_size = int(name_split[4])
    window_size = int(name_split[3][-1])

    if model_size == "tiny":
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "small":
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "base":
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    else:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)

    if "in22k" in swin_name:
        num_classes = 21841
    else:
        num_classes = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

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
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "mask" in key:
            continue
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin_checkpoint(swin_name, pytorch_dump_folder_path):
    timm_model = timm.create_model(swin_name, pretrained=True)
    timm_model.eval()

    config = get_swin_config(swin_name)
    model = SwinForImageClassification(config)
    model.eval()

    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    model.load_state_dict(new_state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(swin_name.replace("_", "-")))
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    timm_outs = timm_model(inputs["pixel_values"])
    hf_outs = model(**inputs).logits

    assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    print(f"Saving model {swin_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--swin_name",
        default="swin_tiny_patch4_window7_224",
        type=str,
        help="Name of the Swin timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_swin_checkpoint(args.swin_name, args.pytorch_dump_folder_path)
