import argparse

import torch
from PIL import Image

import requests
from transformers import ViTMAEConfig, ViTMAEFeatureExtractor, ViTMAEForPreTraining


def rename_key(name):
    if "cls_token" in name:
        name = name.replace("cls_token", "vit.embeddings.cls_token")
    if "mask_token" in name:
        name = name.replace("mask_token", "decoder.mask_token")
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "vit.embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "vit.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "vit.embeddings.norm")
    if "decoder_blocks" in name:
        name = name.replace("decoder_blocks", "decoder.decoder_layers")
    if "blocks" in name:
        name = name.replace("blocks", "vit.encoder.layer")
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
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    if "norm.weight" in name and "decoder" not in name:
        name = name.replace("norm.weight", "vit.layernorm.weight")
    if "norm.bias" in name and "decoder" not in name:
        name = name.replace("norm.bias", "vit.layernorm.bias")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            if "decoder_blocks" in key:
                dim = config.decoder_hidden_size
                prefix = "decoder.decoder_layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]
            else:
                dim = config.hidden_size
                prefix = "vit.encoder.layer."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]

        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_vit_mae_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMAEConfig()
    model = ViTMAEForPreTraining(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    feature_extractor = ViTMAEFeatureExtractor(size=config.image_size)

    new_state_dict = convert_state_dict(state_dict, config)

    for k, v in new_state_dict.items():
        print(k, v.shape)

    model.load_state_dict(new_state_dict)

    url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'

    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = ViTMAEFeatureExtractor(size=config.image_size)
    inputs = feature_extractor(images=image, return_tensors="pt")

    print("Mean of pixel values:", inputs.pixel_values.mean())
    print("Sum of pixel values:", inputs.pixel_values.sum())
    
    # forward pass
    outputs = model(**inputs)
    print("Shape of logits:", outputs.logits.shape)
    print("First values of logits:", outputs.logits[0, :3, :3])

    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_mae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
