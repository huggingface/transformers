# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert OmDet-Turbo checkpoints from the original repository.

URL: https://github.com/om-ai-lab/OmDet"""

import argparse

import requests
import torch
from PIL import Image

from transformers import (
    CLIPTokenizer,
    DetrImageProcessor,
    OmDetTurboConfig,
    OmDetTurboForObjectDetection,
    OmDetTurboProcessor,
)


IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]


def get_omdet_turbo_config(model_name, use_timm_backbone):
    if "tiny" in model_name:
        window_size = 7
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        image_size = 640
    else:
        raise ValueError("Model not supported, only supports tiny variant.")

    config = OmDetTurboConfig(
        backbone_window_size=window_size,
        backbone_image_size=image_size,
        backbone_embed_dim=embed_dim,
        backbone_depths=depths,
        backbone_num_heads=num_heads,
        backbone_out_indices=(1, 2, 3),
        text_config={"model_type": "clip_text_model"},
        use_timm_backbone=use_timm_backbone,
        backbone="swin_tiny_patch4_window7_224" if use_timm_backbone else None,
        apply_layernorm=True if use_timm_backbone else False,
        use_pretrained_backbone=False,
    )

    return config


def create_rename_keys_vision(state_dict, config):
    rename_keys = []
    # fmt: off
    ########################################## VISION BACKBONE - START
    for layer_name in state_dict.keys():
        if layer_name.startswith("backbone") and not layer_name.startswith("backbone.norm"):
            if config.use_timm_backbone:
                layer_name_replace = layer_name.replace("backbone", "vision_backbone.vision_backbone._backbone")
                layer_name_replace = layer_name_replace.replace(".layers.", ".layers_")
                if "downsample" in layer_name:
                    # get layer number
                    layer_num = int(layer_name.split(".")[2])
                    layer_name_replace = layer_name_replace.replace(f"{layer_num}.downsample", f"{layer_num+1}.downsample")
            else:
                layer_name_replace = layer_name.replace("backbone", "vision_backbone.vision_backbone")
                layer_name_replace = layer_name_replace.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
                layer_name_replace = layer_name_replace.replace("patch_embed.norm", "embeddings.norm")
                if layer_name.startswith("backbone.layers"):
                    layer_name_replace = layer_name_replace.replace("norm1", "layernorm_before")
                    layer_name_replace = layer_name_replace.replace("norm2", "layernorm_after")
                    layer_name_replace = layer_name_replace.replace("attn.proj", "attention.output.dense")
                    layer_name_replace = layer_name_replace.replace("mlp.fc1", "intermediate.dense")
                    layer_name_replace = layer_name_replace.replace("mlp.fc2", "output.dense")
                    layer_name_replace = layer_name_replace.replace(".layers.", ".encoder.layers.")
                    layer_name_replace = layer_name_replace.replace(".attn.", ".attention.self.")
        elif layer_name.startswith("backbone.norm"):
            layer_num = int(layer_name.split("norm")[1].split(".")[0])
            if config.use_timm_backbone:
                layer_name_replace = layer_name.replace("backbone", "vision_backbone")
                layer_name_replace = layer_name_replace.replace(f"norm{layer_num}", f"layer_norms.{layer_num-1}")
            else:
                layer_name_replace = layer_name.replace(f"backbone.norm{layer_num}", f"vision_backbone.vision_backbone.hidden_states_norms.stage{layer_num+1}")
        else:
            continue
        rename_keys.append((layer_name, layer_name_replace))
    ########################################## VISION BACKBONE - END

    ########################################## ENCODER - START
    for layer_name, params in state_dict.items():
        if "neck" in layer_name:
            layer_name_replace = layer_name.replace("neck", "encoder")
            layer_name_replace = layer_name_replace.replace("input_proj", "channel_projection_layers")
            if "fpn_blocks" in layer_name or "pan_blocks" in layer_name or "lateral_convs" in layer_name or "downsample_convs" in layer_name:
                layer_name_replace = layer_name_replace.replace(".m.", ".bottlenecks.")
                layer_name_replace = layer_name_replace.replace(".cv", ".conv")
                layer_name_replace = layer_name_replace.replace(".bn", ".norm")
            if "encoder_layer" in layer_name:
                layer_name_replace = layer_name_replace.replace("encoder_layer", "encoder.0.layers.0")
                layer_name_replace = layer_name_replace.replace(".linear", ".fc")
                layer_name_replace = layer_name_replace.replace("norm1", "self_attn_layer_norm")
                layer_name_replace = layer_name_replace.replace("norm2", "final_layer_norm")
            rename_keys.append((layer_name, layer_name_replace))
    ########################################## ENCODER - END

    ########################################## DECODER - START
    for layer_name, params in state_dict.items():
        if layer_name.startswith("decoder"):
            layer_name_replace = layer_name.replace("input_proj", "channel_projection_layers")
            layer_name_replace = layer_name_replace.replace("query_pos_head", "query_position_head")
            layer_name_replace = layer_name_replace.replace("enc_bbox_head", "encoder_bbox_head")
            layer_name_replace = layer_name_replace.replace("enc_output", "encoder_vision_features")
            layer_name_replace = layer_name_replace.replace("dec_score_head", "decoder_class_head")
            layer_name_replace = layer_name_replace.replace("dec_bbox_head", "decoder_bbox_head")
            layer_name_replace = layer_name_replace.replace("enc_score_head", "encoder_class_head")
            rename_keys.append((layer_name, layer_name_replace))
    ########################################## DECODER - END
    # fmt: on
    return rename_keys


def create_rename_keys_language(state_dict):
    rename_keys = []
    # fmt: off
    for layer_name in state_dict.keys():
        if layer_name.startswith("language_backbone") and not layer_name.startswith("language_backbone.text_projection"):
            layer_name_replace = layer_name.replace("language_backbone", "language_backbone.model.text_model")
            layer_name_replace = layer_name_replace.replace("transformer.resblocks", "encoder.layers")
            layer_name_replace = layer_name_replace.replace("token_embedding", "embeddings.token_embedding")
            layer_name_replace = layer_name_replace.replace("positional_embedding", "embeddings.position_embedding.weight")
            layer_name_replace = layer_name_replace.replace(".attn", ".self_attn")
            layer_name_replace = layer_name_replace.replace(".mlp.c_fc", ".mlp.fc1")
            layer_name_replace = layer_name_replace.replace(".mlp.c_proj", ".mlp.fc2")
            layer_name_replace = layer_name_replace.replace("ln_final", "final_layer_norm")
            layer_name_replace = layer_name_replace.replace(".ln_", ".layer_norm")
            rename_keys.append((layer_name, layer_name_replace))
    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v_vision(state_dict, config):
    state_dict_keys = list(state_dict.keys())
    for layer_name_vision in state_dict_keys:
        if layer_name_vision.startswith("vision_backbone") and "qkv" in layer_name_vision:
            layer_num = int(layer_name_vision.split(".")[4])
            hidden_size = config.backbone_config.embed_dim * 2**layer_num
            if "weight" in layer_name_vision:
                in_proj_weight = state_dict.pop(layer_name_vision)
                state_dict[layer_name_vision.replace("qkv.weight", "key.weight")] = in_proj_weight[:hidden_size, :]
                state_dict[layer_name_vision.replace("qkv.weight", "query.weight")] = in_proj_weight[
                    hidden_size : hidden_size * 2, :
                ]
                state_dict[layer_name_vision.replace("qkv.weight", "value.weight")] = in_proj_weight[-hidden_size:, :]
            elif "bias" in layer_name_vision:
                in_proj_bias = state_dict.pop(layer_name_vision)
                state_dict[layer_name_vision.replace("qkv.bias", "key.bias")] = in_proj_bias[:hidden_size]
                state_dict[layer_name_vision.replace("qkv.bias", "query.bias")] = in_proj_bias[
                    hidden_size : hidden_size * 2
                ]
                state_dict[layer_name_vision.replace("qkv.bias", "value.bias")] = in_proj_bias[-hidden_size:]


def read_in_q_k_v_text(state_dict, config):
    state_dict_keys = list(state_dict.keys())
    hidden_size = config.text_config.projection_dim
    for layer_name_text in state_dict_keys:
        if layer_name_text.startswith("language_backbone") and "in_proj" in layer_name_text:
            if "weight" in layer_name_text:
                in_proj_weight = state_dict.pop(layer_name_text)
                state_dict[layer_name_text.replace("in_proj_weight", "q_proj.weight")] = in_proj_weight[
                    :hidden_size, :
                ]
                state_dict[layer_name_text.replace("in_proj_weight", "k_proj.weight")] = in_proj_weight[
                    hidden_size : hidden_size * 2, :
                ]
                state_dict[layer_name_text.replace("in_proj_weight", "v_proj.weight")] = in_proj_weight[
                    -hidden_size:, :
                ]
            elif "bias" in layer_name_text:
                in_proj_bias = state_dict.pop(layer_name_text)
                state_dict[layer_name_text.replace("in_proj_bias", "q_proj.bias")] = in_proj_bias[:hidden_size]
                state_dict[layer_name_text.replace("in_proj_bias", "k_proj.bias")] = in_proj_bias[
                    hidden_size : hidden_size * 2
                ]
                state_dict[layer_name_text.replace("in_proj_bias", "v_proj.bias")] = in_proj_bias[-hidden_size:]


def read_in_q_k_v_encoder(state_dict, config):
    embed_dim = config.encoder_hidden_dim
    # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
    in_proj_weight = state_dict.pop("encoder.encoder.0.layers.0.self_attn.in_proj_weight")
    in_proj_bias = state_dict.pop("encoder.encoder.0.layers.0.self_attn.in_proj_bias")
    # next, add query, keys and values (in that order) to the state dict
    state_dict["encoder.encoder.0.layers.0.self_attn.query.weight"] = in_proj_weight[:embed_dim, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.query.bias"] = in_proj_bias[:embed_dim]
    state_dict["encoder.encoder.0.layers.0.self_attn.key.weight"] = in_proj_weight[embed_dim : embed_dim * 2, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.key.bias"] = in_proj_bias[embed_dim : embed_dim * 2]
    state_dict["encoder.encoder.0.layers.0.self_attn.value.weight"] = in_proj_weight[-embed_dim:, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.value.bias"] = in_proj_bias[-embed_dim:]


def read_in_q_k_v_decoder(state_dict, config):
    for layer_num in range(config.decoder_num_layers):
        embed_dim = config.decoder_hidden_dim
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"decoder.decoder.layers.{layer_num}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.decoder.layers.{layer_num}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.query.weight"] = in_proj_weight[:embed_dim, :]
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.query.bias"] = in_proj_bias[:embed_dim]
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.key.weight"] = in_proj_weight[
            embed_dim : embed_dim * 2, :
        ]
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.key.bias"] = in_proj_bias[embed_dim : embed_dim * 2]
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.value.weight"] = in_proj_weight[-embed_dim:, :]
        state_dict[f"decoder.decoder.layers.{layer_num}.self_attn.value.bias"] = in_proj_bias[-embed_dim:]


def run_test(model, processor):
    # We will verify our results on an image of cute cats
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    classes = ["cat", "remote"]
    task = "Detect {}.".format(", ".join(classes))
    inputs = processor(image, text=classes, task=task, return_tensors="pt")

    # Running forward
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_slice = outputs[1][0, :3, :3]
    print(predicted_slice)
    expected_slice = torch.tensor([[0.9427, -2.5958], [0.2105, -3.4569], [-2.6364, -4.1610]])

    assert torch.allclose(predicted_slice, expected_slice, atol=1e-4)
    print("Looks ok!")


@torch.no_grad()
def convert_omdet_turbo_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    use_timm_backbone = args.use_timm_backbone

    checkpoint_mapping = {
        "omdet-turbo-tiny": [
            "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/OmDet-Turbo_tiny_SWIN_T.pth",
            "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/ViT-B-16.pt",
        ],
    }
    # Define default OmDetTurbo configuation
    config = get_omdet_turbo_config(model_name, use_timm_backbone)

    # Load original checkpoint
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict_vision = torch.hub.load_state_dict_from_url(checkpoint_url[0], map_location="cpu")["model"]
    original_state_dict_vision = {k.replace("module.", ""): v for k, v in original_state_dict_vision.items()}

    # Rename keys
    new_state_dict = original_state_dict_vision.copy()
    rename_keys_vision = create_rename_keys_vision(new_state_dict, config)

    rename_keys_language = create_rename_keys_language(new_state_dict)

    for src, dest in rename_keys_vision:
        rename_key(new_state_dict, src, dest)

    for src, dest in rename_keys_language:
        rename_key(new_state_dict, src, dest)

    if not use_timm_backbone:
        read_in_q_k_v_vision(new_state_dict, config)
    read_in_q_k_v_text(new_state_dict, config)
    read_in_q_k_v_encoder(new_state_dict, config)
    read_in_q_k_v_decoder(new_state_dict, config)
    # Load HF model
    model = OmDetTurboForObjectDetection(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    image_processor = DetrImageProcessor(
        size={"height": config.backbone_image_size, "width": config.backbone_image_size},
        do_rescale=False,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        do_pad=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = OmDetTurboProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # end-to-end consistency test
    run_test(model, processor)

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"yonigozlan/{model_name}")
        processor.push_to_hub(f"yonigozlan/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="omdet-turbo-tiny",
        type=str,
        choices=["omdet-turbo-tiny"],
        help="Name of the OmDetTurbo model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--use_timm_backbone", action="store_true", help="Whether or not to use timm backbone for vision backbone."
    )

    args = parser.parse_args()
    convert_omdet_turbo_checkpoint(args)
