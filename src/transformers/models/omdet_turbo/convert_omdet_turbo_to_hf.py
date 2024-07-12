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

URL: https://github.com/IDEA-Research/GroundingDINO"""

import argparse

import requests
import torch
from PIL import Image

from transformers import (
    CLIPTextConfig,
    OmDetTurboConfig,
    OmDetTurboModel,
    SwinConfig,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_omdet_turbo_config(model_name):
    if "tiny" in model_name:
        window_size = 7
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        image_size = 224
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    vision_config = SwinConfig(
        window_size=window_size,
        image_size=image_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        out_indices=[2, 3, 4],
    )

    clip_config = CLIPTextConfig()

    config = OmDetTurboConfig(vision_config=vision_config, text_config=clip_config)

    return config


def create_rename_keys_vision(state_dict, config):
    rename_keys = []
    # fmt: off
    ########################################## VISION BACKBONE - START
    # embeddings
    rename_keys.append(("backbone.patch_embed.proj.weight", "vision_backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "vision_backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "vision_backbone.embeddings.norm.weight"))
    rename_keys.append(("backbone.patch_embed.norm.bias", "vision_backbone.embeddings.norm.bias"))

    for layer, depth in enumerate(config.vision_config.depths):
        for block in range(depth):
            # layer norms
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.norm1.weight",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.layernorm_before.weight"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.norm1.bias",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.layernorm_before.bias"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.norm2.weight",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.layernorm_after.weight"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.norm2.bias",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.layernorm_after.bias"))

            # relative position bias and index
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.attn.relative_position_bias_table",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.attn.relative_position_index",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_index"))

            # attention projection
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.attn.proj.weight",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.attn.proj.bias",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.output.dense.bias"))

            # mlp
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.mlp.fc1.weight",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.mlp.fc1.bias",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.mlp.fc2.weight",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{layer}.blocks.{block}.mlp.fc2.bias",
                                f"vision_backbone.encoder.layers.{layer}.blocks.{block}.output.dense.bias"))

        # downsample
        if layer != len(config.vision_config.depths)-1:
            rename_keys.append((f"backbone.layers.{layer}.downsample.reduction.weight",
                                f"vision_backbone.encoder.layers.{layer}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.layers.{layer}.downsample.norm.weight",
                                f"vision_backbone.encoder.layers.{layer}.downsample.norm.weight"))
            rename_keys.append((f"backbone.layers.{layer}.downsample.norm.bias",
                                f"vision_backbone.encoder.layers.{layer}.downsample.norm.bias"))

    # outputs
    rename_keys.append(("backbone.norm1.weight",
                        "vision_backbone.hidden_states_norms.stage2.weight"))
    rename_keys.append(("backbone.norm1.bias",
                        "vision_backbone.hidden_states_norms.stage2.bias"))
    rename_keys.append(("backbone.norm2.weight",
                        "vision_backbone.hidden_states_norms.stage3.weight"))
    rename_keys.append(("backbone.norm2.bias",
                        "vision_backbone.hidden_states_norms.stage3.bias"))
    rename_keys.append(("backbone.norm3.weight",
                        "vision_backbone.hidden_states_norms.stage4.weight"))
    rename_keys.append(("backbone.norm3.bias",
                        "vision_backbone.hidden_states_norms.stage4.bias"))
    ########################################## VISION BACKBONE - END

    ########################################## ENCODER - START
    for layer_name, params in state_dict.items():
        if "neck" in layer_name:
            layer_name_replace = layer_name.replace("neck", "encoder")
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

    ########################################## DECODER - END

    # fmt: on
    return rename_keys


def create_rename_keys_language(state_dict, config):
    rename_keys = []
    # fmt: off
    ########################################## Language BACKBONE - START
    # embedding layer
    rename_keys.append(("language_backbone.token_embedding.weight",
                        "language_backbone.model.text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("language_backbone.positional_embedding",
                        "language_backbone.model.text_model.embeddings.position_embedding.weight"))

    # final layernorm
    rename_keys.append(("language_backbone.ln_final.weight",
                        "language_backbone.model.text_model.final_layer_norm.weight"))
    rename_keys.append(("language_backbone.ln_final.bias",
                        "language_backbone.model.text_model.final_layer_norm.bias"))

    # projection layer
    rename_keys.append(("language_backbone.text_projection",
                        "language_backbone.text_projection.weight", ))

    for layer in range(config.text_config.num_hidden_layers):
            # attention
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.attn.out_proj.weight",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.out_proj.weight"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.attn.out_proj.bias",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.out_proj.bias"))
            # layernorms
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.ln_1.weight",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.layer_norm1.weight"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.ln_1.bias",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.layer_norm1.bias"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.ln_2.weight",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.layer_norm2.weight"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.ln_2.bias",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.layer_norm2.bias"))
            # mlp
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.mlp.c_fc.weight",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.mlp.fc1.weight"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.mlp.c_fc.bias",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.mlp.fc1.bias"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.mlp.c_proj.weight",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.mlp.fc2.weight"))
            rename_keys.append((f"language_backbone.transformer.resblocks.{layer}.mlp.c_proj.bias",
                                f"language_backbone.model.text_model.encoder.layers.{layer}.mlp.fc2.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v_vision(state_dict, config):
    ########################################## VISION BACKBONE - START
    embed_dim = config.vision_config.embed_dim
    for layer, depth in enumerate(config.vision_config.depths):
        hidden_size = embed_dim * 2**layer
        for block in range(depth):
            # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.layers.{layer}.blocks.{block}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.layers.{layer}.blocks.{block}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.query.weight"] = (
                in_proj_weight[:hidden_size, :]
            )
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.query.bias"] = (
                in_proj_bias[:hidden_size]
            )
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.key.weight"] = (
                in_proj_weight[hidden_size : hidden_size * 2, :]
            )
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.key.bias"] = (
                in_proj_bias[hidden_size : hidden_size * 2]
            )
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.value.weight"] = (
                in_proj_weight[-hidden_size:, :]
            )
            state_dict[f"vision_backbone.encoder.layers.{layer}.blocks.{block}.attention.self.value.bias"] = (
                in_proj_bias[-hidden_size:]
            )
    ########################################## VISION BACKBONE - END


def read_in_q_k_v_text(state_dict, config):
    hidden_size = config.text_config.projection_dim
    for layer in range(config.text_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"language_backbone.transformer.resblocks.{layer}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"language_backbone.transformer.resblocks.{layer}.attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.q_proj.weight"] = (
            in_proj_weight[:hidden_size, :]
        )
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.q_proj.bias"] = in_proj_bias[
            :hidden_size
        ]
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.k_proj.weight"] = (
            in_proj_weight[hidden_size : hidden_size * 2, :]
        )
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.k_proj.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.v_proj.weight"] = (
            in_proj_weight[-hidden_size:, :]
        )
        state_dict[f"language_backbone.model.text_model.encoder.layers.{layer}.self_attn.v_proj.bias"] = in_proj_bias[
            -hidden_size:
        ]


def read_in_q_k_v_encoder(state_dict, config):
    embed_dim = config.encoder_hidden_dim
    # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
    in_proj_weight = state_dict.pop("encoder.encoder.0.layers.0.self_attn.in_proj_weight")
    in_proj_bias = state_dict.pop("encoder.encoder.0.layers.0.self_attn.in_proj_bias")
    # next, add query, keys and values (in that order) to the state dict
    state_dict["encoder.encoder.0.layers.0.self_attn.q_proj.weight"] = in_proj_weight[:embed_dim, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:embed_dim]
    state_dict["encoder.encoder.0.layers.0.self_attn.k_proj.weight"] = in_proj_weight[embed_dim : embed_dim * 2, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.k_proj.bias"] = in_proj_bias[embed_dim : embed_dim * 2]
    state_dict["encoder.encoder.0.layers.0.self_attn.v_proj.weight"] = in_proj_weight[-embed_dim:, :]
    state_dict["encoder.encoder.0.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-embed_dim:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


@torch.no_grad()
def convert_omdet_turbo_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    verify_logits = args.verify_logits

    checkpoint_mapping = {
        "omdet-turbo-tiny": [
            "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/OmDet-Turbo_tiny_SWIN_T.pth",
            "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/ViT-B-16.pt",
        ],
    }
    # Define default OmDetTurbo configuation
    config = get_omdet_turbo_config(model_name)

    # Load original checkpoint
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict_vision = torch.hub.load_state_dict_from_url(checkpoint_url[0], map_location="cpu")["model"]
    original_state_dict_vision = {k.replace("module.", ""): v for k, v in original_state_dict_vision.items()}

    # Rename keys
    new_state_dict = original_state_dict_vision.copy()
    rename_keys_vision = create_rename_keys_vision(new_state_dict, config)

    rename_keys_language = create_rename_keys_language(new_state_dict, config)

    for src, dest in rename_keys_vision:
        rename_key(new_state_dict, src, dest)

    for src, dest in rename_keys_language:
        rename_key(new_state_dict, src, dest)

    read_in_q_k_v_vision(new_state_dict, config)
    read_in_q_k_v_text(new_state_dict, config)
    # read_in_q_k_v_encoder(new_state_dict, config)

    # Load HF model
    model = OmDetTurboModel(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Load and process test image
    # image = prepare_img()
    # transforms = T.Compose([T.Resize(size=800, max_size=1333), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    # original_pixel_values = transforms(image).unsqueeze(0)

    # image_processor = OmDetTurboImageProcessor()
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # processor = OmDetTurboProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # text = "a cat"
    # inputs = processor(images=image, text=preprocess_caption(text), return_tensors="pt")

    # assert torch.allclose(original_pixel_values, inputs.pixel_values, atol=1e-4)

    # if verify_logits:
    #     # Running forward
    #     with torch.no_grad():
    #         outputs = model(**inputs)

    #     print(outputs.logits[0, :3, :3])

    #     expected_slice = torch.tensor(
    #         [[-4.8913, -0.1900, -0.2161], [-4.9653, -0.3719, -0.3950], [-5.9599, -3.3765, -3.3104]]
    #     )

    #     assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
    #     print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
    #     processor.save_pretrained(pytorch_dump_folder_path)

    # if push_to_hub:
    #     model.push_to_hub(f"EduardoPacheco/{model_name}")
    #     processor.push_to_hub(f"EduardoPacheco/{model_name}")


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
        "--verify_logits", action="store_false", help="Whether or not to verify logits after conversion."
    )

    args = parser.parse_args()
    convert_omdet_turbo_checkpoint(args)
