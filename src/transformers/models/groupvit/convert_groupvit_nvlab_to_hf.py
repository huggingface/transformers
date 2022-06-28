# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

"""
Convert GroupViT checkpoints from the original repository.

URL: https://github.com/NVlabs/GroupViT
"""

import argparse

import torch
from PIL import Image

import requests
from transformers import CLIPProcessor, GroupViTConfig, GroupViTModel


def rename_key(name):
    # vision encoder
    if "img_encoder.pos_embed" in name:
        name = name.replace("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings")
    if "img_encoder.patch_embed.proj" in name:
        name = name.replace("img_encoder.patch_embed.proj", "vision_model.embeddings.patch_embeddings.projection")
    if "img_encoder.patch_embed.norm" in name:
        name = name.replace("img_encoder.patch_embed.norm", "vision_model.embeddings.layernorm")
    if "img_encoder.layers" in name:
        name = name.replace("img_encoder.layers", "vision_model.encoder.stages")
    if "blocks" in name and "res" not in name:
        name = name.replace("blocks", "layers")
    if "attn" in name and "pre_assign" not in name:
        name = name.replace("attn", "self_attn")
    if "proj" in name and "self_attn" in name and "text" not in name:
        name = name.replace("proj", "out_proj")
    if "pre_assign_attn.attn.proj" in name:
        name = name.replace("pre_assign_attn.attn.proj", "pre_assign_attn.attn.out_proj")
    if "norm1" in name:
        name = name.replace("norm1", "layer_norm1")
    if "norm2" in name and "pre_assign" not in name:
        name = name.replace("norm2", "layer_norm2")
    if "img_encoder.norm" in name:
        name = name.replace("img_encoder.norm", "vision_model.layernorm")
    # text encoder
    if "text_encoder.token_embedding" in name:
        name = name.replace("text_encoder.token_embedding", "text_model.embeddings.token_embedding")
    if "text_encoder.positional_embedding" in name:
        name = name.replace("text_encoder.positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "text_encoder.transformer.resblocks." in name:
        name = name.replace("text_encoder.transformer.resblocks.", "text_model.encoder.layers.")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "text_encoder" in name:
        name = name.replace("text_encoder", "text_model")
    if "ln_final" in name:
        name = name.replace("ln_final", "final_layer_norm")
    # projection layers
    if "img_projector.linear_hidden." in name:
        name = name.replace("img_projector.linear_hidden.", "visual_projection.")
    if "img_projector.linear_out." in name:
        name = name.replace("img_projector.linear_out.", "visual_projection.3.")
    if "text_projector.linear_hidden" in name:
        name = name.replace("text_projector.linear_hidden", "text_projection")
    if "text_projector.linear_out" in name:
        name = name.replace("text_projector.linear_out", "text_projection.3")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            # weights and biases of the key, value and query projections of vision encoder's attention layers require special treatment:
            # we need to split them up into separate matrices/vectors
            key_split = key.split(".")
            stage_num, layer_num = int(key_split[2]), int(key_split[4])
            dim = config.vision_config.hidden_size
            if "weight" in key:
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.q_proj.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.k_proj.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.v_proj.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.q_proj.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.k_proj.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"vision_model.encoder.stages.{stage_num}.layers.{layer_num}.self_attn.v_proj.bias"
                ] = val[-dim:]
        elif "in_proj" in key:
            # weights and biases of the key, value and query projections of text encoder's attention layers require special treatment:
            # we need to split them up into separate matrices/vectors
            key_split = key.split(".")
            layer_num = int(key_split[3])
            dim = config.text_config.hidden_size
            if "weight" in key:
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_name = rename_key(key)
            # squeeze if necessary
            if (
                "text_projection.0" in new_name
                or "text_projection.3" in new_name
                or "visual_projection.0" in new_name
                or "visual_projection.3" in new_name
            ):
                orig_state_dict[new_name] = val.squeeze_()
            else:
                orig_state_dict[new_name] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_groupvit_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, model_name="groupvit-gcc-yfcc", push_to_hub=False
):
    """
    Copy/paste/tweak model's weights to the Transformers design.
    """
    config = GroupViTConfig()
    model = GroupViTModel(config).eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    new_state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert missing_keys == ["text_model.embeddings.position_ids"]
    assert (unexpected_keys == ["multi_label_logit_scale"]) or (len(unexpected_keys) == 0)

    # verify result
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = prepare_img()
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    if model_name == "groupvit-gcc-yfcc":
        expected_logits = torch.tensor([[13.3523, 6.3629]])
    elif model_name == "groupvit-gcc-redcaps":
        expected_logits = torch.tensor([[16.1873, 8.6230]])
    else:
        raise ValueError(f"Model name {model_name} not supported.")
    assert torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3)

    processor.save_pretrained(pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)
    print("Successfully saved processor and model to", pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        processor.push_to_hub(model_name, organization="nielsr")
        model.push_to_hub(model_name, organization="nielsr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to dump the processor and PyTorch model."
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to GroupViT checkpoint")
    parser.add_argument(
        "--model_name",
        default="groupvit-gccy-fcc",
        type=str,
        help="Name of the model. Expecting either 'groupvit-gcc-yfcc' or 'groupvit-gcc-redcaps'",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub using the provided `model_name`.",
    )
    args = parser.parse_args()

    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
