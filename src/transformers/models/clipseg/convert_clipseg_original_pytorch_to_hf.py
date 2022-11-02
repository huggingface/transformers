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

"""Convert CLIPSeg checkpoints from the original repository. URL: https://github.com/timojl/clipseg."""

import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from transformers import CLIPSegConfig, CLIPSegForImageSegmentation, CLIPSegTextConfig, CLIPSegVisionConfig


def get_clipseg_config(model_name):
    text_config = CLIPSegTextConfig()
    vision_config = CLIPSegVisionConfig(patch_size=16)

    use_complex_transposed_convolution = True if "refined" in model_name else False

    config = CLIPSegConfig.from_text_vision_configs(
        text_config, vision_config, use_complex_transposed_convolution=use_complex_transposed_convolution
    )
    return config


def rename_key(name):
    # update prefixes
    if "clip_model" in name:
        name = name.replace("clip_model", "clipseg")
    if "transformer" in name:
        if "visual" in name:
            name = name.replace("visual.transformer", "vision_model")
        else:
            name = name.replace("transformer", "text_model")
    if "resblocks" in name:
        name = name.replace("resblocks", "encoder.layers")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "attn" in name and "self" not in name:
        name = name.replace("attn", "self_attn")
    # text encoder
    if "token_embedding" in name:
        name = name.replace("token_embedding", "text_model.embeddings.token_embedding")
    if "positional_embedding" in name and "visual" not in name:
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # vision encoder
    if "visual.class_embedding" in name:
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    if "visual.positional_embedding" in name:
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layrnorm")
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    # projection layers
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    # decoder
    if "trans_conv" in name:
        name = name.replace("trans_conv", "transposed_convolution")
    if "film_mul" in name or "film_add" in name or "reduce" in name or "transposed_convolution" in name:
        name = "decoder." + name
    if "blocks" in name:
        name = name.replace("blocks", "decoder.layers")
    if "linear1" in name:
        name = name.replace("linear1", "mlp.fc1")
    if "linear2" in name:
        name = name.replace("linear2", "mlp.fc2")
    if "norm1" in name and "layer_" not in name:
        name = name.replace("norm1", "layer_norm1")
    if "norm2" in name and "layer_" not in name:
        name = name.replace("norm2", "layer_norm2")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("clip_model") and "attn.in_proj" in key:
            key_split = key.split(".")
            if "visual" in key:
                layer_num = int(key_split[4])
                dim = config.vision_config.hidden_size
                prefix = "vision_model"
            else:
                layer_num = int(key_split[3])
                dim = config.text_config.hidden_size
                prefix = "text_model"

            if "weight" in key:
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        elif "self_attn" in key and "out_proj" not in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            dim = config.reduce_dim
            if "weight" in key:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_name = rename_key(key)
            if "visual_projection" in new_name or "text_projection" in new_name:
                val = val.T
            orig_state_dict[new_name] = val

    return orig_state_dict


image_transforms = Compose(
    [
        ToTensor(),
        Resize((352, 352)),
    ]
)


def convert_clipseg_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    config = get_clipseg_config(model_name)
    model = CLIPSegForImageSegmentation(config)
    model.eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # remove some keys
    for key in state_dict.copy().keys():
        if key.startswith("model"):
            state_dict.pop(key, None)

    # rename some keys
    state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys != ["clipseg.text_model.embeddings.position_ids", "clipseg.vision_model.embeddings.position_ids"]:
        raise ValueError("Missing keys that are not expected: {}".format(missing_keys))
    if unexpected_keys != ["decoder.reduce.weight", "decoder.reduce.bias"]:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    # TODO create feature extractor
    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    image = Image.open("/Users/nielsrogge/Documents/cats.jpg").convert("RGB")
    pixel_values = image_transforms(image).unsqueeze(0).repeat(4, 1, 1, 1)

    # prompts = ["a glass", "something to fill", "wood", "a jar"]
    # tokenizer = CLIPTokenizer.from_pretrained("openai/")
    # input_ids = CLIPTokenizer(prompts, padding="max_length", return_tensors="pt")
    input_ids = torch.tensor([[1, 2] + [9] * 75]).repeat(4, 1)

    with torch.no_grad():
        outputs = model(input_ids, pixel_values)

    print(outputs.predicted_masks[0, 0, :3, :3])

    # verify values
    expected_cond = torch.tensor([0.0548, 0.0067, -0.1543])
    expected_pooled_output = torch.tensor([0.2208, -0.7577, -0.1391])
    if "refined" in model_name:
        expected_masks_slice = torch.tensor(
            [[-6.8533, -6.8308, -6.6634], [-6.7272, -6.4926, -6.4597], [-6.4338, -6.2161, -6.2296]]
        )
    else:
        expected_masks_slice = torch.tensor(
            [[-4.1992, -4.1912, -4.1523], [-4.1509, -4.1442, -4.1091], [-4.0581, -4.0355, -4.0107]]
        )

    assert torch.allclose(outputs.predicted_masks[0, 0, :3, :3], expected_masks_slice, atol=1e-3)
    assert torch.allclose(outputs.conditional_embeddings[0, :3], expected_cond, atol=1e-3)
    assert torch.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=1e-3)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model {model_name} to the hub")
        model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="clipseg",
        type=str,
        choices=["clipseg", "clipseg-rd16", "clipseg-rd64-refined"],
        help=(
            "Name of the model. Supported models are: clipseg-rd64, clipseg-rd16 and clipseg-rd64-refined (rd meaning"
            " reduce dimension)"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/CLIPSeg/clip_plus_rd64-uni.pth",
        type=str,
        help=(
            "Path to the original checkpoint. Note that the script assumes that the checkpoint includes both CLIP and"
            " the decoder weights."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_clipseg_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
