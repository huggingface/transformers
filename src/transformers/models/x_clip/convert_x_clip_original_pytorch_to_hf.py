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

import argparse

import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, XClipConfig, XClipModel


def get_xclip_config(model_name):
    config = XClipConfig()
    return config


def rename_key(name):
    # text encoder
    if name == "token_embedding.weight":
        name = name.replace("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    if name == "positional_embedding":
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if name.startswith("transformer.resblocks"):
        name = name.replace("transformer.resblocks", "text_model.encoder.layers")
    if "attn.out_proj" in name and "message" not in name:
        name = name.replace("attn.out_proj", "self_attn.out_proj")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # visual encoder
    if name == "visual.class_embedding":
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if name == "visual.positional_embedding":
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    if name.startswith("visual.transformer.resblocks"):
        name = name.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layernorm")
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    # things on top
    if "prompts_visual_proj" in name:
        name = name.replace("prompts_visual_proj", "prompts_visual_projection")
    if "prompts_visual_ln" in name:
        name = name.replace("prompts_visual_ln", "prompts_visual_layernorm")
    # mit
    if name == "mit.positional_embedding":
        name = name.replace("positional", "position")
    if name.startswith("mit.resblocks"):
        name = name.replace("mit.resblocks", "mit.encoder.layers")
    # prompts generator
    if name.startswith("prompts_generator.norm"):
        name = name.replace("prompts_generator.norm", "prompts_generator.layernorm")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "attn.in_proj" in key:
            key_split = key.split(".")
            if key.startswith("visual"):
                layer_num = key_split[3]
                dim = config.vision_config.hidden_size
                if "message_attn" in key:
                    if "weight" in key:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.q_proj.weight"] = val[
                            :dim, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.k_proj.weight"] = val[
                            dim : dim * 2, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.v_proj.weight"] = val[
                            -dim:, :
                        ]
                    else:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.q_proj.bias"] = val[
                            :dim
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.k_proj.bias"] = val[
                            dim : dim * 2
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.v_proj.bias"] = val[
                            -dim:
                        ]
                else:
                    if "weight" in key:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[
                            :dim, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                            dim : dim * 2, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[
                            -dim:, :
                        ]
                    else:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                            dim : dim * 2
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
            elif key.startswith("mit"):
                layer_num = key_split[2]
                dim = config.vision_config.mit_hidden_size
                if "weight" in key:
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
                else:
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
            else:
                layer_num = key_split[2]
                dim = config.text_config.hidden_size
                if "weight" in key:
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                        dim : dim * 2, :
                    ]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
                else:
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                        dim : dim * 2
                    ]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_key_name = rename_key(key)
            if new_key_name in ["visual_projection.weight", "text_projection.weight"]:
                val = val.T
            orig_state_dict[new_key_name] = val

    return orig_state_dict


def convert_xclip_checkpoint(checkpoint_url, model_name, pytorch_dump_folder_path):
    config = get_xclip_config(model_name)
    model = XClipModel(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    state_dict = convert_state_dict(state_dict, config)

    model = XClipModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ["text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"]
    model.eval()

    file_path = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video-8-frames", filename="pixel_values.pt", repo_type="dataset"
    )
    pixel_values = torch.load(file_path)

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    # inputs = feature_extractor(images=image, return_tensors="pt")

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    input_ids = tokenizer(
        ["playing sports", "eating spaghetti", "go shopping"], padding="max_length", return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)

    # Verify outputs
    logits_per_image = outputs.logits_per_image
    print("Shape of logits per image:", logits_per_image.shape)
    probs = logits_per_image.softmax(dim=1)
    expected_probs = torch.tensor([[[0.0019], [0.9951], [0.0030]]])
    assert torch.allclose(probs, expected_probs, atol=1e-3)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_8.pth",
        type=str,
        help="URL fo the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--model_name",
        default="xclip-base-patch32",
        type=str,
        help="Name of the model.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_xclip_checkpoint(args.checkpoint_url, args.model_name, args.pytorch_dump_folder_path)
