# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import gc
import glob
import json
from typing import List

import regex as re
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers import (
    CLIPVisionConfig,
    MolmoConfig,
    # See below TODO
    # MolmoForConditionalGeneration,
    # MolmoConfig,
    # MolmoForConditionalGeneration,
    # MolmoImageProcessor,
    Qwen2Config,
)

# TODO why is this import not solved at modular parsing?
from transformers.models.molmo import MolmoForConditionalGeneration
from transformers.models.molmo.configuration_molmo import MolmoTextConfig, MolmoVisionConfig


# from transformers.models.molmo.configuration_molmo import MolmoTextConfig, MolmoVisionConfig

# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"transformer.blocks.(\d+).att_proj.(bias|weight)":                            r"language_model.model.layers.\1.self_attn.qkv_proj.\2", # fused attentions will need to be sliced later
    r"transformer.blocks.(\d+).attn_norm.weight":                                  r"language_model.model.layers.\1.input_layernorm.weight",
    r"transformer.blocks.(\d+).attn_out.weight":                                   r"language_model.model.layers.\1.self_attn.o_proj.weight",
    r"transformer.blocks.(\d+).ff_norm.weight":                                    r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"transformer.blocks.(\d+).ff_out.weight":                                     r"language_model.model.layers.\1.mlp.fc2.weight",
    r"transformer.blocks.(\d+).ff_proj.weight":                                    r"language_model.model.layers.\1.mlp.fc1.weight",
    r"transformer.ff_out.weight":                                                  r"language_model.lm_head.weight",
    r"transformer.ln_f.(weight|bias)":                                             r"language_model.model.norm.\1", # no post layernorm bias
    r"transformer.wte.embedding":                                                  r"language_model.model.word_embeddings.weight",
    r"transformer.wte.new_embedding":                                              r"language_model.model.new_embeddings.weight",

    r"vision_backbone.image_pooling_2d.w(q|k|v|o).bias":                           r"vision_tower.image_pooling_2d.\1_proj.bias",
    r"vision_backbone.image_pooling_2d.w(q|k|v|o).weight":                         r"vision_tower.image_pooling_2d.\1_proj.weight",

    r"vision_backbone.image_projector.w(\d+).weight":                              r"multi_modal_projector.linear_\1.weight",

    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention.w(k|q|v).(weight|bias)":   r"vision_tower.vision_model.encoder.layers.\1.self_attn.\2_proj.\3",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention.wo.(weight|bias)":         r"vision_tower.vision_model.encoder.layers.\1.self_attn.out_proj.\2",

    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention_norm.(weight|bias)":       r"vision_tower.vision_model.encoder.layers.\1.layer_norm1.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).feed_forward.w1.(weight|bias)":      r"vision_tower.vision_model.encoder.layers.\1.mlp.fc1.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).feed_forward.w2.(weight|bias)":      r"vision_tower.vision_model.encoder.layers.\1.mlp.fc2.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).ffn_norm.(weight|bias)":             r"vision_tower.vision_model.encoder.layers.\1.layer_norm2.\2",

    r"vision_backbone.image_vit.positional_embedding":                             r"vision_tower.vision_model.embeddings.position_embedding.weight",
    r"vision_backbone.image_vit.class_embedding":                                  r"vision_tower.vision_model.embeddings.class_embedding",
    r"vision_backbone.image_vit.patch_embedding.weight":                           r"vision_tower.vision_model.embeddings.patch_embedding.weight",
    r"vision_backbone.image_vit.pre_ln.(weight|bias)":                             r"vision_tower.vision_model.pre_layrnorm.\1",
    r"vision_backbone.pad_embed":                                                  r"vision_tower.pad_embed",

}
# fmt: on


# fmt: on

CONTEXT_LENGTH = 131072  # TODO change this up


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def compute_intermediate_size(hidden_dim, multiple_of=1024, ffn_dim_multiplier=1.3):
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def interpolate_positional_embedding(
    embeddings: torch.Tensor, vision_tile_size: int, vision_patch_size: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position embeddings, to be able to use the model on higher resolution
    images.
    """
    cls_embedding, positional_embedding = embeddings[:1], embeddings[1:]
    total_num_patches, dim = positional_embedding.shape

    # compute current and target number of patches for height and width
    num_patches = int(round(total_num_patches**0.5))
    new_num_patches = vision_tile_size // vision_patch_size

    # Check if the number of patches is already the desired size
    if num_patches == new_num_patches:
        return embeddings

    positional_embedding = positional_embedding.transpose(0, 1)
    positional_embedding = positional_embedding.reshape(1, dim, num_patches, num_patches)
    positional_embedding = F.interpolate(
        positional_embedding,
        size=(new_num_patches, new_num_patches),
        mode="bicubic",
        align_corners=False,
    )
    positional_embedding = positional_embedding.reshape(dim, -1).transpose(0, 1)

    embeddings = torch.cat([cls_embedding, positional_embedding], dim=0)
    return embeddings


def write_model(
    model_path,
    input_base_path,
    safe_serialization=True,
    instruct=False,
):
    # os.makedirs(model_path, exist_ok=True)
    # torch_dtype = torch.bfloat16

    #
    # Text model params and config
    # TODO
    text_config = MolmoTextConfig()
    # ------------------------------------------------------------
    # Vision model params and config
    # ------------------------------------------------------------
    # TODO
    vision_config = MolmoVisionConfig()
    # save config
    # TODO adapt this depending on model variants
    config = MolmoConfig.from_text_vision_configs(text_config=text_config, vision_config=vision_config)

    # config = MolmoConfig(vision_config=vision_config, text_config=text_config, torch_dtype=torch_dtype)
    # config.architectures = ["MolmoForConditionalGeneration"]
    # config.save_pretrained(model_path)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------
    state_dict = {}
    # TODO move from fixed path to configurable/hub
    weight_files = glob.glob("/raid/pablo/molmo/model-000*")
    for file in weight_files:
        partial_state_dict = load_file(file)
        state_dict.update(partial_state_dict)
        del partial_state_dict
    print("Fetch keys from safetensors index map")
    with open("/raid/pablo/molmo/model.safetensors.index.json", "r") as index_file:
        original_weights_file = json.load(index_file)

    print("Converting model...")
    all_keys = list(original_weights_file["weight_map"].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    # Some post-processing of specific params.
    for old_key, new_key in new_keys.items():
        new_key = new_key.removeprefix("model.")
        # remap keys
        state_dict[new_key] = state_dict.pop(old_key)
        # Post-process the current_parameter.
        if "qkv_proj" in new_key:
            # need to slice qkv fusing here
            fused_qkv = state_dict[new_key]
            fused_dims = (
                config.text_config.hidden_size,
                config.text_config.num_key_value_heads * config.text_config.head_dim,
                config.text_config.num_key_value_heads * config.text_config.head_dim,
            )
            q_proj, k_proj, v_proj = torch.split(fused_qkv, fused_dims, 0)
            if "bias" in new_key:
                state_dict[new_key.replace("qkv_proj", "q_proj")] = q_proj.clone()
                state_dict[new_key.replace("qkv_proj", "k_proj")] = k_proj.clone()
                state_dict[new_key.replace("qkv_proj", "v_proj")] = v_proj.clone()
            else:
                state_dict[new_key.replace("qkv_proj", "q_proj")] = q_proj.reshape(
                    config.text_config.hidden_size, config.text_config.hidden_size
                ).clone()
                state_dict[new_key.replace("qkv_proj", "k_proj")] = k_proj.reshape(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    config.text_config.hidden_size,
                ).clone()
                state_dict[new_key.replace("qkv_proj", "v_proj")] = v_proj.clone()
            del state_dict[new_key]

    # convert word embeddings. They exist separately in the Molmo custom Embedding layer.
    initial_word_embeddings = state_dict.pop("language_model.model.word_embeddings.weight")
    new_word_embeddings = state_dict.pop("language_model.model.new_embeddings.weight")
    state_dict["language_model.model.embed_tokens.weight"] = torch.cat([initial_word_embeddings, new_word_embeddings], dim=0)
    gc.collect()
    print("Loading the checkpoint in a Molmo model.")
    with torch.device("meta"):
        model = MolmoForConditionalGeneration(config)

    model.load_state_dict(state_dict, strict=True, assign=True)

    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    MolmoForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")

    # generation config
    # TODO should be provided by defaults in Molmo original code

    #
    """
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generation_config.save_pretrained(model_path)
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="Molmo-7B-D-0924",
        help="Location of Molmo weights, which contains tokenizer.model and model folders in safetensors",
    )
    parser.add_argument(
        "--output_dir",
        default="Molmo-7B-D-hf",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
