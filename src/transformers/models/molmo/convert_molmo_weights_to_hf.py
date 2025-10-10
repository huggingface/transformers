# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import os

import regex as re
import torch
from safetensors.torch import load_file

from transformers import (
    GenerationConfig,
    GPT2TokenizerFast,
    MolmoImageProcessor,
    MolmoImageProcessorFast,
    MolmoProcessor,
    Qwen2TokenizerFast,
)
from transformers.models.molmo import MolmoForConditionalGeneration
from transformers.models.molmo.configuration_molmo import (
    MolmoConfig,
    MolmoPoolingConfig,
    MolmoTextConfig,
    MolmoVisionConfig,
)


CHAT_TEMPLATE = (
    "{{ bos_token or '' }}"
    "{% for message in messages %}"
    "{%- if (loop.index % 2 == 1 and message['role'] != 'user') or (loop.index % 2 == 0 and message['role'].lower() != 'assistant') -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "{% if message['content'] is not string %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "{{ '<image> ' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ message['role'].capitalize() + ': ' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] + ' ' }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'text' %}"
    "{{ content['text'] + ' ' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}"
    "{% endif %}"
)


# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"model.language_model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"transformer.blocks.(\d+).att_proj.(bias|weight)":                            r"model.language_model.layers.\1.self_attn.qkv_proj.\2", # fused attentions will need to be sliced later
    r"transformer.blocks.(\d+).(q|k)_norm.weight":                                 r"model.language_model.layers.\1.self_attn.\2_norm.weight",
    r"transformer.blocks.(\d+).attn_norm.weight":                                  r"model.language_model.layers.\1.input_layernorm.weight",
    r"transformer.blocks.(\d+).attn_out.weight":                                   r"model.language_model.layers.\1.self_attn.o_proj.weight",
    r"transformer.blocks.(\d+).ff_norm.weight":                                    r"model.language_model.layers.\1.post_attention_layernorm.weight",
    r"transformer.blocks.(\d+).ff_out.weight":                                     r"model.language_model.layers.\1.mlp.fc2.weight",
    r"transformer.blocks.(\d+).ff_proj.weight":                                    r"model.language_model.layers.\1.mlp.fc1.weight",
    r"transformer.ff_out.weight":                                                  r"lm_head.weight",
    r"transformer.ln_f.(weight|bias)":                                             r"model.language_model.norm.\1", # no post layernorm bias
    r"transformer.wte.embedding":                                                  r"model.language_model.word_embeddings.weight",
    r"transformer.wte.new_embedding":                                              r"model.language_model.new_embeddings.weight",

    r"vision_backbone.image_pooling_2d.w(q|k|v|o).bias":                           r"model.adapter.image_pooling_2d.\1_proj.bias",
    r"vision_backbone.image_pooling_2d.w(q|k|v|o).weight":                         r"model.adapter.image_pooling_2d.\1_proj.weight",

    r"vision_backbone.image_projector.w(\d+).weight":                              r"model.adapter.multi_modal_projector.linear_\1.weight",

    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention.w(k|q|v).(weight|bias)":   r"model.vision_tower.vision_model.encoder.layers.\1.self_attn.\2_proj.\3",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention.wo.(weight|bias)":         r"model.vision_tower.vision_model.encoder.layers.\1.self_attn.out_proj.\2",

    r"vision_backbone.image_vit.transformer.resblocks.(\d+).attention_norm.(weight|bias)":       r"model.vision_tower.vision_model.encoder.layers.\1.layer_norm1.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).feed_forward.w1.(weight|bias)":      r"model.vision_tower.vision_model.encoder.layers.\1.mlp.fc1.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).feed_forward.w2.(weight|bias)":      r"model.vision_tower.vision_model.encoder.layers.\1.mlp.fc2.\2",
    r"vision_backbone.image_vit.transformer.resblocks.(\d+).ffn_norm.(weight|bias)":             r"model.vision_tower.vision_model.encoder.layers.\1.layer_norm2.\2",

    r"vision_backbone.image_vit.positional_embedding":                             r"model.vision_tower.vision_model.embeddings.position_embedding.weight",
    r"vision_backbone.image_vit.class_embedding":                                  r"model.vision_tower.vision_model.embeddings.class_embedding",
    r"vision_backbone.image_vit.patch_embedding.weight":                           r"model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    r"vision_backbone.image_vit.pre_ln.(weight|bias)":                             r"model.vision_tower.vision_model.pre_layernorm.\1",
    r"vision_backbone.pad_embed":                                                  r"model.adapter.pad_embed",

}
# fmt: on

CONTEXT_LENGTH = 131072


def convert_old_keys_to_new_keys(state_dict_keys: dict):
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


def write_model(
    model_path,
    input_base_path,
    variant,
    safe_serialization=True,
):
    os.makedirs(model_path, exist_ok=True)

    if os.path.isdir(input_base_path):
        weight_files = glob.glob(os.path.join(input_base_path, "model-000*"))
        config_file = os.path.join(input_base_path, "config.json")
    else:
        raise NotADirectoryError("Pass a directory for where the weights are found")

    with open(config_file, "r") as f:
        original_config = json.load(f)
    text_config = MolmoTextConfig(
        hidden_size=original_config["hidden_size"],
        num_attention_heads=original_config["num_attention_heads"],
        num_hidden_layers=original_config["num_hidden_layers"],
        num_key_value_heads=original_config["num_key_value_heads"],
        intermediate_size=original_config["intermediate_size"],
        max_position_embeddings=original_config["max_position_embeddings"],
        layer_norm_eps=original_config["layer_norm_eps"],
        rope_theta=original_config["rope_theta"],
        vocab_size=original_config["vocab_size"] if variant == "7B-D" else original_config["embedding_size"],
        additional_embedding_size=128,
        tie_word_embeddings=original_config["tie_word_embeddings"],
        use_qk_norm=variant == "7B-O",
    )

    # vision and pooling args should be same across al model checkpoints which are the default values
    vision_config = MolmoVisionConfig()
    pooling_config = MolmoPoolingConfig()
    if variant == "72B":
        pooling_config.text_intermediate_size = 59136
        pooling_config.text_hidden_size = 8192
    elif variant == "7B-O":
        pooling_config.text_intermediate_size = 22016
        pooling_config.text_hidden_size = 4096

    text_config.attention_bias = original_config["qkv_bias"]
    text_config.use_postnorm = original_config["norm_after"]
    text_config.use_attention_layer_norm = original_config["attention_layer_norm"]

    config = MolmoConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        pooling_config=pooling_config.to_dict(),
    )

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------
    state_dict = {}
    for file in weight_files:
        partial_state_dict = load_file(file)
        state_dict.update(partial_state_dict)
        del partial_state_dict

    print("Fetch keys from safetensors index map")
    safetensors_path = os.path.join(input_base_path, "model.safetensors.index.json")
    with open(safetensors_path, "r") as index_file:
        original_weights_file = json.load(index_file)
    print("Converting model...")
    all_keys = list(original_weights_file["weight_map"].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    # Some post-processing of specific params.
    for old_key, new_key in new_keys.items():
        new_key = new_key.removeprefix("model.")
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
        elif "q_norm" in new_key:
            state_dict[new_key] = state_dict[new_key]  # .reshape(config.text_config.num_attention_heads, -1)
        elif "k_norm" in new_key:
            state_dict[new_key] = state_dict[new_key]  # .reshape(config.text_config.num_key_value_heads, -1)

    gc.collect()

    with torch.device("meta"):
        model = MolmoForConditionalGeneration(config)

    # convert word embeddings. They exist separately in the Molmo custom Embedding layer.
    initial_word_embeddings = state_dict.pop("model.language_model.word_embeddings.weight")
    new_word_embeddings = state_dict.pop("model.language_model.new_embeddings.weight")
    state_dict["model.language_model.embed_tokens.weight"] = torch.cat([initial_word_embeddings, new_word_embeddings])
    print("Loading the checkpoint in a Molmo model.")
    model.load_state_dict(state_dict, strict=True, assign=True)

    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    # ------------------------------------------------------------
    # Convert processor, add special tokens, etc etc
    # ------------------------------------------------------------
    extra_special_tokens = {
        "image_token": "<image>",
        "boi_token": "<im_start>",
        "eoi_token": "<im_end>",
        "im_patch_token": "<im_patch>",
        "im_col_token": "<im_col>",
    }
    if variant in ["7B-D", "72B"]:
        tokenizer = Qwen2TokenizerFast.from_pretrained(input_base_path, extra_special_tokens=extra_special_tokens)
    elif variant == "7B-O":
        tokenizer = GPT2TokenizerFast.from_pretrained(input_base_path, extra_special_tokens=extra_special_tokens)
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.pad_token = "<|pad|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    tokenizer.save_pretrained(model_path)
    model.config.text_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    model.config.text_config.vocab_size = len(tokenizer)
    model.config.text_config.additional_embedding_size = 0

    model.resize_token_embeddings(len(tokenizer))
    print("Resizing embeddings and saving the model.")

    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    print("Model saved successfully.")

    del state_dict, model
    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    MolmoForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    print("Model reloaded successfully.")

    image_processor_class = MolmoImageProcessor if MolmoImageProcessorFast is None else MolmoImageProcessorFast
    image_processor = image_processor_class.from_pretrained(input_base_path)
    processor = MolmoProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=CHAT_TEMPLATE)
    processor.save_pretrained(model_path)
    print("Processor saved successfully.")

    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config.save_pretrained(model_path)
    print("Generation config saved successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/raid/raushan/Molmo-7B-D-0924",
        help="Location locally or on the hub of Molmo weights, which contains tokenizer.model and model folders in safetensors",
    )
    parser.add_argument(
        "--output_dir",
        default="/raid/raushan/Molmo-7B-D-hf",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=list[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--variant",
        default="7B-D",
        nargs="?",
        choices=["7B-D", "7B-O", "72B"],
        help="Whether to convert the 7B-D, 7B-O or 72B variant.",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        variant=args.variant,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
