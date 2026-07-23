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
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    MistralCommonBackend,
    VoxtralRealtimeConfig,
    VoxtralRealtimeFeatureExtractor,
    VoxtralRealtimeForConditionalGeneration,
    VoxtralRealtimeProcessor,
)
from transformers.utils.hub import cached_file


# fmt: off
STATE_DICT_MAPPING = {
    # Text model keys
    r"^output.weight":                                                                  r"language_model.lm_head.weight",
    r"^norm.weight":                                                                    r"language_model.model.norm.weight",
    r"^tok_embeddings.weight":                                                          r"language_model.model.embed_tokens.weight",
    r"^layers.(\d+).attention_norm.weight":                                             r"language_model.model.layers.\1.input_layernorm.weight",
    r"^layers.(\d+).ffn_norm.weight":                                                   r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"^layers.(\d+).attention.w(q|k|v|o).weight":                                       r"language_model.model.layers.\1.self_attn.\2_proj.weight",
    r"^layers.(\d+).feed_forward.w1.weight":                                            r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"^layers.(\d+).feed_forward.w2.weight":                                            r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"^layers.(\d+).feed_forward.w3.weight":                                            r"language_model.model.layers.\1.mlp.up_proj.weight",

    r"mm_streams_embeddings.embedding_module.tok_embeddings.weight":                    r"language_model.model.embed_tokens.weight",
    r"^layers.(\d+).ada_rms_norm_t_cond.0.weight":                                      r"language_model.model.layers.\1.ada_rms_norm.linear1.weight",
    r"^layers.(\d+).ada_rms_norm_t_cond.2.weight":                                      r"language_model.model.layers.\1.ada_rms_norm.linear2.weight",

    # Audio model keys
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.conv_layers\.0\.conv\.(weight|bias)": r"audio_tower.embedder.conv1.\1",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.conv_layers\.1\.conv\.(weight|bias)": r"audio_tower.embedder.conv2.\1",

    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.norm\.(weight|bias)": r"audio_tower.norm.\1",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.w([qkv])\.(weight|bias)": r"audio_tower.layers.\1.self_attn.\2_proj.\3",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(weight|bias)": r"audio_tower.layers.\1.self_attn.o_proj.\2",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(weight|bias)": r"audio_tower.layers.\1.self_attn_layer_norm.\2",

    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)": r"audio_tower.layers.\1.mlp.gate_proj.\2",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)": r"audio_tower.layers.\1.mlp.down_proj.\2",
    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(weight|bias)": r"audio_tower.layers.\1.mlp.up_proj.\2",

    r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(weight|bias)": r"audio_tower.layers.\1.final_layer_norm.\2",

    r"mm_streams_embeddings\.embedding_module\.audio_language_projection\.0\.weight":               r"multi_modal_projector.linear_1.weight",
    r"mm_streams_embeddings\.embedding_module\.audio_language_projection\.2\.weight":               r"multi_modal_projector.linear_2.weight",
}
# fmt: on


def convert_config(original_config: dict, max_position_embeddings: int = 131072):
    original_audio_config = original_config.pop("multimodal")
    original_audio_config = original_audio_config["whisper_model_args"]["encoder_args"]
    original_text_config = original_config

    # Text config
    text_key_mapping = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "intermediate_size": "hidden_dim",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "rms_norm_eps": "norm_eps",
    }
    similar_text_keys_to_keep = [
        "head_dim",
        "vocab_size",
        "rope_theta",
    ]
    new_text_config_kwargs = {k: original_text_config[v] for k, v in text_key_mapping.items()}
    new_text_config_kwargs.update({k: v for k, v in original_text_config.items() if k in similar_text_keys_to_keep})
    # These are not always defined depending on `params.json`
    new_text_config_kwargs["sliding_window"] = original_text_config.get("sliding_window", None)
    new_text_config_kwargs["max_position_embeddings"] = original_text_config.get(
        "max_seq_len", max_position_embeddings
    )
    # This may sometimes be a string in `params.json`
    if new_text_config_kwargs["sliding_window"] is not None:
        new_text_config_kwargs["sliding_window"] = int(new_text_config_kwargs["sliding_window"])

    # Audio config
    audio_key_mapping = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "intermediate_size": "hidden_dim",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_heads",
    }
    similar_audio_keys_to_keep = [
        "head_dim",
        "vocab_size",
        "rope_theta",
    ]
    new_audio_config_kwargs = {k: original_audio_config[v] for k, v in audio_key_mapping.items()}
    new_audio_config_kwargs.update({k: v for k, v in original_audio_config.items() if k in similar_audio_keys_to_keep})

    new_config = VoxtralRealtimeConfig(
        audio_config=new_audio_config_kwargs,
        text_config=new_text_config_kwargs,
        projector_hidden_act="gelu",
    )

    return new_config


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def permute_for_rope(tensor, n_heads, dim1, dim2=None):
    """Permute the weights for the ROPE formulation."""
    if dim2 is None:
        # Handle bias case where tensor is 1D with shape (dim1,)
        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2)
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(dim1)
    else:
        # Handle weight case where tensor is 2D with shape (dim1, dim2)
        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(dim1, dim2)
    return tensor


def convert_state_dict(original_state_dict, config):
    """Convert a state dict file, when a single `nn.Module` is never sharded in different files (usual case)."""
    new_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "audio_tower" in new_key:
            num_attention_heads = config.audio_config.num_attention_heads
            hidden_size = config.audio_config.hidden_size
            head_dim = config.audio_config.head_dim
            num_key_value_heads = config.audio_config.num_key_value_heads
            key_value_dim = head_dim * num_key_value_heads
            query_dim = head_dim * num_attention_heads

            tensor = _permute_projection_weights(
                tensor,
                new_key,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                hidden_size,
                query_dim,
                key_value_dim,
            )

        elif "language_model" in new_key:
            num_attention_heads = config.text_config.num_attention_heads
            hidden_size = config.text_config.hidden_size
            head_dim = config.text_config.head_dim
            num_key_value_heads = config.text_config.num_key_value_heads
            key_value_dim = head_dim * num_key_value_heads
            query_dim = head_dim * num_attention_heads

            tensor = _permute_projection_weights(
                tensor,
                new_key,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                hidden_size,
                query_dim,
                key_value_dim,
            )

        new_dict[new_key] = tensor
    return new_dict


def _permute_projection_weights(
    tensor, key, num_attention_heads, num_key_value_heads, head_dim, hidden_size, query_dim, key_value_dim
):
    """Permute projection weights for q_proj, k_proj, and v_proj."""
    if "q_proj" in key:
        if "weight" in key:
            tensor = tensor.view(num_attention_heads, head_dim, hidden_size).reshape(query_dim, hidden_size)
            tensor = permute_for_rope(tensor, num_attention_heads, query_dim, hidden_size)
        elif "bias" in key:
            tensor = tensor.view(num_attention_heads, head_dim).reshape(query_dim)
            tensor = permute_for_rope(tensor, num_attention_heads, query_dim)
    elif "k_proj" in key:
        if "weight" in key:
            tensor = tensor.view(num_key_value_heads, head_dim, hidden_size).reshape(key_value_dim, hidden_size)
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, hidden_size)
        elif "bias" in key:
            tensor = tensor.view(num_key_value_heads, head_dim).reshape(key_value_dim)
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim)
    elif "v_proj" in key:
        if "weight" in key:
            tensor = tensor.view(num_key_value_heads, head_dim, hidden_size).reshape(key_value_dim, hidden_size)
        elif "bias" in key:
            tensor = tensor.view(num_key_value_heads, head_dim).reshape(key_value_dim)

    return tensor


def write_model(
    input_path_or_repo,
    model_name,
    config_name,
    output_dir,
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    # --------------
    # convert config
    # --------------

    config_path = cached_file(input_path_or_repo, config_name)
    with open(config_path, "r") as f:
        original_config = json.load(f)

    config = convert_config(original_config)

    # ---------------
    # convert weights
    # ---------------

    model_path = cached_file(input_path_or_repo, model_name)
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    state_dict = load_file(model_path)
    print("Converting model...")
    converted_state_dict = convert_state_dict(state_dict, config)

    converted_state_dict["language_model.lm_head.weight"] = converted_state_dict[
        "language_model.model.embed_tokens.weight"
    ].clone()

    # -------------------------
    # load the weights and save
    # -------------------------

    print("Loading the checkpoint in a VoxtralRealtime model.")
    with torch.device("meta"):
        model = VoxtralRealtimeForConditionalGeneration(config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    model.tie_weights()
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    del model.generation_config._from_model_config
    model.generation_config.pad_token_id = 11

    print("Saving the model.")
    model.save_pretrained(output_dir)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    VoxtralRealtimeForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_processor(input_path_or_repo: str, output_dir: str):
    tokenizer = MistralCommonBackend.from_pretrained(input_path_or_repo)
    feature_extractor = VoxtralRealtimeFeatureExtractor()

    print("Creating the processor...")
    # Create the processor and save it
    processor = VoxtralRealtimeProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(output_dir)
    print("Processor saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Convert VoxtralRealtime weights to Hugging Face format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing the original weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the config in input_path_or_repo",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        args.config_name,
        args.output_dir,
    )

    write_processor(
        args.input_path_or_repo,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
