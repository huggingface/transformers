# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import (
    GenerationConfig,
    Ministral3Config,
    Ministral3ForCausalLM,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessorFast,
    PixtralProcessor,
    PixtralVisionConfig,
)
from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
from transformers.integrations.mistral import convert_tekken_tokenizer
from transformers.quantizers.auto import AutoQuantizationConfig


# fmt: off
def get_sd_mapping(has_vision: bool) -> dict:
    model_key = "model.language_model" if has_vision else "model"
    return {
        # Text model keys
        r"^output.weight":                            r"lm_head.weight",
        r"^norm.weight":                              rf"{model_key}.norm.weight",
        r"^tok_embeddings.weight":                    rf"{model_key}.embed_tokens.weight",
        r"^layers.(\d+).attention_norm.weight":       rf"{model_key}.layers.\1.input_layernorm.weight",
        r"^layers.(\d+).ffn_norm.weight":             rf"{model_key}.layers.\1.post_attention_layernorm.weight",
        r"^layers.(\d+).attention.w(q|k|v|o).weight": rf"{model_key}.layers.\1.self_attn.\2_proj.weight",
        r"^layers.(\d+).feed_forward.w1.weight":      rf"{model_key}.layers.\1.mlp.gate_proj.weight",
        r"^layers.(\d+).feed_forward.w2.weight":      rf"{model_key}.layers.\1.mlp.down_proj.weight",
        r"^layers.(\d+).feed_forward.w3.weight":      rf"{model_key}.layers.\1.mlp.up_proj.weight",
        r"^layers.(\d+).attention.w(q|k|v|o).qscale_act": rf"{model_key}.layers.\1.self_attn.\2_proj.activation_scale",
        r"^layers.(\d+).feed_forward.w1.qscale_act":      rf"{model_key}.layers.\1.mlp.gate_proj.activation_scale",
        r"^layers.(\d+).feed_forward.w2.qscale_act":      rf"{model_key}.layers.\1.mlp.down_proj.activation_scale",
        r"^layers.(\d+).feed_forward.w3.qscale_act":      rf"{model_key}.layers.\1.mlp.up_proj.activation_scale",
        r"^layers.(\d+).attention.w(q|k|v|o).qscale_weight": rf"{model_key}.layers.\1.self_attn.\2_proj.weight_scale_inv",
        r"^layers.(\d+).feed_forward.w1.qscale_weight":      rf"{model_key}.layers.\1.mlp.gate_proj.weight_scale_inv",
        r"^layers.(\d+).feed_forward.w2.qscale_weight":      rf"{model_key}.layers.\1.mlp.down_proj.weight_scale_inv",
        r"^layers.(\d+).feed_forward.w3.qscale_weight":      rf"{model_key}.layers.\1.mlp.up_proj.weight_scale_inv",

        # Vision model keys
        r"vision_encoder.transformer.layers.(\d+).attention_norm.weight": r"model.vision_tower.transformer.layers.\1.attention_norm.weight",
        r"^vision_encoder.transformer.layers.(\d+).ffn_norm.weight": r"model.vision_tower.transformer.layers.\1.ffn_norm.weight",
        r"^vision_encoder.transformer.layers.(\d+).attention.w(q|k|v|o).weight": r"model.vision_tower.transformer.layers.\1.attention.\2_proj.weight",
        r"^vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight": r"model.vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight",
        r"^vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight": r"model.vision_tower.transformer.layers.\1.feed_forward.down_proj.weight",
        r"^vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight": r"model.vision_tower.transformer.layers.\1.feed_forward.up_proj.weight",
        r"^vision_language_adapter.w_in": r"model.multi_modal_projector.linear_1",
        r"^vision_language_adapter.w_out": r"model.multi_modal_projector.linear_2",
        r"^vision_encoder.ln_pre.weight": r"model.vision_tower.ln_pre.weight",
        r"^vision_encoder.patch_conv.weight": r"model.vision_tower.patch_conv.weight",
        r"^patch_merger.merging_layer.weight": r"model.multi_modal_projector.patch_merger.merging_layer.weight",
        r"^pre_mm_projector_norm.weight": r"model.multi_modal_projector.norm.weight",
    }
# fmt: on


def map_old_key_to_new(old_key, mapping):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in mapping.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def permute_for_rope(tensor, n_heads, dim1, dim2):
    """Permute the weights for the ROPE formulation."""
    tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(dim1, dim2)
    return tensor


def convert_state_dict(original_state_dict: dict, config: Mistral3Config):
    """Convert a state dict file, when a single `nn.Module` is never sharded in different files (usual case)."""
    new_dict = {}

    is_vision = isinstance(config, Mistral3Config)
    mapping = get_sd_mapping(is_vision)
    for old_key, tensor in original_state_dict.items():
        if "fake_quantizer" in old_key:
            continue

        new_key = map_old_key_to_new(old_key, mapping)

        if "vision" in old_key:
            num_attention_heads = config.vision_config.num_attention_heads
            num_key_value_heads = num_attention_heads
            hidden_size = config.vision_config.hidden_size
            head_dim = config.vision_config.head_dim
            key_value_dim = head_dim * num_attention_heads
            query_dim = head_dim * num_attention_heads
        else:
            text_config = config.text_config if is_vision else config
            num_attention_heads = text_config.num_attention_heads
            hidden_size = text_config.hidden_size
            head_dim = text_config.head_dim
            num_key_value_heads = text_config.num_key_value_heads
            key_value_dim = head_dim * num_key_value_heads
            query_dim = head_dim * num_attention_heads

        if "q_proj" in new_key and new_key.endswith("weight"):
            tensor = permute_for_rope(tensor, num_attention_heads, query_dim, hidden_size)
        elif "k_proj" in new_key and new_key.endswith("weight"):
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, hidden_size)

        new_dict[new_key] = tensor
    return new_dict


def convert_config(original_config: dict, max_position_embeddings: int = 262144, is_vision: bool = True):
    original_vision_config = original_config.pop("vision_encoder", None)
    assert is_vision == (original_vision_config is not None), (
        f"is_vision={is_vision} but original_vision_config={original_vision_config}"
    )
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
    ]

    new_text_config_kwargs = {k: original_text_config[v] for k, v in text_key_mapping.items()}
    new_text_config_kwargs.update({k: v for k, v in original_text_config.items() if k in similar_text_keys_to_keep})
    tie_word_embeddings = original_text_config.get("tied_embeddings", False)
    new_text_config_kwargs["tie_word_embeddings"] = tie_word_embeddings
    new_text_config_kwargs["rope_parameters"] = {
        "type": "yarn",
        "rope_theta": original_config.get("rope_theta", 1000000.0),
        "factor": float(original_config["yarn"]["factor"]),
        "original_max_position_embeddings": original_config["yarn"]["original_max_position_embeddings"],
        "beta_fast": float(original_config["yarn"]["beta"]),
        "beta_slow": float(original_config["yarn"]["alpha"]),
        "mscale_all_dim": 1.0 if is_vision else 0.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": original_config.get("llama_4_scaling", {}).get("beta", 0),
    }

    # These are not always defined depending on `params.json`
    new_text_config_kwargs["sliding_window"] = original_text_config.get("sliding_window", None)
    new_text_config_kwargs["max_position_embeddings"] = original_text_config.get(
        "max_position_embeddings", original_text_config.get("max_seq_len", max_position_embeddings)
    )
    # This may sometimes be a string in `params.json`
    if new_text_config_kwargs["sliding_window"] is not None:
        new_text_config_kwargs["sliding_window"] = int(new_text_config_kwargs["sliding_window"])

    def get_maybe_quant_config() -> dict:
        kwargs = {}
        if original_config.get("quantization", {}).get("qformat_weight") == "fp8_e4m3":
            assert original_config["quantization"]["qscheme_act"] == "TENSOR"
            quantization_config = {
                "activation_scheme": "static",
                "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"],
                "quant_method": "fp8",
                "weight_block_size": None,
            }
            kwargs["quantization_config"] = AutoQuantizationConfig.from_dict(quantization_config)
        return kwargs

    # No vision
    if original_vision_config is None:
        new_text_config = Ministral3Config(**new_text_config_kwargs, **get_maybe_quant_config())
        return new_text_config
    else:
        new_text_config = Ministral3Config(**new_text_config_kwargs)

    # Vision config
    new_vision_config = original_vision_config
    adapter_bias = new_vision_config.pop("adapter_bias", False)
    _ = new_vision_config.pop("mm_projector_id", None)
    _ = new_vision_config.pop("add_pre_mm_projector_layer_norm", None)
    spatial_merge_size = new_vision_config.pop("spatial_merge_size")
    image_token_id = new_vision_config.pop("image_token_id", 10)
    _ = new_vision_config.pop("image_break_token_id", 12)
    _ = new_vision_config.pop("image_end_token_id", 13)
    _ = new_vision_config.pop("max_image_size")
    new_vision_config = PixtralVisionConfig(hidden_act="silu", **new_vision_config)

    new_config = Mistral3Config(
        vision_config=new_vision_config,
        text_config=new_text_config,
        multimodal_projector_bias=adapter_bias,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
        vision_feature_layer=-1,
        **get_maybe_quant_config(),
    )
    return new_config


def convert_and_write_model(input_dir: str, output_dir: str, max_position_embeddings: int):
    """Convert the model and save it (this implicitly save the config as well)."""
    params = read_json(os.path.join(input_dir, "params.json"))

    is_vision = params.get("vision_encoder") is not None
    config = convert_config(params, max_position_embeddings, is_vision)

    full_state_dict = {}
    # The model may be split between different files, but a single nn.Module is always fully present in a single file
    shards = [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
    for shard_file in shards:
        original_state_dict = load_file(os.path.join(input_dir, shard_file))
        new_dict = convert_state_dict(original_state_dict, config)
        full_state_dict.update(new_dict)

    text_config = config.text_config if is_vision else config
    if text_config.tie_word_embeddings:
        model_key = "model.language_model" if is_vision else "model"
        full_state_dict["lm_head.weight"] = full_state_dict[f"{model_key}.embed_tokens.weight"]

    # Load weights into model and resave them
    with torch.device("meta"):
        if isinstance(config, Mistral3Config):
            model = Mistral3ForConditionalGeneration(config)
        elif isinstance(config, Ministral3Config):
            model = Ministral3ForCausalLM(config)
        else:
            raise ValueError(f"Unknown config type {type(config)}.")

        # let's swap nn.Linear to FP8 Linear before loading
        if hasattr(model.config, "quantization_config"):
            model = replace_with_fp8_linear(
                model, model.config.quantization_config.modules_to_not_convert, model.config.quantization_config
            )

    model.load_state_dict(full_state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)
    return config


def convert_and_write_processor_and_tokenizer(
    input_dir: str, output_dir: str, model_config: Mistral3Config | Ministral3ForCausalLM
):
    """Convert the tokenizer and save it."""
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tokenizer_file = os.path.join(input_dir, "tekken.json")
    tokenizer = convert_tekken_tokenizer(tokenizer_file)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # No vision
    if isinstance(model_config, Ministral3Config):
        tokenizer.save_pretrained(output_dir)
        return

    tekkenizer = Tekkenizer.from_file(tokenizer_file)
    config = read_json(os.path.join(input_dir, "params.json"))
    patch_size = config["vision_encoder"]["patch_size"]
    spatial_merge_size = config["vision_encoder"]["spatial_merge_size"]
    max_image_size = config["vision_encoder"]["max_image_size"]
    image_processor = PixtralImageProcessorFast(patch_size=patch_size, size={"longest_edge": max_image_size})

    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="[IMG]",
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )

    # Finally save it
    processor.save_pretrained(output_dir)

    generation_config = GenerationConfig(
        eos_token_id=tekkenizer.eos_id,
        bos_token_id=tekkenizer.bos_id,
        pad_token_id=tekkenizer.pad_id,
        max_length=model_config.text_config.max_position_embeddings,
    )

    generation_config.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=262144,
        help="`max_position_embeddings` field in the config. This needs to be manually passed (not present anywhere otherwise).",
    )

    args = parser.parse_args()

    config = convert_and_write_model(args.input_dir, args.output_dir, args.max_position_embeddings)
    convert_and_write_processor_and_tokenizer(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
