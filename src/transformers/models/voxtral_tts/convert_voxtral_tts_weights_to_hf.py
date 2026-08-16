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

from transformers import VoxtralTtsConfig, VoxtralTtsForTextToSpeech
from transformers.models.voxtral_tts.configuration_voxtral_tts import (
    VoxtralTtsCodecConfig,
    VoxtralTtsFlowMatchingConfig,
)
from transformers.utils.hub import cached_file


# fmt: off
STATE_DICT_MAPPING = {
    # ── Backbone decoder layers ──
    r"^layers\.(\d+)\.attention_norm\.weight":                                          r"backbone_model.layers.\1.input_layernorm.weight",
    r"^layers\.(\d+)\.ffn_norm\.weight":                                                r"backbone_model.layers.\1.post_attention_layernorm.weight",
    r"^layers\.(\d+)\.attention\.w(q|k|v|o)\.weight":                                   r"backbone_model.layers.\1.self_attn.\2_proj.weight",
    r"^layers\.(\d+)\.feed_forward\.w1\.weight":                                        r"backbone_model.layers.\1.mlp.gate_proj.weight",
    r"^layers\.(\d+)\.feed_forward\.w2\.weight":                                        r"backbone_model.layers.\1.mlp.down_proj.weight",
    r"^layers\.(\d+)\.feed_forward\.w3\.weight":                                        r"backbone_model.layers.\1.mlp.up_proj.weight",

    # ── Audio embeddings ──
    r"^mm_audio_embeddings\.tok_embeddings\.weight":                                    r"embed_text_tokens.weight",
    r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.weight":              r"backbone_model.embed_tokens.embed_audio_tokens.weight",

    # ── Backbone final norm ──
    r"^norm\.weight":                                                                   r"backbone_model.norm.weight",

    # ── Flow-matching transformer: projections ──
    r"^acoustic_transformer\.llm_projection\.weight":                                   r"flow_matching_transformer.llm_projection.weight",
    r"^acoustic_transformer\.time_projection\.weight":                                  r"flow_matching_transformer.time_projection.weight",
    r"^acoustic_transformer\.input_projection\.weight":                                 r"flow_matching_transformer.input_projection.weight",

    # ── Flow-matching transformer: layers ──
    r"^acoustic_transformer\.layers\.(\d+)\.attention_norm\.weight":                    r"flow_matching_transformer.layers.\1.input_layernorm.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.ffn_norm\.weight":                          r"flow_matching_transformer.layers.\1.post_attention_layernorm.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.attention\.w(q|k|v|o)\.weight":             r"flow_matching_transformer.layers.\1.self_attn.\2_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w1\.weight":                  r"flow_matching_transformer.layers.\1.mlp.gate_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w2\.weight":                  r"flow_matching_transformer.layers.\1.mlp.down_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w3\.weight":                  r"flow_matching_transformer.layers.\1.mlp.up_proj.weight",

    # ── Flow-matching transformer: norm & output heads ──
    r"^acoustic_transformer\.norm\.weight":                                             r"flow_matching_transformer.norm.weight",
    r"^acoustic_transformer\.semantic_codebook_output\.weight":                          r"flow_matching_transformer.semantic_codebook_output.weight",
    r"^acoustic_transformer\.acoustic_codebook_output\.weight":                          r"flow_matching_transformer.acoustic_codebook_output.weight",

    # ── Codec: conv blocks (weight-normalized) ──
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.conv\.parametrizations\.weight\.(original0|original1)":  r"codec_model.decoder_blocks.\1.conv.parametrizations.weight.\2",

    # ── Codec: transformer layers ──
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention\.w(q|k|v|o)\.weight":           r"codec_model.decoder_blocks.\1.layers.\2.self_attn.\3_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention\.(q_norm|k_norm)\.weight":      r"codec_model.decoder_blocks.\1.layers.\2.self_attn.\3.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention_norm\.weight":                  r"codec_model.decoder_blocks.\1.layers.\2.input_layernorm.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention_scale":                         r"codec_model.decoder_blocks.\1.layers.\2.self_attn_layer_scale",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w1\.weight":                r"codec_model.decoder_blocks.\1.layers.\2.mlp.gate_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w2\.weight":                r"codec_model.decoder_blocks.\1.layers.\2.mlp.down_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w3\.weight":                r"codec_model.decoder_blocks.\1.layers.\2.mlp.up_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.ffn_norm\.weight":                        r"codec_model.decoder_blocks.\1.layers.\2.post_attention_layernorm.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.ffn_scale":                               r"codec_model.decoder_blocks.\1.layers.\2.mlp_layer_scale",

    # ── Codec: output projection (weight-normalized) ──
    r"^audio_tokenizer\.output_proj\.conv\.parametrizations\.weight\.(original0|original1)":             r"codec_model.output_proj.conv.parametrizations.weight.\1",

    # ── Codec: quantizer (buffers) ──
    r"^audio_tokenizer\.quantizer\.semantic_codebook\.cluster_usage":                                    r"codec_model.quantizer.semantic_codebook.cluster_usage",
    r"^audio_tokenizer\.quantizer\.semantic_codebook\.embedding_sum":                                    r"codec_model.quantizer.semantic_codebook.embedding_sum",
}
# fmt: on


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def convert_config(original_config: dict) -> VoxtralTtsConfig:
    """Build a VoxtralTtsConfig from the original params.json dict."""
    multimodal = original_config.pop("multimodal")
    audio_model_args = multimodal["audio_model_args"]
    audio_tokenizer_args = multimodal["audio_tokenizer_args"]
    fm_args = audio_model_args["acoustic_transformer_args"]

    flow_matching_config = VoxtralTtsFlowMatchingConfig(
        input_dim=fm_args["input_dim"],
        hidden_size=fm_args["dim"],
        intermediate_size=fm_args["hidden_dim"],
        num_hidden_layers=fm_args["n_layers"],
        num_attention_heads=fm_args["n_heads"],
        num_key_value_heads=fm_args["n_kv_heads"],
        head_dim=fm_args["head_dim"],
        rope_theta=fm_args.get("rope_theta", 10000.0),
        sigma=fm_args.get("sigma", 1e-5),
        sigma_max=fm_args.get("sigma_max", 1.0),
        acoustic_dim=audio_model_args.get("n_acoustic_codebook", 36),
    )

    codec_config = VoxtralTtsCodecConfig(
        hidden_size=audio_tokenizer_args["dim"],
        intermediate_size=audio_tokenizer_args["hidden_dim"],
        num_attention_heads=audio_tokenizer_args["n_heads"],
        num_key_value_heads=audio_tokenizer_args.get("n_kv_heads", audio_tokenizer_args["n_heads"]),
        head_dim=audio_tokenizer_args["head_dim"],
        semantic_codebook_size=audio_tokenizer_args["semantic_codebook_size"],
        semantic_dim=audio_tokenizer_args["semantic_dim"],
        acoustic_codebook_size=audio_tokenizer_args["acoustic_codebook_size"],
        acoustic_dim=audio_tokenizer_args["acoustic_dim"],
        sampling_rate=audio_tokenizer_args["sampling_rate"],
        patch_size=audio_tokenizer_args.get("pretransform_patch_size", 240),
        patch_proj_kernel_size=audio_tokenizer_args.get("patch_proj_kernel_size", 7),
        conv_weight_norm=audio_tokenizer_args.get("conv_weight_norm", True),
        causal=audio_tokenizer_args.get("causal", True),
        qk_norm=audio_tokenizer_args.get("qk_norm", True),
        qk_norm_eps=audio_tokenizer_args.get("qk_norm_eps", 1e-6),
        rms_norm_eps=audio_tokenizer_args.get("norm_eps", 0.01),
        layer_scale=audio_tokenizer_args.get("layer_scale", True),
        layer_scale_init=audio_tokenizer_args.get("layer_scale_init", 0.01),
        decoder_transformer_lengths=_parse_int_list(
            audio_tokenizer_args.get("decoder_transformer_lengths_str", "2,2,2,2")
        ),
        decoder_conv_kernels=_parse_int_list(audio_tokenizer_args.get("decoder_convs_kernels_str", "3,4,4,4")),
        decoder_conv_strides=_parse_int_list(audio_tokenizer_args.get("decoder_convs_strides_str", "1,2,2,2")),
        channels=audio_tokenizer_args.get("channels", 1),
    )

    config = VoxtralTtsConfig(
        hidden_size=original_config["dim"],
        intermediate_size=original_config["hidden_dim"],
        num_hidden_layers=original_config["n_layers"],
        num_attention_heads=original_config["n_heads"],
        num_key_value_heads=original_config["n_kv_heads"],
        head_dim=original_config["head_dim"],
        vocab_size=original_config["vocab_size"],
        rms_norm_eps=original_config.get("norm_eps", 1e-5),
        rope_theta=original_config.get("rope_theta", 1000000.0),
        max_position_embeddings=original_config.get("max_position_embeddings", 128000),
        tie_word_embeddings=original_config.get("tied_embeddings", True),
        audio_token_id=audio_model_args.get("audio_token_id", 24),
        begin_audio_token_id=audio_model_args.get("begin_audio_token_id", 25),
        condition_dropped_token_id=audio_model_args.get("condition_dropped_token_id", 42),
        num_codebooks=audio_model_args.get("num_codebooks", 37),
        semantic_codebook_size=audio_model_args.get("semantic_codebook_size", 8192),
        acoustic_codebook_size=audio_model_args.get("acoustic_codebook_size", 21),
        n_acoustic_codebook=audio_model_args.get("n_acoustic_codebook", 36),
        sampling_rate=audio_model_args.get("sampling_rate", 24000),
        frame_rate=audio_model_args.get("frame_rate", 12.5),
        codec_config=codec_config,
        flow_matching_config=flow_matching_config,
    )

    return config


def map_old_key_to_new(old_key: str) -> str:
    """Map an original state dict key to the HF equivalent."""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        if n_replace > 0:
            return new_key
    raise ValueError(f"Key: {old_key!r} could not be mapped (check STATE_DICT_MAPPING).")


def permute_for_rope(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """Permute Q/K weight from Mistral interleaved RoPE format to HF split format."""
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_state_dict(original_state_dict: dict, config: VoxtralTtsConfig) -> dict:
    """Convert the original state dict to HF format with key remapping and Q/K permutation."""
    new_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "backbone_model.layers" in new_key and "self_attn" in new_key:
            tensor = _permute_backbone_attn(tensor, new_key, config)
        elif "flow_matching_transformer.layers" in new_key and "self_attn" in new_key:
            tensor = _permute_fm_attn(tensor, new_key, config)

        new_dict[new_key] = tensor

    return new_dict


def _permute_backbone_attn(tensor: torch.Tensor, key: str, config: VoxtralTtsConfig) -> torch.Tensor:
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    hidden_size = config.hidden_size

    if "q_proj" in key:
        tensor = permute_for_rope(tensor, n_heads, n_heads * head_dim, hidden_size)
    elif "k_proj" in key:
        tensor = permute_for_rope(tensor, n_kv_heads, n_kv_heads * head_dim, hidden_size)
    return tensor


def _permute_fm_attn(tensor: torch.Tensor, key: str, config: VoxtralTtsConfig) -> torch.Tensor:
    fm_config = config.flow_matching_config
    n_heads = fm_config.num_attention_heads
    n_kv_heads = fm_config.num_key_value_heads
    head_dim = fm_config.head_dim
    hidden_size = fm_config.hidden_size

    if "q_proj" in key:
        tensor = permute_for_rope(tensor, n_heads, n_heads * head_dim, hidden_size)
    elif "k_proj" in key:
        tensor = permute_for_rope(tensor, n_kv_heads, n_kv_heads * head_dim, hidden_size)
    return tensor


def write_model(
    input_path_or_repo: str,
    output_dir: str,
    model_name: str = "consolidated.safetensors",
    config_name: str = "params.json",
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    # ── Convert config ──
    config_path = cached_file(input_path_or_repo, config_name)
    with open(config_path, "r") as f:
        original_config = json.load(f)

    config = convert_config(original_config)

    # ── Convert weights ──
    model_path = cached_file(input_path_or_repo, model_name)
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    state_dict = load_file(model_path)
    print("Converting state dict...")
    converted = convert_state_dict(state_dict, config)

    if config.tie_word_embeddings:
        converted["lm_head.weight"] = converted["embed_text_tokens.weight"].clone()

    # ── Load and save ──
    print("Loading the checkpoint in a VoxtralTtsForTextToSpeech model.")
    with torch.device("meta"):
        model = VoxtralTtsForTextToSpeech(config)
    model.load_state_dict(converted, strict=True, assign=True)
    model.tie_weights()
    print("Checkpoint loaded successfully.")

    del model.config._name_or_path
    print("Saving the model.")
    model.save_pretrained(output_dir)
    del state_dict, model

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    VoxtralTtsForTextToSpeech.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def inspect_weights(input_path_or_repo: str, model_name: str = "consolidated.safetensors"):
    """Print all keys, shapes, and dtypes in the original checkpoint for debugging."""
    model_path = cached_file(input_path_or_repo, model_name)
    state_dict = load_file(model_path)
    total_params = 0
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        numel = tensor.numel()
        total_params += numel
        print(f"  {key:90s}  {str(list(tensor.shape)):25s}  {str(tensor.dtype):15s}  ({numel / 1e6:.1f}M)")
    print(f"\nTotal: {len(state_dict)} keys, {total_params / 1e6:.1f}M params")


def main():
    parser = argparse.ArgumentParser(description="Convert Voxtral-TTS weights to Hugging Face format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        default="mistralai/Voxtral-4B-TTS-2603",
        help="Path or repo containing the original weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="voxtral-tts-hf",
        help="Location to write converted HF model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="consolidated.safetensors",
        help="Name of the model file in the repo",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="params.json",
        help="Name of the config file in the repo",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print original weight keys and exit (for debugging)",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_weights(args.input_path_or_repo, args.model_name)
        return

    write_model(
        args.input_path_or_repo,
        args.output_dir,
        args.model_name,
        args.config_name,
    )


if __name__ == "__main__":
    main()
