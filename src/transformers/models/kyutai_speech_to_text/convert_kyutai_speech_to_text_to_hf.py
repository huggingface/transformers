# coding=utf-8
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
import os
import re

import safetensors.torch
import sentencepiece
import torch

from transformers import (
    KyutaiSpeechToTextConfig,
    KyutaiSpeechToTextFeatureExtractor,
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import MoshiConverter
from transformers.utils.hub import cached_file


# fmt: off
MOSHI_ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"out_norm":                                                r"norm",
    r"gating\.linear_in":                              r"mlp.fc1",
    r"gating\.linear_out":                             r"mlp.fc2",
    r"self_attn\.out_proj":                r"self_attn.o_proj.linear",
    r"norm1":                                      r"input_layernorm",
    r"norm2":                              r"post_attention_layernorm",
    r"layer_scale_1":                          r"self_attn_layer_scale",
    r"layer_scale_2":                             r"mlp_layer_scale",
    r"alpha":                                              r"weight",
}
# fmt: on


# fmt: off
MIMI_ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"conv\.conv\.conv": "conv",
    r"convtr\.convtr\.convtr": "conv",
    r"conv\.conv": "conv",
    r"convtr\.convtr": "conv",
    r"quantizer\.rvq_first\.vq": "quantizer.semantic_residual_vector_quantizer",
    r"quantizer\.rvq_first": "quantizer.semantic_residual_vector_quantizer",
    r"quantizer\.rvq_rest\.vq": "quantizer.acoustic_residual_vector_quantizer",
    r"quantizer\.rvq_rest": "quantizer.acoustic_residual_vector_quantizer",
    r"_codebook": "codebook",
    r"_initialized": "initialized",
    r"embedding_sum": "embed_sum",
    r"encoder\.model": "encoder.layers",
    r"decoder\.model": "decoder.layers",
    r"encoder_transformer\.transformer": "encoder_transformer",
    r"decoder_transformer\.transformer": "decoder_transformer",
    r"linear1": "mlp.fc1",
    r"linear2": "mlp.fc2",
    r"self_attn\.out_proj": "self_attn.o_proj",
    r"norm1": "input_layernorm",
    r"norm2": "post_attention_layernorm",
    r"layer_scale_1": "self_attn_layer_scale",
    r"layer_scale_2": "mlp_layer_scale",
}
# fmt: on


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    return input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def convert_kyutai_speech_to_text_state_dict(state_dict, config, unwanted_prefix="transformer."):
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = int(config.hidden_size // config.head_dim)
    num_key_value_heads = config.num_key_value_heads
    key_value_head_dim = config.num_key_value_heads * head_dim

    # concat embeddings
    embed_tokens_weight = []
    for i in range(32):
        embed_tokens_weight.append(state_dict.pop(f"emb.{i}.weight"))

    embed_tokens_weight = torch.cat(embed_tokens_weight, dim=0)
    embed_tokens_weight = torch.cat([state_dict.pop("text_emb.weight"), embed_tokens_weight])
    embed_tokens_weight = torch.cat([embed_tokens_weight, torch.zeros(1, config.hidden_size)], dim=0)
    state_dict["embed_tokens.embed_tokens.weight"] = embed_tokens_weight

    for key, value in list(state_dict.items()):
        if unwanted_prefix is not None and unwanted_prefix in key:
            new_key = key[len(unwanted_prefix) :]
        else:
            new_key = key

        new_key = convert_key(new_key, MOSHI_ORIGINAL_TO_CONVERTED_KEY_MAPPING)

        # Post-process the current_parameter.
        if "alpha" in key:
            state_dict[key] = state_dict[key].squeeze()

        if "in_proj_weight" in new_key:
            # split qkv into query key and value
            mixed_qkv = state_dict.pop(key)
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]
            state_dict[new_key.replace("in_proj_weight", "q_proj.linear.weight")] = permute_for_rope(
                query_layer, num_heads, hidden_size, hidden_size
            )
            state_dict[new_key.replace("in_proj_weight", "k_proj.linear.weight")] = permute_for_rope(
                key_layer, num_key_value_heads, key_value_head_dim, hidden_size
            )

            state_dict[new_key.replace("in_proj_weight", "v_proj.linear.weight")] = value_layer
        else:
            state_dict[new_key] = state_dict.pop(key)

    return state_dict


def convert_mimi_state_dict(state_dict, config, unwanted_prefix=None):
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = int(config.hidden_size // config.head_dim)
    num_key_value_heads = config.num_key_value_heads
    key_value_head_dim = config.num_key_value_heads * head_dim

    for key, value in list(state_dict.items()):
        if unwanted_prefix is not None and unwanted_prefix in key:
            new_key = key[len(unwanted_prefix) :]
        else:
            new_key = key

        new_key = convert_key(new_key, MIMI_ORIGINAL_TO_CONVERTED_KEY_MAPPING)

        if "in_proj_weight" in new_key:
            # split qkv into query key and value
            mixed_qkv = state_dict.pop(key)
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            state_dict[new_key.replace("in_proj_weight", "q_proj.weight")] = permute_for_rope(
                query_layer, num_heads, hidden_size, hidden_size
            )
            state_dict[new_key.replace("in_proj_weight", "k_proj.weight")] = permute_for_rope(
                key_layer, num_key_value_heads, key_value_head_dim, hidden_size
            )
            state_dict[new_key.replace("in_proj_weight", "v_proj.weight")] = value_layer
        else:
            state_dict[new_key] = state_dict.pop(key)

    return state_dict


def write_model(
    input_path_or_repo,
    model_name,
    codec_model_path_or_repo,
    codec_model_name,
    output_dir,
    safe_serialization=True,
    unwanted_prefix="transformer.",
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    config = KyutaiSpeechToTextConfig(
        vocab_size=8001,
        max_position_embeddings=375,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
    )
    config.use_cache = True
    config.codec_config.sliding_window = 250

    model_path = cached_file(
        input_path_or_repo,
        model_name,
    )

    codec_path = cached_file(
        codec_model_path_or_repo,
        codec_model_name,
    )

    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    state_dict = safetensors.torch.load_file(model_path)

    print(f"Fetching all parameters from the checkpoint at {codec_path}...")
    codec_state_dict = safetensors.torch.load_file(codec_path)

    print("Converting model...")
    # -----------------------
    # convert parameter names
    # -----------------------
    state_dict = convert_kyutai_speech_to_text_state_dict(state_dict, config, unwanted_prefix=unwanted_prefix)
    codec_state_dict = convert_mimi_state_dict(codec_state_dict, config.codec_config, unwanted_prefix=None)

    # -------------------------
    # load the weights and save
    # -------------------------
    print("Loading the checkpoint in a Moshi ASR model.")
    with torch.device("meta"):
        model = KyutaiSpeechToTextForConditionalGeneration(config)

    linear_weight = state_dict.pop("text_linear.weight")
    model.model.load_state_dict(state_dict, strict=True, assign=True)

    linear_weight = torch.cat([linear_weight, torch.zeros(1, config.hidden_size)])
    model.lm_head.load_state_dict({"weight": linear_weight}, strict=True, assign=True)

    model.codec_model.load_state_dict(codec_state_dict, strict=True, assign=True)

    print("Checkpoint loaded successfully.")
    del model.config._name_or_path
    del model.config.codec_config._name_or_path

    # default generation config
    model.generation_config._from_model_config = False
    model.generation_config.audio_window_size = 1
    model.generation_config.cache_implementation = "sliding_window"

    model.codec_model.generation_config._from_model_config = False
    model.codec_model.generation_config.cache_implementation = "sliding_window"
    model.codec_model.generation_config.use_cache = True

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
        output_dir, dtype=torch.bfloat16, device_map="auto"
    )
    print("Model reloaded successfully.")


def write_processor(
    input_path_or_repo,
    tokenizer_model_name,
    codec_model_path_or_repo,
    output_dir,
    audio_delay_seconds,
    audio_silence_prefix_seconds,
):
    tokenizer_path = cached_file(
        input_path_or_repo,
        tokenizer_model_name,
    )

    tokenizer = MoshiConverter(tokenizer_path).converted()
    original_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        chat_template=None,
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
        clean_up_tokenization_spaces=False,
        bos_token_id=original_tokenizer.bos_id(),
        eos_token_id=original_tokenizer.eos_id(),
        pad_token_id=original_tokenizer.pad_id(),
    )

    feature_extractor = KyutaiSpeechToTextFeatureExtractor(
        audio_delay_seconds=audio_delay_seconds,
        audio_silence_prefix_seconds=audio_silence_prefix_seconds,
    )

    processor = KyutaiSpeechToTextProcessor(feature_extractor, tokenizer)
    processor.save_pretrained(output_dir)
    print(f"Processor saved successfully to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Moshi ASR weights to HuggingFace format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing Moshi ASR weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--tokenizer_model_name",
        type=str,
        required=True,
        help="Name of the tokenizer model in input_path_or_repo",
    )
    parser.add_argument(
        "--codec_model_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing the Mimi weights",
    )
    parser.add_argument(
        "--mimi_name",
        type=str,
        required=True,
        help="Name of the Mimi model in codec_model_path_or_repo",
    )
    parser.add_argument(
        "--preprocessor_model_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing the preprocessor config",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--audio_delay_seconds",
        type=float,
        required=True,
        help="Audio delay in seconds to add to the right of the input",
    )
    parser.add_argument(
        "--audio_silence_prefix_seconds",
        type=float,
        required=True,
        help="Audio silence prefix in seconds to add to the left of the input",
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        args.codec_model_path_or_repo,
        args.mimi_name,
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    write_processor(
        args.input_path_or_repo,
        args.tokenizer_model_name,
        args.preprocessor_model_path_or_repo,
        args.output_dir,
        args.audio_delay_seconds,
        args.audio_silence_prefix_seconds,
    )


if __name__ == "__main__":
    main()
