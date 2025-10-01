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

import torch
from tokenizers.processors import TemplateProcessing

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    CsmConfig,
    CsmDepthDecoderConfig,
    CsmForConditionalGeneration,
    CsmProcessor,
    MimiModel,
)
from transformers.utils.hub import cached_file


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone\.layers\.(\d+)":                r"backbone_model.layers.\1",
    r"decoder\.layers\.(\d+)":            r"depth_decoder.model.layers.\1",

    r"attn":                                                  r"self_attn",
    r"output_proj":                                              r"o_proj",
    r"w1":                                                    r"gate_proj",
    r"w2":                                                    r"down_proj",
    r"w3":                                                      r"up_proj",

    r"text_embeddings":   r"embed_text_tokens",
    r"audio_embeddings": r"backbone_model.embed_tokens.embed_audio_tokens",

    r"codebook0_head":                                          r"lm_head",
    r"audio_head":                  r"depth_decoder.codebooks_head.weight",
    r"projection":          r"depth_decoder.model.inputs_embeds_projector",

    r"sa_norm.scale":                            r"input_layernorm.weight",
    r"mlp_norm.scale":                  r"post_attention_layernorm.weight",
    r"decoder.norm.scale":              r"depth_decoder.model.norm.weight",
    r"backbone.norm.scale":                  r"backbone_model.norm.weight",
}
# fmt: on


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def write_model(
    input_path_or_repo,
    model_name,
    codec_model_path_or_repo,
    output_dir,
    safe_serialization=True,
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    codec_model = MimiModel.from_pretrained(codec_model_path_or_repo)
    codec_model.config._attn_implementation_autoset = False

    # prepare rope scaling args: the model uses originally
    # 1 - for the depth decoder
    # rope_theta=500000,
    # rope_scaling={
    # 	"factor": 32.0,
    # 	"high_freq_factor": 4.0,
    # 	"low_freq_factor": 1.0,
    # 	"original_max_position_embeddings": 8192,
    # 	"rope_type": "llama3",
    # },
    # 2 - for the backbone
    # rope_theta=500000,
    # rope_scaling={
    # 	"factor": 32.0,
    # 	"high_freq_factor": 4.0,
    # 	"low_freq_factor": 1.0,
    # 	"original_max_position_embeddings": 8192,
    # 	"rope_type": "llama3",
    # },
    #
    # Yet we want to use max_position_embeddings=32, resp. 2048
    # This will throw warning as we would have original_max_position_embeddings >= max_position_embeddings
    # Therefore, we convert values to equivalent ones

    depth_decoder_config = CsmDepthDecoderConfig(
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 0.0078125,
            "low_freq_factor": 0.001953125,
            "original_max_position_embeddings": 16,
            "rope_type": "llama3",
        },
    )

    config = CsmConfig(
        codec_config=codec_model.config,
        depth_decoder_config=depth_decoder_config,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 0.5,
            "low_freq_factor": 0.125,
            "original_max_position_embeddings": 1024,
            "rope_type": "llama3",
        },
    )

    params = {
        "backbone": {
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "dim_per_head": config.head_dim,
            "key_value_dim": config.head_dim * config.num_key_value_heads,
            "dim": config.hidden_size,
        },
        "depth_decoder": {
            "num_attention_heads": config.depth_decoder_config.num_attention_heads,
            "num_key_value_heads": config.depth_decoder_config.num_key_value_heads,
            "dim_per_head": config.depth_decoder_config.head_dim,
            "key_value_dim": config.depth_decoder_config.head_dim * config.depth_decoder_config.num_key_value_heads,
            "dim": config.depth_decoder_config.hidden_size,
        },
    }

    model_path = cached_file(
        input_path_or_repo,
        model_name,
    )
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    loaded = torch.load(model_path, map_location="cpu")

    print("Converting model...")
    state_dict = {}

    # -----------------------
    # convert parameter names
    # -----------------------

    # Add codec_model. prefix to every key in the codec model state dict
    codec_state_dict = {f"codec_model.{k}": v for k, v in codec_model.state_dict().items()}
    state_dict.update(codec_state_dict)

    for key, value in loaded.items():
        new_key = convert_key(key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)
        current_parameter = value

        # Post-process the current_parameter.
        if re.search("(k|q)_proj.weight", new_key):
            params_keys = "backbone" if "backbone" in new_key else "depth_decoder"
            if "q_proj" in new_key:
                num_heads = params[params_keys]["num_attention_heads"]
                dim_per_head = params[params_keys]["dim_per_head"]
                param_dim = params[params_keys]["dim"]
                dim = params[params_keys]["dim"]
            else:
                num_heads = params[params_keys]["num_key_value_heads"]
                dim_per_head = params[params_keys]["dim_per_head"]
                param_dim = params[params_keys]["key_value_dim"]
                dim = params[params_keys]["dim"]

            current_parameter = permute_for_rope(value, num_heads, param_dim, dim)
            state_dict[new_key] = current_parameter.reshape(num_heads * dim_per_head, dim)

        state_dict[new_key] = current_parameter

    # add the depth decoder embed audio tokens weights, latter tied to the backbone embed audio tokens weights
    state_dict["depth_decoder.model.embed_tokens.weight"] = state_dict[
        "backbone_model.embed_tokens.embed_audio_tokens.weight"
    ].clone()
    del loaded
    gc.collect()

    # -------------------------
    # load the weights and save
    # -------------------------

    print("Loading the checkpoint in a Csm model.")
    with torch.device("meta"):
        model = CsmForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    # default generation config
    model.generation_config._from_model_config = False
    model.generation_config.max_new_tokens = 125
    model.generation_config.do_sample = True
    model.generation_config.top_k = 50
    model.generation_config.temperature = 0.9
    model.generation_config.depth_decoder_do_sample = True
    model.generation_config.depth_decoder_top_k = 50
    model.generation_config.depth_decoder_temperature = 0.9

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    CsmForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_tokenizer(output_dir):
    # from https://github.com/SesameAILabs/csm/blob/2d720827843b653c4d67bb4445b1c0a4f59e646f/generator.py#L22-L36
    def load_llama3_tokenizer():
        """
        https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
        """
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )

        return tokenizer

    tokenizer = load_llama3_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    # manually modify in tokenizer_config.json
    # "128002": {
    #     "content": "<|AUDIO|>",
    #     ...
    # }
    # "128003": {
    #     "content": "<|audio_eos|>",
    #     ...
    # }
    print(
        "Tokenizer saved successfully. Please manually modify in tokenizer_config.json AND tokenizer.json as follows: "
    )
    print("""
    # "128002": {
    #     "content": "<|AUDIO|>",
    #     ...
    # }
    # "128003": {
    #     "content": "<|audio_eos|>",
    #     ...
    # }
    """)


def write_processor(output_dir, codec_model_path_or_repo):
    chat_template = "\n{%- for message in messages %}\n    {#-- Validate role is a stringified integer --#}\n    {%- if not message['role'] is string or not message['role'].isdigit() %}\n        {{- raise_exception(\"The role must be an integer or a stringified integer (e.g. '0') designating the speaker id\") }}\n    {%- endif %}\n\n    {#-- Validate content is a list --#}\n    {%- set content = message['content'] %}\n    {%- if content is not iterable or content is string %}\n        {{- raise_exception(\"The content must be a list\") }}\n    {%- endif %}\n\n    {#-- Collect content types --#}\n    {%- set content_types = content | map(attribute='type') | list %}\n    {%- set is_last = loop.last %}\n\n    {#-- Last message validation --#}\n    {%- if is_last %}\n        {%- if 'text' not in content_types %}\n            {{- raise_exception(\"The last message must include one item of type 'text'\") }}\n        {%- elif (content_types | select('equalto', 'text') | list | length > 1) or (content_types | select('equalto', 'audio') | list | length > 1) %}\n            {{- raise_exception(\"At most two items are allowed in the last message: one 'text' and one 'audio'\") }}\n        {%- endif %}\n\n    {#-- All other messages validation --#}\n    {%- else %}\n        {%- if content_types | select('equalto', 'text') | list | length != 1\n              or content_types | select('equalto', 'audio') | list | length != 1 %}\n            {{- raise_exception(\"Each message (except the last) must contain exactly one 'text' and one 'audio' item\") }}\n        {%- elif content_types | reject('in', ['text', 'audio']) | list | length > 0 %}\n            {{- raise_exception(\"Only 'text' and 'audio' types are allowed in content\") }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n\n{%- for message in messages %}\n    {{- bos_token }}\n    {{- '[' + message['role'] + ']' }}\n    {{- message['content'][0]['text'] }}\n    {{- eos_token }}\n    {%- if message['content']|length > 1 %}\n        {{- '<|AUDIO|><|audio_eos|>' }}\n    {%- endif %}\n{%- endfor %}\n"
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(codec_model_path_or_repo)

    processor = CsmProcessor(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        chat_template=chat_template,
    )

    processor.save_pretrained(output_dir)
    print("Processor saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Convert Csm weights to HuggingFace format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing Csm weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--codec_model_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing the codec model",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        args.codec_model_path_or_repo,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    write_tokenizer(args.output_dir)

    write_processor(args.output_dir, args.codec_model_path_or_repo)


if __name__ == "__main__":
    main()
