# coding=utf-8
# Copyright 2025 TII and the HuggingFace Inc. team. All rights reserved.
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
"""This script can be used to convert checkpoints provided in the `mamba_ssm` library into the format provided in HuggingFace `transformers`. It depends on the `mamba2_ssm` package to be installed."""

import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, FalconH1Config, FalconH1ForCausalLM


CONVERSION_MAPPING = {
    "backbone": "model",
    "embeddings": "embed_tokens",
    "mixer.": "",
    "mixer_ssm": "mamba",
    "mixer_attn": "self_attn",
    "mlp.": "feed_forward.",
    "mlp_norm": "pre_ff_layernorm",
    "ssm_proj": "mamba.in_proj",
    "attn_out_proj": "o_proj",
    ".norm.": ".input_layernorm.",
    ".mamba.input_layernorm.": ".mamba.norm.",
    ".ssm_out_proj.": ".mamba.out_proj.",
    "norm_f": "final_layernorm",
}


def convert_falcon_h1_to_hf(input_model_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    model = AutoModelForCausalLM.from_pretrained(input_model_path, dtype=torch.bfloat16, trust_remote_code=True)

    intermediate_size = int(model.config.expansion_factor * model.config.hidden_size)

    if intermediate_size % 2 != 0:
        intermediate_size = intermediate_size + (intermediate_size % 2)

    new_config = FalconH1Config(
        vocab_size=model.config.vocab_size,
        tie_word_embeddings=model.config.tie_word_embeddings,
        hidden_size=model.config.hidden_size,
        intermediate_size=intermediate_size,
        mamba_d_state=model.config.state_size,
        num_hidden_layers=model.config.num_hidden_layers,
        mamba_use_mlp=model.config.use_mlp,
        rms_norm_eps=model.config.layer_norm_epsilon,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        mamba_expand=model.config.expand,
        mamba_d_conv=model.config.conv_kernel,
        mamba_n_groups=model.config.n_groups,
        mamba_n_heads=model.config.num_heads,
        mamba_norm_before_gate=model.config.norm_before_gate,
        mamba_rms_norm=model.config.rms_norm,
        mamba_d_ssm=model.config.d_ssm,
        attention_bias=model.config.use_bias,
        projectors_bias=model.config.use_bias,
        mamba_conv_bias=model.config.use_conv_bias,
        hidden_act=model.config.hidden_act,
        use_cache=model.config.use_cache,
        mamba_chunk_size=model.config.chunk_size,
        num_attention_heads=model.config.num_heads_mha,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim_mha,
        lm_head_multiplier=model.config.lm_head_multiplier,
        embedding_multiplier=model.config.embedding_multiplier,
        mlp_multipliers=model.config.mlp_multipliers,
        key_multiplier=model.config.key_multiplier,
        attention_out_multiplier=model.config.attention_out_multiplier,
        attention_in_multiplier=model.config.attention_in_multiplier,
        ssm_multipliers=model.config.ssm_multipliers,
        ssm_in_multiplier=model.config.ssm_in_multiplier,
        ssm_out_multiplier=model.config.ssm_out_multiplier,
        rope_theta=model.config.rope_theta,
    )

    old_state_dict = model.state_dict()
    new_state_dict = {}

    for old_key, old_value in old_state_dict.items():
        new_key = old_key
        for conversion_key, conversion_value in CONVERSION_MAPPING.items():
            if conversion_key in old_key:
                new_key = new_key.replace(conversion_key, conversion_value)

        if "mamba.input_layernorm" in new_key:
            new_key = new_key.replace("mamba.input_layernorm", "mamba.norm")

        # Special processing for attention layers
        if "self_attn.attn_proj" in new_key:
            num_heads = new_config.num_attention_heads
            num_kv_heads = new_config.num_key_value_heads
            head_dim = new_config.head_dim
            q_proj, k_proj, v_proj = old_value.split(
                [
                    num_heads * head_dim,
                    num_kv_heads * head_dim,
                    num_kv_heads * head_dim,
                ],
                dim=0,
            )
            new_state_dict[new_key.replace("attn_proj", "q_proj")] = q_proj
            new_state_dict[new_key.replace("attn_proj", "k_proj")] = k_proj
            new_state_dict[new_key.replace("attn_proj", "v_proj")] = v_proj
        else:
            new_state_dict[new_key] = old_value

    with torch.device("meta"):
        new_model = FalconH1ForCausalLM(new_config)

    del model

    new_model.load_state_dict(new_state_dict, strict=True, assign=True)

    new_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_ssm_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_falcon_h1_to_hf(
        args.mamba_ssm_checkpoint_directory,
        args.output_dir,
    )
