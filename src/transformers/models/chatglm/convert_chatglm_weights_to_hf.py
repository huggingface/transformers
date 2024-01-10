# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoModelForCausalLM, ChatGlmConfig, ChatGlmForCausalLM


KEYS_TO_MODIFY_MAPPING = {
    "transformer.encoder.": "model.",
    "transformer.output_layer": "lm_head",
    "model.final_layernorm": "model.norm",
    "transformer.embedding.word_embeddings": "model.embed_tokens",
}

KEYS_TO_POP = ["transformer.rotary_pos_emb.inv_freq"]


def convert_state_dict_to_hf(state_dict):
    """
    Simple utility method that converts the old ChatGLM format to HF format
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value

    for key_to_pop in KEYS_TO_POP:
        new_state_dict.pop(key_to_pop, None)

    return new_state_dict


def load_and_convert(model_id, output_dir, push_to_hub=False):
    model_remote_code = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    # Get state dict and convert it
    state_dict = model_remote_code.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)

    # TODO: load the config directly from AutoConfig + trust_remote_code
    config = ChatGlmConfig(
        vocab_size=65024, hidden_size=4096, num_attention_heads=32, num_hidden_layers=28, intermediate_size=13696
    )

    with torch.device("meta"):
        model = ChatGlmForCausalLM(config)

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir, push_to_hub=push_to_hub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of old chatglm weights",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--push_to_hub", type=bool, help="Whether or not to push the converted weights to hub")
    args = parser.parse_args()
    load_and_convert(args.model_id, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
