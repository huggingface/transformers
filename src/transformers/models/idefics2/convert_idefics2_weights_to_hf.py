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
import copy

import torch
from accelerate import init_empty_weights

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    Idefics2ImageProcessor,
    Idefics2Processor,
    MistralConfig,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py --original_model_id HuggingFaceM4/idefics2-8b --output_hub_path org/idefics2
"""


KEYS_TO_MODIFY_MAPPING = {
    "lm_head.weight": "lm_head.linear.weight",
    "model.layers": "model.text_model.layers",
    "model.norm": "model.text_model.norm",
    "model.perceiver_resampler": "model.connector.perceiver_resampler",
    "model.modality_projection": "model.connector.modality_projection",
}


WEIGHTS_TO_MERGE_MAPPING = (
    # (weights to merge in merging order), (new weight name)
    (
        ("model.embed_tokens.weight", "model.embed_tokens.additional_embedding.weight"),
        "model.text_model.embed_tokens.weight",
    ),
    (("lm_head.linear.weight", "additional_fc.weight"), "lm_head.weight"),
)


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def merge_weights(state_dict):
    new_state_dict = copy.deepcopy(state_dict)

    # Merge the weights
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight in weights_to_merge:
            assert weight in state_dict, f"Weight {weight} is missing in the state dict"
            if new_weight_name not in new_state_dict:
                new_state_dict[new_weight_name] = [state_dict[weight]]
            else:
                new_state_dict[new_weight_name].append(state_dict[weight])
        new_state_dict[new_weight_name] = torch.cat(new_state_dict[new_weight_name], dim=0)

    # Remove the weights that were merged
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight in weights_to_merge:
            if weight in new_state_dict and weight != new_weight_name:
                new_state_dict.pop(weight)

    return new_state_dict


def get_config(checkpoint):
    if checkpoint == "HuggingFaceM4/idefics2":
        # We load the config then recreate to use the text_config
        config = AutoConfig.from_pretrained(checkpoint)
        text_config = MistralConfig(
            vocab_size=config.vocab_size + config.additional_vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            sliding_window=config.sliding_window,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
        perceiver_config = config.perceiver_config.to_dict()
        config = Idefics2Config(
            text_config=text_config.to_dict(),
            vision_config=config.vision_config,
            perceiver_config=perceiver_config,
            use_cache=config.use_cache,
            image_token_id=config.image_token_id,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        return config

    return AutoConfig.from_pretrained(checkpoint)


def convert_idefics2_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
    # The original model maps to AutoModelForCausalLM, converted we map to Idefics2ForConditionalGeneration
    original_model = AutoModelForCausalLM.from_pretrained(original_model_id, trust_remote_code=True)
    # The original model doesn't use the idefics2 processing objects
    image_seq_len = original_model.config.perceiver_config.resampler_n_latents
    image_processor = Idefics2ImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained(original_model_id)
    processor = Idefics2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_seq_len=image_seq_len,
    )
    state_dict = original_model.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)

    # Merge weights
    state_dict = merge_weights(state_dict)

    config = get_config(original_model_id)

    with init_empty_weights():
        model = Idefics2ForConditionalGeneration(config)

    model.load_state_dict(state_dict, strict=True, assign=True)

    model.save_pretrained(output_hub_path)
    processor.save_pretrained(output_hub_path)

    if push_to_hub:
        model.push_to_hub(output_hub_path, private=True)
        processor.push_to_hub(output_hub_path, private=True)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--original_model_id",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, the model will be pushed to the hub after conversion.",
    )
    args = parser.parse_args()
    convert_idefics2_hub_to_hf(args.original_model_id, args.output_hub_path, args.push_to_hub)


if __name__ == "__main__":
    main()
