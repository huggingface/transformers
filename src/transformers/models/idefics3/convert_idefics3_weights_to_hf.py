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
import json

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    Idefics3ImageProcessor,
    Idefics3Processor,
    LlamaConfig,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/idefics3/convert_idefics3_weights_to_hf.py --original_model_id HuggingFaceM4/idefics3-8b --output_hub_path org/idefics3
"""


KEYS_TO_MODIFY_MAPPING = {
    "lm_head.weight": "lm_head.linear.weight",
    "model.layers": "model.text_model.layers",
    "model.norm": "model.text_model.norm",
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

WEIGHTS_TO_DROP = (
    # The original model had a vision head, but this is never used
    "model.vision_model.head",
)


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    old_state_dict_keys = set(state_dict.keys())

    # Flattened list of weights to merge. We keep these in the original state dict to merge them later
    original_weights_to_merge = [w for weights in WEIGHTS_TO_MERGE_MAPPING for w in weights[0]]

    # for key, value in state_dict.items():
    for old_key in old_state_dict_keys:
        if old_key.endswith(".inv_freq") or any(w in old_key for w in WEIGHTS_TO_DROP):
            state_dict.pop(old_key)
            continue

        key = old_key
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        weight = state_dict.pop(old_key)
        if key in original_weights_to_merge:
            new_state_dict[key] = weight
            # Bit of a hack - we need to keep the original weights to merge them later
            state_dict[key] = weight
        else:
            new_state_dict[key] = weight

    return new_state_dict


def merge_weights(state_dict, new_state_dict):
    old_weight_names = set(state_dict.keys())

    # Merge the weights
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight_to_merge in weights_to_merge:
            print(weight_to_merge)
            assert weight_to_merge in state_dict, f"Weight {weight_to_merge} is missing in the state dict"

            weight = state_dict.pop(weight_to_merge)
            if new_weight_name not in new_state_dict:
                new_state_dict[new_weight_name] = [weight]
            else:
                new_state_dict[new_weight_name].append(weight)

            old_weight_names.remove(weight_to_merge)

        new_state_dict[new_weight_name] = torch.cat(new_state_dict[new_weight_name], dim=0)

    # Remove the weights that were merged
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight in weights_to_merge:
            if weight in new_state_dict and weight != new_weight_name:
                new_state_dict.pop(weight)

    return new_state_dict


def get_config(checkpoint):
    # We load the config then recreate to use the text_config

    # download the config file
    filepath = hf_hub_download(repo_id=checkpoint, filename="config.json")
    with open(filepath, "r") as f:
        config_json = json.load(f)

    # Setup the vision config
    vision_config = config_json.pop("vision_config")
    vision_config.pop("vision_model_name", None)
    if "embed_dim" in vision_config:
        vision_config["hidden_size"] = vision_config.pop("embed_dim")

    config_json["vocab_size"] = config_json.pop("vocab_size") + config_json.pop("additional_vocab_size")

    image_token_id = config_json.pop("image_token_id", config_json["vocab_size"] - 2)
    use_cache = config_json.pop("use_cache", True)
    tie_word_embeddings = config_json.pop("tie_word_embeddings", True)
    scale_factor = config_json.pop("scale_factor", 2)
    vocab_size = config_json.pop("vocab_size", 100000)

    # Remove "freeze" params from the config
    config_json = {k: v for k, v in config_json.items() if not k.startswith("freeze_")}
    text_config = LlamaConfig(**config_json)

    config = Idefics3Config(
        text_config=text_config,
        vision_config=vision_config,
        use_cache=use_cache,
        image_token_id=image_token_id,
        tie_word_embeddings=tie_word_embeddings,
        scale_factor=scale_factor,
        vocab_size=vocab_size,
    )
    return config


def convert_idefics3_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
    # The original model maps to AutoModelForCausalLM, converted we map to Idefics3ForConditionalGeneration
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    # The original model doesn't use the Idefics3 processing objects
    image_processor = Idefics3ImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained(original_model_id)
    processor = Idefics3Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    state_dict = original_model.state_dict()
    new_state_dict = convert_state_dict_to_hf(state_dict)

    # Merge weights
    new_state_dict = merge_weights(state_dict, new_state_dict)
    del state_dict

    config = get_config(original_model_id)
    print(config)

    with init_empty_weights():
        model = Idefics3ForConditionalGeneration(config)

    model.load_state_dict(new_state_dict, strict=True, assign=True)

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
    convert_idefics3_hub_to_hf(args.original_model_id, args.output_hub_path, args.push_to_hub)


if __name__ == "__main__":
    main()
