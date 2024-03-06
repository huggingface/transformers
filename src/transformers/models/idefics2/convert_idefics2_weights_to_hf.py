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

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Idefics2ForConditionalGeneration,
    Idefics2ImageProcessor,
    Idefics2Processor,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py --original_model_id HuggingFaceM4/idefics2 --output_hub_path org/idefics2
"""


KEYS_TO_MODIFY_MAPPING = {
    "lm_head.weight": "lm_head.linear.weight",
}


WEIGHTS_TO_MERGE_MAPPING = (
    # (weights to merge), (new weight name)
    (
        ("model.embed_tokens.weight", "model.embed_tokens.additional_embedding.weight"),
        "model.embed_tokens.weight",
    ),
    (("lm_head.linear.weight", "lm_head.additional_fc.weight"), "lm_head.weight"),
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
    new_state_dict = {}
    for key, value in state_dict.items():
        for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
            if key in weights_to_merge:
                if new_weight_name not in new_state_dict:
                    new_state_dict[new_weight_name] = value
                else:
                    new_state_dict[new_weight_name] = torch.cat((new_state_dict[new_weight_name], value), dim=0)
            else:
                new_state_dict[key] = value

    # Remove the weights that were merged
    for weights_to_merge, new_weight_name in WEIGHTS_TO_MERGE_MAPPING:
        for weight in weights_to_merge:
            if weight in new_state_dict and weight != new_weight_name:
                new_state_dict.pop(weight)

    return new_state_dict


def convert_idefics2_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
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

    # Pad embeddings to multiple of 64 for performance reasons
    for weight in ["model.embed_tokens.weight", "lm_head.weight"]:
        w = state_dict[weight]
        in_dim, out_dim = w.shape
        pad_size = 64 - in_dim % 64
        w = torch.cat((w, torch.zeros(pad_size, out_dim)), dim=0)
        state_dict[weight] = w

    config = AutoConfig.from_pretrained(original_model_id)

    model = Idefics2ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)

    model.save_pretrained(output_hub_path)
    processor.save_pretrained(output_hub_path)

    if push_to_hub
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
