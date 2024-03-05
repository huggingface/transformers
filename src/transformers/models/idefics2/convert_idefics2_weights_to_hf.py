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

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Idefics2ForConditionalGeneration,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py --original_model_id HuggingFaceM4/idefics2 --output_hub_path org/idefics2
"""


KEYS_TO_MODIFY_MAPPING = {
    "lm_head.weight": "lm_head.linear.weight",
}


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


def convert_idefics2_hub_to_hf(original_model_id, output_hub_path):
    original_model = AutoModelForCausalLM.from_pretrained(original_model_id, trust_remote_code=True)

    state_dict = original_model.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)

    config = AutoConfig.from_pretrained(original_model_id)

    model = Idefics2ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)

    model.save_pretrained(output_hub_path)
    # processor.save_pretrained(output_hub_path)

    # model.push_to_hub(output_hub_path)
    # processor.push_to_hub(output_hub_path)


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
    args = parser.parse_args()
    convert_idefics2_hub_to_hf(args.original_model_id, args.output_hub_path)


if __name__ == "__main__":
    main()
