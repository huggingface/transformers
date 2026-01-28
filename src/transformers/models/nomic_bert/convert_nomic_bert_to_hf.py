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
    NomicBertConfig,
    NomicBertModel,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/nomic_bert/convert_nomic_bert_to_hf.py --original_model_id nomic-ai/nomic-embed-text-v1.5 --output_hub_path org/nomic_bert
"""


def get_config(checkpoint):
    base_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    if checkpoint == "nomic-ai/nomic-embed-text-v1.5":
        return NomicBertConfig(
            vocab_size=30528,
            rotary_emb_fraction=base_config.rotary_emb_fraction,
            rotary_emb_base=base_config.rotary_emb_base,
            rotary_emb_scale_base=base_config.rotary_emb_scale_base,
            rotary_emb_interleaved=base_config.rotary_emb_interleaved,
            type_vocab_size=base_config.type_vocab_size,
            pad_vocab_size_multiple=base_config.pad_vocab_size_multiple,
            tie_word_embeddings=base_config.tie_word_embeddings,
            max_position_embeddings=base_config.max_position_embeddings,
        )
    return base_config


def convert_nomic_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
    config = get_config(original_model_id)
    config.model_type = "nomic_bert"

    model = NomicBertModel.from_pretrained(
        original_model_id, config=config, trust_remote_code=True, ignore_mismatched_sizes=True
    )

    model.save_pretrained(output_hub_path)
    print(f"Model saved to {output_hub_path}")

    if push_to_hub:
        model.push_to_hub(output_hub_path, private=True)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--original_model_id",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Hub location of the model",
    )
    parser.add_argument(
        "--output_hub_path",
        default="org/nomic_bert",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, the model will be pushed to the hub after conversion.",
    )
    args = parser.parse_args()
    convert_nomic_hub_to_hf(args.original_model_id, args.output_hub_path, args.push_to_hub)


if __name__ == "__main__":
    main()
