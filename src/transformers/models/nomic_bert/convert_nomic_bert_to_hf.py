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

from transformers import AutoModelForMaskedLM
import torch
import argparse

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    NomicBertConfig,
    NomicBertModel,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/nomic_bert/convert_nomic_bert_to_hf.py --tokenizer_model_id bert-base-uncased --original_model_id nomic-ai/nomic-bert-2048 --output_hub_path org/nomic_bert
"""


# TODO write correct modify mappings
KEYS_TO_MODIFY_MAPPING = {
    "bert.embeddings.word_embeddings": "embeddings.word_embeddings",
    "bert.embeddings.position_embeddings": "embeddings.position_embeddings",
    "bert.embeddings.token_type_embeddings": "embeddings.token_type_embeddings",
    "bert.embeddings.LayerNorm": "embeddings.LayerNorm",
    "bert.encoder.layer": "encoder.layer",
    "bert.pooler.dense": "pooler.dense",
    "bert.pooler.LayerNorm": "pooler.LayerNorm",
    "cls.predictions.bias": "cls.predictions.bias",
    "cls.predictions.transform.dense": "cls.predictions.transform.dense",
    "cls.predictions.transform.LayerNorm": "cls.predictions.transform.LayerNorm",
}


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for old, new in KEYS_TO_MODIFY_MAPPING.items():
            if old in key:
                key = key.replace(old, new)
        new_state_dict[key] = value
    return new_state_dict


def get_config(checkpoint):
    base_config = AutoConfig.from_pretrained(checkpoint)
    if checkpoint == "nomic-ai/nomic-bert-2048":
        return NomicBertConfig(
            rotary_emb_fraction = base_config.rotary_emb_fraction,
            rotary_emb_base = base_config.rotary_emb_base,
            rotary_emb_scale_base = base_config.rotary_emb_scale_base,
            rotary_emb_interleaved = base_config.rotary_emb_interleaved,
            type_vocab_size = base_config.type_vocab_size,
            pad_vocab_size_multiple = base_config.pad_vocab_size_multiple,
            tie_word_embeddings = base_config.tie_word_embeddings,
            rotary_scaling_factor = base_config.rotary_scaling_factor,
            max_position_embeddings = base_config.max_position_embeddings,
        )

    return base_config


def convert_nomic_hub_to_hf(tokenizer_model_id, original_model_id, output_hub_path, push_to_hub):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)

    config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True) # the config needs to be passed in
    original_model = AutoModelForMaskedLM.from_pretrained(original_model_id,config=config, trust_remote_code=True)
    print(f'original keys: {original_model.state_dict().keys()}')


    config = get_config(original_model_id)

    with torch.device("meta"):
        model = NomicBertModel(config)

    print(f'model keys: {model.state_dict().keys()}')
    state_dict = original_model.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_hub_path)

    if push_to_hub:
        model.push_to_hub(output_hub_path, private=True)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer_model_id",
        default='bert-base-uncased',
        help="Hub location of the tokenizer model",
    )
    parser.add_argument(
        "--original_model_id",
        default='nomic-ai/nomic-bert-2048',
        help="Hub location of the model",
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
    convert_nomic_hub_to_hf(args.tokenizer_model_id, args.original_model_id, args.output_hub_path, args.push_to_hub)


if __name__ == "__main__":
    main()