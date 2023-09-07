# Copyright 2023 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import shutil
import warnings

import torch

from transformers import PersimmonConfig, PersimmonForCausalLM, PersimmonTokenizer


try:
    from transformers import PersimmonTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    PersimmonTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py \
    --input_dir /path/to/downloaded/persimmon/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

# ! need to git clone their repo to have the tool_use folder

import torch
import flatdict

# model_state_dict_chat = torch.load("/home/arthur_huggingface_co/adept-inference/weights/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt", map_location="cpu")
# config_chat = model_state_dict_chat["args"].__dict__
model_state_dict_base = torch.load("/home/arthur_huggingface_co/adept-inference/weights/8b_base_model_release/iter_0375000/mp_rank_00/model_optim_rng.pt", map_location="cpu")
config_base = model_state_dict_base["args"].__dict__


KEYS_TO_MODIFY_MAPPING = {
    "self_attention": "self_attn",
    "language_model.encoder": "model",
    "word_embeddings_for_head": "lm_head",
    "embedding": "embed_tokens",
}

# TO REMOVE: 
keys_to_remove = "rotary_emb.inv_freq"

new_dict = flatdict.FlatDict(model_state_dict_base["model"], '.')
out_state_dict = {}

def rename_state_dict(state_dict):
    model_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        model_state_dict[key] = value
    return model_state_dict
    
new_dict = rename_state_dict(new_dict)
torch.save(new_dict, "/home/arthur_huggingface_co/transformers/ArthurZ/persimmon-8b-base/pytorch_model.bin")
from transformers import PersimmonConfig, PersimmonForCausalLM, FalconForCausalLM


config = PersimmonConfig()
model = PersimmonForCausalLM(config = config)
model.load_state_dict(model_state_dict_base["model"])




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Persimmon weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "7Bf", "13B", "13Bf", "30B", "34B", "65B", "70B", "70Bf", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Persimmon2 official release. For more details on Persimmon2, checkout the original repo: https://huggingface.co/meta-persimmon",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            tokenizer_path=spm_path,
        )
    else:
        write_tokenizer(args.output_dir, spm_path)


if __name__ == "__main__":
    main()
