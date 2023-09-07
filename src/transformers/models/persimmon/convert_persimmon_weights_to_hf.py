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

from transformers import PersimmonConfig, PersimmonForCausalLM, LlamaTokenizer


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path
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
        if keys_to_remove in key:
            continue
        model_state_dict[key] = value
    return model_state_dict
    
    
def convert_persimmon_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, enable_fusion=False):
    model_state_dict_base = torch.load(checkpoint_path, map_location="cpu")
    state_dict = flatdict.FlatDict(model_state_dict_base["model"], '.')
    state_dict = rename_state_dict(state_dict)
    
    transformers_config = PersimmonConfig()
    model = PersimmonModelForCausalLM(transformers_config).to(torch.bfloat16)
    model.save_pretrained(pytorch_dump_folder_path)
    transformers_config.save_pretrained(pytorch_dump_folder_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Persimmon weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    spm_path = os.path.join(args.input_dir, "tokenizer.model")

    convert_persimmon_checkpoint(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        safe_serialization=args.safe_serialization,
        tokenizer_path=spm_path,
    )
    tokenizer = LlamaTokenizer(spm_path, bos_token = "|ENDOFTEXT", eos_token = None)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
