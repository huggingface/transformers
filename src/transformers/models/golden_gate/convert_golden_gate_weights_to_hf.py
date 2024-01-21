# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import os
import warnings

import torch

from transformers import GoldenGateConfig, GoldenGateForCausalLM, LlamaTokenizer


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
python src/transformers/models/golden_gate/convert_golden_gate_weights_to_hf.py \
    --input_dir /path/to/downloaded/golden_gate/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import GoldenGateForCausalLM, LlamaTokenizer

model = GoldenGateForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

golden_gate_2b_config = GoldenGateConfig(
    hidden_size=2048,
    intermediate_size=16384,
    num_attention_heads=8,
    num_hidden_layers=18,
    rms_norm_eps=1e-6,
    num_key_value_heads=1,
    vocab_size=256128,
    head_dim=256,
    max_position_embeddings=8192,
    rope=10000,
    hidden_act = "gelu"
)

golden_gate_7b_config = GoldenGateConfig(
    hidden_size=3072,
    intermediate_size=24576,
    num_attention_heads=16,
    num_hidden_layers=28,
    rms_norm_eps= 1e-6,
    num_key_value_heads=16,
    vocab_size=256128,
    head_dim=256,
    max_position_embeddings=8192,
    rope=10000,
    hidden_act = "gelu"
)

CONFIG_MAPPING = {"2B":golden_gate_2b_config,"7B":golden_gate_7b_config}

LAYER_NAME_MAPPING = {
    "embedder.weight":"model.embed_tokens.weight"
}   

def write_model(save_path, input_base_path, config, safe_serialization=True):
    num_attn_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # permute for sliced rotary
    def permute(w, n_heads=num_attn_heads, dim1=hidden_size, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    model_state_dict = torch.load(os.path.join(input_base_path), map_location="cpu")["model_state_dict"]
    model_state_dict.pop("freqs_cis")

    state_dict = {}
    for k,v in model_state_dict.items():
        if "qkv_proj" in k:            
            if num_kv_heads == 1:
                v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim, hidden_size)
                q_proj = v[:num_attn_heads, ...]
                k_proj = v[num_attn_heads:num_attn_heads + num_kv_heads, ...].repeat(num_kv_heads, 1, 1)
                v_proj = v[-num_kv_heads:, ...].repeat(num_kv_heads, 1, 1)
            else:
                q_proj, k_proj , v_proj = torch.split(v, v.shape[0] // 3, 0)

            state_dict[k.replace("qkv_proj", "q_proj")] = permute(q_proj.transpose(1, 0))
            state_dict[k.replace("qkv_proj", "k_proj")] = permute(k_proj.transpose(1, 0), dim2=k_proj.shape[1])
            state_dict[k.replace("qkv_proj", "v_proj")] = v_proj[0]

        elif k == "embedder.weight":
            state_dict[LAYER_NAME_MAPPING[k]] = v
            state_dict["lm_head.weight"] = v
        else:
            state_dict[k] = v

    print("Loading the checkpoint in a GoldenGate model.")
    with torch.device("meta"):
        model = GoldenGateForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=True)
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(save_path, safe_serialization=safe_serialization)


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of GoldenGate weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["2B", "7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the GoldenGate2 official release. For more details on GoldenGate2, checkout the original repo: https://huggingface.co/meta-golden_gate",
    )
    parser.add_argument(
        "--output_dir",
        default="golden_gate_2b",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    spm_path = os.path.join(args.input_dir, "tokenizer.model")

    config = CONFIG_MAPPING[args.model_size]
    write_model(
        config = config,
        input_base_path=args.input_dir,
        save_path=args.output_dir,
        safe_serialization=args.safe_serialization,
    )
    write_tokenizer(
        spm_path
    )



if __name__ == "__main__":
    main()
