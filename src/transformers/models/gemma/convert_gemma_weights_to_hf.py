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
import os
import warnings

import torch
from accelerate import init_empty_weights

from transformers import GemmaConfig, GemmaForCausalLM, GemmaTokenizer


try:
    from transformers import GemmaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    GemmaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/gemma/convert_gemma_weights_to_hf.py \
    --input_dir /path/to/downloaded/gemma/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import GemmaForCausalLM, GemmaTokenizerFast

model = GemmaForCausalLM.from_pretrained("/output/path")
tokenizer = GemmaTokenizerFast.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

gemma_2b_config = GemmaConfig(
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    hidden_size=2048,
    intermediate_size=16384,
)

gemma_7b_config = GemmaConfig()

CONFIG_MAPPING = {"2B": gemma_2b_config, "7B": gemma_7b_config}
LAYER_NAME_MAPPING = {"embedder.weight": "model.embed_tokens.weight"}


def write_model(save_path, input_base_path, config, safe_serialization=True, push_to_hub=False, dtype=torch.float32):
    num_attn_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    print(f"Fetching all parameters from the checkpoint at '{input_base_path}'")
    model_state_dict = torch.load(input_base_path, map_location="cpu", weights_only=True)["model_state_dict"]
    model_state_dict.pop("freqs_cis")

    state_dict = {}
    for k, v in model_state_dict.items():
        if "qkv_proj" in k:
            if num_kv_heads == 1:
                v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim, hidden_size)
                q_proj = v[:num_attn_heads, ...]
                k_proj = v[num_attn_heads : num_attn_heads + num_kv_heads, ...].repeat(num_kv_heads, 1, 1)
                v_proj = v[-num_kv_heads:, ...].repeat(num_kv_heads, 1, 1)

                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(
                    num_attn_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(
                    num_kv_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj[0].clone()
            else:
                q_proj, k_proj, v_proj = torch.split(v, v.shape[0] // 3, 0)
                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(
                    num_attn_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(
                    num_kv_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj.clone()

        elif k == "embedder.weight":
            state_dict[LAYER_NAME_MAPPING[k]] = v
            state_dict["lm_head.weight"] = v
        else:
            state_dict[k] = v

    torch.set_default_dtype(dtype)

    print("Loading the checkpoint in a Gemma model.")
    with init_empty_weights():
        model = GemmaForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=False)

    model.config.torch_dtype = torch.float32
    del model.config._name_or_path
    print("Saving in the Transformers format.")

    if push_to_hub:
        print(f"pushing the model to {save_path}")
        model.push_to_hub(save_path, safe_serialization=safe_serialization, private=True)
    else:
        model.save_pretrained(save_path, safe_serialization=safe_serialization)


def write_tokenizer(input_tokenizer_path, save_path, push_to_hub=False):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = GemmaTokenizer if GemmaTokenizerFast is None else GemmaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {save_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    if push_to_hub:
        tokenizer.push_to_hub(save_path)
    else:
        tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_checkpoint",
        help="Absolute path to the target Gemma weights.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        help="Location of Gemma tokenizer model",
    )
    parser.add_argument(
        "--model_size",
        default="7B",
        choices=["2B", "7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Gemma2 official release. For more details on Gemma2, check out the original repo: https://huggingface.co/google/gemma-7b",
    )
    parser.add_argument(
        "--output_dir",
        default="google/gemma-7b",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--pickle_serialization",
        help="Whether or not to save using `safetensors`.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--convert_tokenizer",
        help="Whether or not to convert the tokenizer as well.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Target dtype of the converted model",
    )
    args = parser.parse_args()

    if args.convert_tokenizer:
        if args.tokenizer_checkpoint is None:
            raise ValueError("Path to the tokenizer is required when passing --convert_tokenizer")

        spm_path = os.path.join(args.tokenizer_checkpoint)
        write_tokenizer(spm_path, args.output_dir, args.push_to_hub)

    config = CONFIG_MAPPING[args.model_size]
    dtype = getattr(torch, args.dtype)
    write_model(
        config=config,
        input_base_path=args.input_checkpoint,
        save_path=args.output_dir,
        safe_serialization=not args.pickle_serialization,
        push_to_hub=args.push_to_hub,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
