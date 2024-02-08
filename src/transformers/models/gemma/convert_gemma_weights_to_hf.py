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

from transformers import FlaxGemmaForCausalLM, GemmaConfig, GemmaForCausalLM, GemmaTokenizer


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


def write_model(save_path, input_base_path, config, safe_serialization=True):
    num_attn_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    model_state_dict = torch.load(os.path.join(input_base_path), map_location="cpu")["model_state_dict"]
    model_state_dict.pop("freqs_cis")

    state_dict = {}
    for k, v in model_state_dict.items():
        if "qkv_proj" in k:
            if num_kv_heads == 1:
                v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim, hidden_size)
                q_proj = v[:num_attn_heads, ...]
                k_proj = v[num_attn_heads : num_attn_heads + num_kv_heads, ...].repeat(num_kv_heads, 1, 1)
                v_proj = v[-num_kv_heads:, ...].repeat(num_kv_heads, 1, 1)

                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(num_attn_heads*head_dim, hidden_size)
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(num_kv_heads*head_dim, hidden_size)
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj[0]
            else:
                q_proj, k_proj, v_proj = torch.split(v, v.shape[0] // 3, 0)
                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(num_attn_heads*head_dim, hidden_size)
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(num_kv_heads*head_dim, hidden_size)
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj

        elif k == "embedder.weight":
            state_dict[LAYER_NAME_MAPPING[k]] = v
            state_dict["lm_head.weight"] = v
        else:
            state_dict[k] = v

    print("Loading the checkpoint in a Gemma model.")
    with torch.device("meta"):
        model = GemmaForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=False)

    model.config.torch_dtype = torch.float32
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(save_path, safe_serialization=safe_serialization)


def write_tokenizer(input_tokenizer_path, save_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = GemmaTokenizer  # if GemmaTokenizerFast is None else GemmaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {save_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Gemma weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        default="7B",
        choices=["2B", "7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Gemma2 official release. For more details on Gemma2, checkout the original repo: https://huggingface.co/meta-gemma",
    )
    parser.add_argument(
        "--output_dir",
        default="gemma_7b",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    spm_path = os.path.join(args.input_dir,"tokenizer.model")

    config = CONFIG_MAPPING[args.model_size]
    write_model(
        config=config,
        input_base_path=args.input_dir,
        save_path=args.output_dir,
        safe_serialization=args.safe_serialization,
    )
    write_tokenizer(spm_path, args.output_dir)

    import random
    import jax.numpy as jnp

    import numpy as np

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    tokenizer = GemmaTokenizerFast.from_pretrained(
        args.output_dir, pad_token="<pad>", from_slow=True, eos_token="<eos>", bos_tokens="<bos>"
    )
    tokenizer.padding_side = "left"
    model = GemmaForCausalLM.from_pretrained(args.output_dir, pad_token_id=0, eos_token_id=1, bos_token_id=2, torch_dtype = torch.float32) #, low_cpu_mem_usage = True)
    
    model.push_to_hub("gg-hf/gemma-7b", safe_serialization = True, max_shard_size = "2GB")
    # del model 
    # model = FlaxGemmaForCausalLM.from_pretrained(args.output_dir, pad_token_id=0, eos_token_id=1, bos_token_id=2, from_pt=True, dtype=jnp.float32)
    
    # model.push_to_hub("gg-hf/gemma-7b", safe_serialization = False, max_shard_size = "2GB")
    
    exit(0)
    
    device = "cpu"
    model = model.to(device)
    # model.generation_config.temperature = 1
    # model.generation_config.top_p = 1

    inputs = torch.tensor([[2, 651, 6037, 576, 6081, 603]])
    

    outputs = model.generate(inputs, do_sample=False, max_new_tokens=100)
    print(outputs)
    print(tokenizer.batch_decode(outputs))
    import jax.numpy as jnp

    model = FlaxGemmaForCausalLM.from_pretrained(args.output_dir, pad_token_id=0, eos_token_id=1, bos_token_id=2, from_pt=True, dtype=jnp.float32)

    inputs = jnp.array([[2, 651, 6037, 576, 6081, 603]])
    outputs = model.generate(inputs, do_sample=True, max_new_tokens=20)
    print(outputs)
    print(tokenizer.batch_decode(outputs.sequences))


if __name__ == "__main__":
    main()