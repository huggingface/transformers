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

from transformers import GemmaTokenizer, RecurrentGemmaConfig, RecurrentGemmaForCausalLM


try:
    from transformers import GemmaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    GemmaTokenizerFast = None

import regex as re


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

gemma_2b_config = RecurrentGemmaConfig(
    num_attention_heads=10,
    num_key_value_heads=1,
    hidden_size=2560,
    intermediate_size=15360,
    vocab_size=256000,
    num_hidden_layers=26,
)

gemma_7b_config = RecurrentGemmaConfig()

CONFIG_MAPPING = {"2B": gemma_2b_config, "7B": gemma_7b_config}
LAYER_NAME_MAPPING = {"embedder.weight": "model.embed_tokens.weight"}


def write_model(save_path, input_base_path, config, safe_serialization=True, push_to_hub=False, dtype=torch.float32):
    print(f"Fetching all parameters from the checkpoint at '{input_base_path}'")
    model_state_dict = torch.load(input_base_path, map_location="cpu")

    REPLACEMENT = {
        "blocks.": "layers.",
        ".ffw_down.b": ".down_proj.b",
        ".ffw_down.w": ".down_proj.w",
        ".ffw_up.b": ".up_proj.bias",
        ".ffw_up.w": ".up_proj.weight",
        "recurrent_block": "temporal_block",
        "attention_block": "temporal_block",
        "temporal_block.proj_final": "temporal_block.out_proj",
        "norm.scale": "norm.weight",
        ".proj_k": ".k_proj",
        ".proj_q": ".q_proj",
        ".proj_v": ".v_proj",
        ".proj_final": ".o_proj",
        "embedder.input_embedding": "embed_tokens.weight",
        "conv_1d.w": "conv_1d.weight",
        "conv_1d.b": "conv_1d.bias",
        "input_gate.w": "input_gate.weight",
        "input_gate.b": "input_gate.bias",
        "a_param": "recurrent_param",
        "a_gate.b": "recurrent_gate.bias",
        "a_gate.w": "recurrent_gate.weight",
    }

    state_dict = {}
    for k, v in model_state_dict.items():
        k = "model." + k
        pattern = re.compile("|".join(map(re.escape, REPLACEMENT.keys())))
        key = pattern.sub(lambda match: REPLACEMENT[match.group(0)], k)
        if "conv_1d.weight" in key:
            v = v[:, None, :].transpose(0, 2)
        if "up_proj.weight" in key:
            state_dict[key.replace("up_proj", "gate_proj")] = v[0].T.contiguous()
            v = v[1].T.contiguous()
        if "up_proj.bias" in key:
            state_dict[key.replace("up_proj", "gate_proj")] = v[0, 0, 0].clone()
            v = v[1, 0, 0].contiguous()
        if "recurrent_gate.bias" in key:
            state_dict[key.replace("gate.", "gate_")] = v.contiguous().clone()
        elif "recurrent_gate.weight" in key:
            state_dict[key.replace("gate.", "gate_")] = v.contiguous().clone()
        elif "input_gate.b" in key:
            state_dict[key.replace("gate.", "gate_")] = v.contiguous().clone()
        elif "input_gate.w" in key:
            state_dict[key.replace("gate.", "gate_")] = v.contiguous().clone()
        elif "embed_tokens" in key:
            state_dict[key] = v[: config.vocab_size, :].contiguous().clone()
            state_dict["lm_head.weight"] = v[: config.vocab_size, :].contiguous().clone()
        else:
            state_dict[key] = v.contiguous()

    torch.set_default_dtype(dtype)

    print("Loading the checkpoint in a Gemma model.")
    with init_empty_weights():
        model = RecurrentGemmaForCausalLM(config)
    model.load_state_dict(state_dict, assign=True, strict=True)

    model.config.torch_dtype = torch.float32
    del model.config._name_or_path
    print("Saving in the Transformers format.")

    if push_to_hub:
        print(f"pushing the model to {save_path}")
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
        default="/home/arthur/transformers_recurrentgemma/google/recurrent-gemma-2b-it/ToBeDeleted/2b-it.pt",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        help="Location of Gemma tokenizer model",
    )
    parser.add_argument(
        "--model_size",
        default="2B",
        choices=["2B", "7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Gemma2 official release. For more details on Gemma2, checkout the original repo: https://huggingface.co/google/gemma-7b",
    )
    parser.add_argument(
        "--output_dir",
        default="google/recurrent-gemma-2b-it-hf",
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
