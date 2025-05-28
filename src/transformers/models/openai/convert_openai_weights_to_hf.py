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
import gc
import json
import os
from typing import List, Optional

import regex as re
import torch
from tqdm import tqdm
from safetensors.torch import load_file as safe_load

from transformers import (
    GenerationConfig,
    OpenaiConfig,
    OpenaiForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter


# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"layers.(\d+).attention.wqkv.weight": r"layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"norm.weight":                 r"norm.weight",
    r"^norm.scale":                 r"norm.weight",
    r"unembedding.weight":          r"lm_head.weight",
    r"embedding":                   r"embed_tokens",
    # special key, wqkv needs to be split afterwards
    r"block.(\d+).attn.qkv":        r"layers.\1.self_attn.qkv_proj",
    r"block.(\d+).attn.out":        r"layers.\1.self_attn.o_proj",
    r"block.(\d+).attn.sdpa.sinks":      r"layers.\1.self_attn.sinks",
    r"block.(\d+).attn.norm.scale":       r"layers.\1.input_layernorm.weight",

    r"block.(\d+).mlp.mlp1_weight": r"layers.\1.mlp.experts.gate_up_proj",
    r"block.(\d+).mlp.mlp1_bias":   r"layers.\1.mlp.experts.gate_up_proj_bias",
    r"block.(\d+).mlp.mlp2_weight": r"layers.\1.mlp.experts.down_proj",
    r"block.(\d+).mlp.mlp2_bias":   r"layers.\1.mlp.experts.down_proj_bias",
    r"block.(\d+).mlp.norm.scale":        r"layers.\1.post_attention_layernorm.weight",
    r"block.(\d+).mlp.gate":        r"layers.\1.mlp.router",
}
# fmt: on



def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict



def write_model(
    model_path,
    input_base_path,
    safe_serialization=True,
    instruct=False,
):
    os.makedirs(model_path, exist_ok=True)
    bos_token_id = 128000
    eos_token_id = [128001, 128008, 128009] if instruct else 128001
    pad_token_id = 128004

    config = OpenaiConfig()

    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    final_ = {}
    for file in list(os.listdir(input_base_path)):
        if file.endswith(".safetensors"):
            final_.update(safe_load(os.path.join(input_base_path,file)) )

    print("Converting ..")
    all_keys = final_.keys()
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        # Post-process the current_parameter.
        new_key = new_keys.get(key, key)
        if "lm_head" not in new_key:
            new_key = "model." + new_key
        print(f"Processing key: {key} -> {new_key}")
        if re.search("qkv_proj", new_key):
            q_len = config.head_dim * config.num_attention_heads
            k_len = config.head_dim * config.num_key_value_heads
            q, k, v = final_[key][:q_len, ...], final_[key][q_len:k_len+q_len, ...], final_[key][k_len+q_len:, ...]
            q_key = re.sub(r"qkv_proj", "q_proj", new_key)
            k_key = re.sub(r"qkv_proj", "k_proj", new_key)
            v_key = re.sub(r"qkv_proj", "v_proj", new_key)
            state_dict[q_key] = q
            state_dict[k_key] = k
            state_dict[v_key] = v
        elif re.search("gate_up_proj", new_key) and "bias" not in new_key:
            state_dict[new_key] = final_[key].permute(0,2,1)
        else:
            state_dict[new_key] = final_[key].to(torch.bfloat16) # TODO slow, let's rmeove

    del final_
    gc.collect()

    print("Loading the checkpoint in a OpenAI ")
    with torch.device("meta"):
        model = OpenaiForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del config._name_or_path

    print("Saving the ")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    OpenaiForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")

    # generation config
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generation_config.save_pretrained(model_path)


class OpenaiConverter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens: List[str],
        pattern: str,
        model_max_length: int,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(vocab_file, pattern=pattern)
        # TODO 1st donwload the vocabfile!!!
        self.additional_special_tokens = special_tokens
        tokenizer = self.converted()
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False):
    model_max_length = CONTEXT_LENGTH
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: W605
    special_tokens = []

    # Chat template
    chat_template = (
        "{% for message in messages %}"
        "{% if loop.index0 == 0 %}"
        "{{ bos_token }}"
        "{% endif %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|image|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    converter = OpenaiConverter(
        vocab_file=tokenizer_path,
        pattern=pattern,
        special_tokens=special_tokens,
        model_max_length=model_max_length,
        chat_template=chat_template if instruct else None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>" if not instruct else "<|eot_id|>",
        pad_token="<|finetune_right_pad_id|>",
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)

    if instruct:
        print("Saving chat template...")
        chat_template_path = os.path.join(save_dir, "chat_template.json")
        with open(chat_template_path, "w") as f:
            json.dump({"chat_template": chat_template}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/fsx/arthur/oai",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="/fsx/arthur/oai_hf",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the ",
    )

    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        instruct=args.instruct,
    )

    write_tokenizer(
        tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
        save_dir=args.output_dir,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
