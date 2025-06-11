# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import tiktoken
import torch
from safetensors.torch import load_file as safe_load

from transformers import (
    GenerationConfig,
    OpenAIMoeConfig,
    OpenAIMoeForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter


# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"layers.(\d+).attention.wqkv.weight": r"layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"norm.weight":                 r"norm.weight",
    r"\nnorm.scale":                 r"\nnorm.weight",
    r"unembedding.weight":          r"lm_head.weight",
    r"embedding":                   r"embed_tokens",
    # special key, wqkv needs to be split afterwards
    r"block.(\d+).attn.qkv":        r"layers.\1.self_attn.qkv_proj",
    r"block.(\d+).attn.out":        r"layers.\1.self_attn.o_proj",
    r"block.(\d+).attn.sinks":      r"layers.\1.self_attn.sinks",
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
    eos_token_id = 199999 if not instruct else [199999, 200018]
    pad_token_id = 128004

    config = OpenAIMoeConfig()

    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    final_ = {}
    for file in list(os.listdir(input_base_path)):
        if file.endswith(".safetensors"):
            final_.update(safe_load(os.path.join(input_base_path, file)))

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
            q, k, v = (
                final_[key][:q_len, ...],
                final_[key][q_len : k_len + q_len, ...],
                final_[key][k_len + q_len :, ...],
            )
            q_key = re.sub(r"qkv_proj", "q_proj", new_key)
            k_key = re.sub(r"qkv_proj", "k_proj", new_key)
            v_key = re.sub(r"qkv_proj", "v_proj", new_key)
            state_dict[q_key] = q.contiguous().to(torch.bfloat16)
            state_dict[k_key] = k.contiguous().to(torch.bfloat16)
            state_dict[v_key] = v.contiguous().to(torch.bfloat16)
        elif re.search("gate_up_proj|down_proj", new_key) and "bias" not in new_key:
            state_dict[new_key] = final_[key].permute(0, 2, 1).contiguous()  # einsum in orignal, I use bmm
        else:
            weight = final_[key]
            if not re.search("norm", new_key):
                weight = weight.to(torch.bfloat16)  # norms are the only ones in float32
            state_dict[new_key] = weight

    del final_
    gc.collect()

    print("Loading the checkpoint in a OpenAIMoe model")
    with torch.device("meta"):
        model = OpenAIMoeForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del config._name_or_path

    print("Saving the model")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    OpenAIMoeForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
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


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class OpenAIMoeConverter(TikTokenConverter):
    def extract_vocab_merges_from_model(self, tiktoken_url: str):
        tokenizer = tiktoken.get_encoding(tiktoken_url)
        self.pattern = tokenizer._pat_str
        bpe_ranks = tokenizer._mergeable_ranks
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def __init__(
        self,
        vocab_file,
        model_max_length: int,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(vocab_file, pattern=None)

        # TODO 1st donwload the vocabfile!!!
        tokenizer = tiktoken.get_encoding(vocab_file)
        self.additional_special_tokens = {}
        # 199998 is not defined either
        self.additional_special_tokens["<|reserved_199998|>"] = 199998
        self.additional_special_tokens = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}
        for k in range(199999, 200018):
            self.additional_special_tokens[f"<|reserved_{k}|>"] = k
        sorted_list = sorted(self.additional_special_tokens.items(), key=lambda x: x[1])
        self.additional_special_tokens = [k[0] for k in sorted_list]
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

    converter = OpenAIMoeConverter(
        vocab_file=tokenizer_path,
        model_max_length=None,
        chat_template=chat_template if instruct else None,
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
        tokenizer_path="o200k_base",
        save_dir=args.output_dir,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
