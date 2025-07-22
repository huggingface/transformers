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
from pathlib import Path
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

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math
    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total   = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    # to match for now existing implementation
    return out.to(torch.float8_e5m2)


def write_model(
    model_path,
    input_base_path,
    safe_serialization=True,
    instruct=False,
    unpack=True,
):
    os.makedirs(model_path, exist_ok=True)
    bos_token_id = 128000
    eos_token_id = 199999 if not instruct else [199999, 200018]
    pad_token_id = 128004

    original_config = json.loads((Path(input_base_path) / "config.json").read_text())

    num_local_experts = original_config.pop("num_experts")
    rope_scaling = {
        "beta_fast": float(original_config.pop("rope_ntk_beta")),
        "beta_slow": float(original_config.pop("rope_ntk_alpha")),
        "factor": float(original_config.pop('rope_scaling_factor')),
        "rope_type": "yarn",
        "truncate": False,
        "original_max_position_embeddings": 4096
      }

    config = OpenAIMoeConfig(num_local_experts=num_local_experts, rope_scaling=rope_scaling, **original_config)
    print(config)
    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    final_ = {}
    for file in list(os.listdir(input_base_path)):
        if file.endswith(".safetensors"):
            final_.update(safe_load(os.path.join(input_base_path, file)))

    print("Converting ..", unpack)
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
            if unpack:
                if "scales" in new_key:
                    continue
                elif "blocks" in new_key:
                    # deal with packed weights
                    blocks = final_[key]
                    scales = final_[key.replace("blocks", "scales")]
                    new_key = new_key.replace(".blocks","")
                    unpacked_tensors = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
                    unpacked_tensors = unpacked_tensors.permute(0, 2, 1).contiguous()  # einsum in orignal, I use bmm
                    state_dict[new_key] = unpacked_tensors
                else:
                    raise(f"Unidentified {key}, please double check the state dict")
            else:
                if "scales" in new_key:
                    new_key = new_key.replace(".scales", "_scales")
                    state_dict[new_key] = final_[key].contiguous()
                elif "blocks" in new_key:
                    new_key = new_key.replace(".blocks", "_blocks")
                    state_dict[new_key] = final_[key].contiguous()
                else:
                    raise(f"Unidentified {key}, please double check the state dict")
        else:
            weight = final_[key]
            if not re.search("norm", new_key):
                weight = weight.to(torch.bfloat16)  # norms are the only ones in float32
            state_dict[new_key] = weight

    del final_
    gc.collect()

    if unpack:
        print("Loading the checkpoint in a OpenAIMoe model for unpacked format")
        with torch.device("meta"):
            model = OpenAIMoeForCausalLM(config)
        model.load_state_dict(state_dict, strict=True, assign=True)
        print("Checkpoint loaded successfully.")
        del config._name_or_path

        print("Saving the model")
        model.save_pretrained(model_path, safe_serialization=safe_serialization)
        del state_dict, model

    else:
        print("Saving the checkpoint in packed format")
        config.quantization_config = {
                                        "quant_method": "mxfp4",
                                        "modules_to_not_convert":[
                                            "model.layers.*.self_attn",
                                            "model.layers.*.mlp.router",
                                            "model.embed_tokens",
                                            "lm_head"
                                    ]}
        config.save_pretrained(model_path)
        save_sharded_model(state_dict, model_path)
        del state_dict

    # Safety check: reload the converted model
    gc.collect()
    # TODO: remove when mxfp4 pr is merged
    if unpack:
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


def save_sharded_model(state_dict, model_path):
    import math
    from safetensors.torch import save_file

    max_shard_size = 4800000000 # 4.8 GB
    os.makedirs(model_path, exist_ok=True)
    shard_size_counter = 0
    shard_id = 0
    shard_state_dict = {}
    total_sharded_dict = {}
    for key in state_dict.keys():
        size = state_dict[key].numel()*state_dict[key].element_size()
        if shard_size_counter + size > max_shard_size:
            total_sharded_dict[shard_id] = shard_state_dict
            shard_id += 1
            shard_size_counter = 0
            shard_state_dict = {}
        shard_state_dict[key] = state_dict[key]
        shard_size_counter += size
    total_sharded_dict[shard_id] = shard_state_dict
    num_shards = len(total_sharded_dict) - 1
    for shard_id, shard_state_dict in total_sharded_dict.items():
        save_file(
            shard_state_dict,
            os.path.join(model_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors")
        )

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
        # Complete list of Harmony special tokens as per o200k_harmony spec
        special_tokens_map = {
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|call|>": 200012,
            "<|endofprompt|>": 200018,
        }

        # Add the remaining reserved slots while skipping IDs already present above.
        used_ids = set(special_tokens_map.values())
        for k in range(199999, 200018):
            if k in used_ids:
                continue
            special_tokens_map.setdefault(f"<|reserved_{k}|>", k)

        # Keep only token strings (sorted by ID) for TikTokenConverter.
        self.additional_special_tokens = [
            tok for tok, _ in sorted(special_tokens_map.items(), key=lambda x: x[1])
        ]
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
    # Updated Harmony chat template
    chat_template = """{# Harmony chat template --------------------------------------------------
   This template mirrors the message rendering logic implemented in
   `harmony/src/encoding.rs`.  It can be consumed by Hugging Face
   Transformers (``chat_template`` field) so that *text → tokens*
   conversion of chat conversations happens fully on the Python side
   without relying on the Rust renderer.

   Supported *message* keys (per ``chat::Message``):
     - role (user│assistant│system│developer│tool)
     - name (optional author name)
     - recipient (optional recipient – omitted or "all" → broadcast)
     - channel   (optional meta channel)
     - content_type (optional content-type qualifier)
     - content (string – the actual message payload)

   The template renders each historical message *fully* (incl. the
   trailing <|end|>/<|return|> sentinel) and – if ``add_generation_prompt``
   is True – appends a partial header for the **next** assistant turn
   exactly like ``render_conversation_for_completion`` does on the Rust
   side: ``<|start|>assistant``.
#}

{%- macro harmony_header(m) -%}
    <|start|>{% if m['role'] == 'tool' %}{{ m['name'] }}{% else %}{{ m['role'] }}{% if m.get('name') %}:{{ m['name'] }}{% endif %}{% endif %}{% if m.get('recipient') and m['recipient'] != 'all' %} to={{ m['recipient'] }}{% endif %}{% if m.get('channel') %}<|channel|>{{ m['channel'] }}{% endif %}{% if m.get('content_type') %} {{ m['content_type'] }}{% endif %}<|message|>
{%- endmacro -%}

{# Add CoT dropping logic -------------------------------------------- #}
{%- set last_final_idx = None -%}
{%- for idx in range(messages|length) -%}
    {%- set m = messages[idx] -%}
    {%- if m['role'] == 'assistant' and m.get('channel') == 'final' -%}
        {%- set last_final_idx = idx -%}
    {%- endif -%}
{%- endfor -%}
{%- set last_user_idx = None -%}
{%- if last_final_idx is not none -%}
    {%- for idx in range(last_final_idx - 1, -1, -1) -%}
        {%- if messages[idx]['role'] == 'user' -%}
            {%- set last_user_idx = idx -%}
            {%- break -%}
        {%- endif -%}
    {%- endfor -%}
{%- endif -%}

{# ---------------------------------------------------------------------
   Render complete history (with CoT dropping)
#}
{%- for idx in range(messages|length) -%}
    {%- set message = messages[idx] -%}
    {%- set skip = false -%}
    {%- if last_final_idx is not none and idx < last_final_idx and (last_user_idx is none or idx > last_user_idx) -%}
        {%- if message['role'] == 'assistant' and message.get('channel') != 'final' -%}
            {%- set skip = true -%}
        {%- endif -%}
    {%- endif -%}
    {%- if not skip -%}
        {{- harmony_header(message) -}}{{ message['content'] }}{%- if message['role'] == 'assistant' and message.get('channel') == 'final' -%}<|return|>{%- else -%}<|end|>{%- endif -%}
    {%- endif -%}
{%- endfor -%}

{# ---------------------------------------------------------------------
   Generation prompt for *next* assistant answer
#}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}
"""

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
        default="/fsx/mohamed/oai-hf/tests/20b",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="/fsx/mohamed/oai-hf/tests/20b_converted_packed",
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

    parser.add_argument(
        "--unpack",
        action="store_true",
        help="Whether to unpack the model or keep the scales as in the original format. Defaults to True if not specified.",
    )

    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        instruct=args.instruct,
        unpack=args.unpack,
    )

    write_tokenizer(
        tokenizer_path="o200k_base",
        save_dir=args.output_dir,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
