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
from typing import Optional

import regex as re
import tiktoken
import torch
from safetensors.torch import load_file as safe_load

from transformers import (
    GenerationConfig,
    GptOssConfig,
    GptOssForCausalLM,
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
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
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

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

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
    mxfp4=False,
):
    os.makedirs(model_path, exist_ok=True)
    eos_token_id = 199999 if not instruct else 200002
    pad_token_id = 199999

    original_config = json.loads((Path(input_base_path) / "config.json").read_text())

    num_local_experts = original_config.pop("num_experts")
    rope_scaling = {
        "beta_fast": float(original_config.pop("rope_ntk_beta")),
        "beta_slow": float(original_config.pop("rope_ntk_alpha")),
        "factor": float(original_config.pop("rope_scaling_factor")),
        "rope_type": "yarn",
        "truncate": False,
        "original_max_position_embeddings": 4096,
    }

    config = GptOssConfig(
        num_local_experts=num_local_experts,
        rope_scaling=rope_scaling,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        **original_config,
    )

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
            if not mxfp4:
                if "scales" in new_key:
                    continue
                elif "blocks" in new_key:
                    # deal with packed weights
                    blocks = final_[key]
                    scales = final_[key.replace("blocks", "scales")]
                    new_key = new_key.replace(".blocks", "")
                    unpacked_tensors = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
                    unpacked_tensors = unpacked_tensors.permute(0, 2, 1).contiguous()  # einsum in orignal, I use bmm
                    state_dict[new_key] = unpacked_tensors
                else:
                    raise (f"Unidentified {key}, please double check the state dict")
            else:
                if "scales" in new_key:
                    new_key = new_key.replace(".scales", "_scales")
                    state_dict[new_key] = final_[key].contiguous()
                elif "blocks" in new_key:
                    new_key = new_key.replace(".blocks", "_blocks")
                    state_dict[new_key] = final_[key].contiguous()
                else:
                    raise (f"Unidentified {key}, please double check the state dict")
        else:
            weight = final_[key]
            if not re.search("norm", new_key):
                weight = weight.to(torch.bfloat16)  # norms are the only ones in float32
            state_dict[new_key] = weight

    del final_
    gc.collect()

    if not mxfp4:
        print("Loading the checkpoint in a GptOss model for unpacked format")
        with torch.device("meta"):
            model = GptOssForCausalLM(config)
        model.load_state_dict(state_dict, strict=True, assign=True)
        print("Checkpoint loaded successfully.")
        del config._name_or_path

        print("Saving the model")
        model.save_pretrained(model_path, safe_serialization=safe_serialization)
        del state_dict, model

    else:
        print("Saving the checkpoint in mxfp4 format")
        config.quantization_config = {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        }
        # required as we don't save the model with save_pretrained
        config.architectures = ["GptOssForCausalLM"]
        config.save_pretrained(model_path)
        save_sharded_model(state_dict, model_path)
        del state_dict

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    GptOssForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")

    # generation config
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            bos_token_id=199998,  # <|startoftext|>
            do_sample=True,
            eos_token_id=[200002, 199999],  # <|return|>, <|endoftext|>
            pad_token_id=199999,  # <|endoftext|>
            temperature=1.0,
            top_p=1.0,
        )
        generation_config.save_pretrained(model_path)


def save_sharded_model(state_dict, model_path):
    from safetensors.torch import save_file

    max_shard_size = 4800000000  # 4.8 GB
    os.makedirs(model_path, exist_ok=True)
    shard_size_counter = 0
    shard_id = 0
    shard_state_dict = {}
    total_sharded_dict = {}
    safetensors_index = {}
    safetensors_index["metadata"] = {"total_size": 0}
    safetensors_index["weight_map"] = {}
    for key in state_dict.keys():
        size = state_dict[key].numel() * state_dict[key].element_size()
        if shard_size_counter + size > max_shard_size:
            total_sharded_dict[shard_id] = shard_state_dict
            shard_id += 1
            shard_size_counter = 0
            shard_state_dict = {}
        shard_state_dict[key] = state_dict[key]
        shard_size_counter += size
        safetensors_index["metadata"]["total_size"] += size
        safetensors_index["weight_map"][key] = shard_id
    total_sharded_dict[shard_id] = shard_state_dict
    num_shards = len(total_sharded_dict) - 1
    for shard_id, shard_state_dict in total_sharded_dict.items():
        save_file(shard_state_dict, os.path.join(model_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"))
    create_safetensors_index(safetensors_index, num_shards, model_path)


def create_safetensors_index(safetensors_index, num_shards, model_path):
    for key in safetensors_index["weight_map"].keys():
        shard_id = safetensors_index["weight_map"][key]
        safetensors_index["weight_map"][key] = f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
    with open(os.path.join(model_path, "model.safetensors.index.json"), "w") as f:
        json.dump(safetensors_index, f)


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


class GptOssConverter(TikTokenConverter):
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
        self.additional_special_tokens = [tok for tok, _ in sorted(special_tokens_map.items(), key=lambda x: x[1])]
        tokenizer = self.converted()
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|startoftext|>",
            eos_token="<|return|>" if chat_template else "<|endoftext|>",
            pad_token="<|endoftext|>",
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False):
    # Updated Harmony chat template
    chat_template = """{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - "builtin_tools": A list, can contain "browser" and/or "python".
  - "model_identity": A string that optionally describes the model identity.
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
 #}

{#- Tool Definition Rendering ============================================== #}
{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}
    {%- if param_spec.type == "array" -%}
        {%- if param_spec['items'] -%}
            {%- if param_spec['items']['type'] == "string" -%}
                {{- "string[]" }}
            {%- elif param_spec['items']['type'] == "number" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "integer" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "boolean" -%}
                {{- "boolean[]" }}
            {%- else -%}
                {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
                {%- if inner_type == "object | object" or inner_type|length > 50 -%}
                    {{- "any[]" }}
                {%- else -%}
                    {{- inner_type + "[]" }}
                {%- endif -%}
            {%- endif -%}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- else -%}
            {{- "any[]" }}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
        {%- if param_spec.type | length > 1 -%}
            {{- param_spec.type | join(" | ") }}
        {%- else -%}
            {{- param_spec.type[0] }}
        {%- endif -%}
    {%- elif param_spec.oneOf -%}
        {#- Handle oneOf schemas - check for complex unions and fallback to any #}
        {%- set has_object_variants = false -%}
        {%- for variant in param_spec.oneOf -%}
            {%- if variant.type == "object" -%}
                {%- set has_object_variants = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
            {{- "any" }}
        {%- else -%}
            {%- for variant in param_spec.oneOf -%}
                {{- render_typescript_type(variant, required_params) -}}
                {%- if variant.description %}
                    {{- "// " + variant.description }}
                {%- endif -%}
                {%- if variant.default is defined %}
                    {{ "// default: " + variant.default|tojson }}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- " | " }}
                {% endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif param_spec.type == "string" -%}
        {%- if param_spec.enum -%}
            {{- '"' + param_spec.enum|join('" | "') + '"' -}}
        {%- else -%}
            {{- "string" }}
            {%- if param_spec.nullable %}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type == "number" -%}
        {{- "number" }}
    {%- elif param_spec.type == "integer" -%}
        {{- "number" }}
    {%- elif param_spec.type == "boolean" -%}
        {{- "boolean" }}

    {%- elif param_spec.type == "object" -%}
        {%- if param_spec.properties -%}
            {{- "{\n" }}
            {%- for prop_name, prop_spec in param_spec.properties.items() -%}
                {{- prop_name -}}
                {%- if prop_name not in (param_spec.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{ render_typescript_type(prop_spec, param_spec.required or []) }}
                {%- if not loop.last -%}
                    {{-", " }}
                {%- endif -%}
            {%- endfor -%}
            {{- "}" }}
        {%- else -%}
            {{- "object" }}
        {%- endif -%}
    {%- else -%}
        {{- "any" }}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_namespace(namespace_name, tools) -%}
    {{- "## " + namespace_name + "\n\n" }}
    {{- "namespace " + namespace_name + " {\n\n" }}
    {%- for tool in tools %}
        {%- set tool = tool.function %}
        {{- "// " + tool.description + "\n" }}
        {{- "type "+ tool.name + " = " }}
        {%- if tool.parameters and tool.parameters.properties %}
            {{- "(_: {\n" }}
            {%- for param_name, param_spec in tool.parameters.properties.items() %}
                {%- if param_spec.description %}
                    {{- "// " + param_spec.description + "\n" }}
                {%- endif %}
                {{- param_name }}
                {%- if param_name not in (tool.parameters.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
                {%- if param_spec.default is defined -%}
                    {%- if param_spec.enum %}
                        {{- ", // default: " + param_spec.default }}
                    {%- elif param_spec.oneOf %}
                        {{- "// default: " + param_spec.default }}
                    {%- else %}
                        {{- ", // default: " + param_spec.default|tojson }}
                    {%- endif -%}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- ",\n" }}
                {%- else %}
                    {{- ",\n" }}
                {%- endif -%}
            {%- endfor %}
            {{- "}) => any;\n\n" }}
        {%- else -%}
            {{- "() => any;\n\n" }}
        {%- endif -%}
    {%- endfor %}
    {{- "} // namespace " + namespace_name }}
{%- endmacro -%}

{%- macro render_builtin_tools(browser_tool, python_tool) -%}
    {%- if browser_tool %}
        {{- "## browser\n\n" }}
        {{- "// Tool for browsing.\n" }}
        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\n" }}
        {{- "// Cite information from the tool using the following format:\n" }}
        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\n" }}
        {{- "// Do not quote more than 10 words directly from the tool output.\n" }}
        {{- "// sources=web (default: web)\n" }}
        {{- "namespace browser {\n\n" }}
        {{- "// Searches for information related to `query` and displays `topn` results.\n" }}
        {{- "type search = (_: {\n" }}
        {{- "query: string,\n" }}
        {{- "topn?: number, // default: 10\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\n" }}
        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\n" }}
        {{- "// If `cursor` is not provided, the most recent page is implied.\n" }}
        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\n" }}
        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\n" }}
        {{- "// Use this function without `id` to scroll to a new location of an opened page.\n" }}
        {{- "type open = (_: {\n" }}
        {{- "id?: number | string, // default: -1\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "loc?: number, // default: -1\n" }}
        {{- "num_lines?: number, // default: -1\n" }}
        {{- "view_source?: boolean, // default: false\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\n" }}
        {{- "type find = (_: {\n" }}
        {{- "pattern: string,\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "}) => any;\n\n" }}
        {{- "} // namespace browser\n\n" }}
    {%- endif -%}

    {%- if python_tool %}
        {{- "## python\n\n" }}
        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\n" }}
        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\n\n" }}
    {%- endif -%}
{%- endmacro -%}

{#- System Message Construction ============================================ #}
{%- macro build_system_message() -%}
    {%- if model_identity is not defined %}
        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
    {%- endif %}
    {{- model_identity + "\n" }}
    {{- "Knowledge cutoff: 2024-06\n" }}
    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}
    {%- if reasoning_effort is not defined %}
        {%- set reasoning_effort = "medium" %}
    {%- endif %}
    {{- "Reasoning: " + reasoning_effort + "\n\n" }}
    {%- if builtin_tools %}
        {{- "# Tools\n\n" }}
        {%- set available_builtin_tools = namespace(browser=false, python=false) %}
        {%- for tool in builtin_tools %}
            {%- if tool == "browser" %}
                {%- set available_builtin_tools.browser = true %}
            {%- elif tool == "python" %}
                {%- set available_builtin_tools.python = true %}
            {%- endif %}
        {%- endfor %}
        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}
    {%- endif -%}
    {{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}
    {%- if tools -%}
        {{- "\nCalls to these tools must go to the commentary channel: 'functions'." }}
    {%- endif -%}
{%- endmacro -%}

{#- Main Template Logic ================================================= #}
{#- Set defaults #}

{#- Render system message #}
{{- "<|start|>system<|message|>" }}
{{- build_system_message() }}
{{- "<|end|>" }}

{#- Extract developer message #}
{%- if messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}

{#- Render developer message #}
{%- if developer_message or tools %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- "# Instructions\n\n" }}
        {{- developer_message }}
    {%- endif %}
    {%- if tools -%}
        {{- "\n\n" }}
        {{- "# Tools\n\n" }}
        {{- render_tool_namespace("functions", tools) }}
    {%- endif -%}
    {{- "<|end|>" }}
{%- endif %}

{#- Render messages #}
{%- set last_tool_call = namespace(name=none) %}
{%- for message in loop_messages -%}
    {#- At this point only assistant/user/tool messages should remain #}
    {%- if message.role == 'assistant' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "thinking" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "tool_calls" in message %}
            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}
            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}
            {#- when we render CoT/analysis messages in inference. #}
            {%- set future_final_message = namespace(found=false) %}
            {%- for future_message in loop_messages[loop.index:] %}
                {%- if future_message.role == 'assistant' and "tool_calls" not in future_message %}
                    {%- set future_final_message.found = true %}
                {%- endif %}
            {%- endfor %}
            {#- We assume max 1 tool call per message, and so we infer the tool call name #}
            {#- in "tool" messages from the most recent assistant tool call name #}
            {%- set tool_call = message.tool_calls[0] %}
            {%- if tool_call.function %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {%- if message.content and message.thinking %}
                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}
            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- elif message.thinking and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant to=" }}
            {{- "functions." + tool_call.name + "<|channel|>commentary " }}
            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}
            {{- tool_call.arguments|tojson }}
            {{- "<|call|>" }}
            {%- set last_tool_call.name = tool_call.name %}
        {%- elif loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
        {%- else %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- endif %}
    {%- elif message.role == 'tool' -%}
        {%- if last_tool_call.name is none %}
            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
        {%- endif %}
        {{- "<|start|>functions." + last_tool_call.name }}
        {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
    {%- elif message.role == 'user' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}

{#- Generation prompt #}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}"""

    converter = GptOssConverter(
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
        default="/fsx/mohamed/oai-hf/tests/120b",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="/fsx/mohamed/oai-hf/tests/120b_converted_packed",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=list[str],
        help="The list of special tokens that should be added to the ",
    )

    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )

    # Only specify this if you want to use the model with mxfp4 quantization
    # It means the model will be unpacked, and quantized using mxfp4 during inference if all the triton requirements are satisfied (triton >= 3.4.0)
    # Else we have a fallback to the full precision model (bfloat16)
    # If not specified, the model will be unpacked during conversion, and will be in fp8/bfloat16 during inference
    # Note: mxfp4 should bring an important speedup in inference time with blackwell gpus
    parser.add_argument(
        "--mxfp4",
        action="store_true",
        help="Whether to use the original model with mxfp4 quantization or default to the full precision model.",
    )

    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        instruct=args.instruct,
        mxfp4=args.mxfp4,
    )

    write_tokenizer(
        tokenizer_path="o200k_base",
        save_dir=args.output_dir,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
