# Copyright 2023 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import re

import torch
from safetensors.torch import load_file
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import AutoTokenizer, LlamaTokenizerFast, MistralConfig, MistralForCausalLM
from transformers.convert_slow_tokenizer import bytes_to_unicode


# fmt: off
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"^output.weight":                            r"lm_head.weight",

    # Model keys
    r"^norm.weight":                              r"model.norm.weight",
    r"^tok_embeddings.weight":                    r"model.embed_tokens.weight",

    # Layers keys
    r"^layers.(\d+).attention_norm.weight":       r"model.layers.\1.input_layernorm.weight",
    r"^layers.(\d+).ffn_norm.weight":             r"model.layers.\1.post_attention_layernorm.weight",

    # Attention keys
    r"^layers.(\d+).attention.w(q|k|v|o).weight": r"model.layers.\1.self_attn.\2_proj.weight",


    # MLP keys
    r"^layers.(\d+).feed_forward.w1.weight":      r"model.layers.\1.mlp.gate_proj.weight",
    r"^layers.(\d+).feed_forward.w2.weight":      r"model.layers.\1.mlp.down_proj.weight",
    r"^layers.(\d+).feed_forward.w3.weight":      r"model.layers.\1.mlp.up_proj.weight",
}
# fmt: on


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def permute_for_rope(tensor, n_heads, dim1, dim2):
    """Permute the weights for the ROPE formulation."""
    tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(dim1, dim2)
    return tensor


def convert_state_dict(original_state_dict: dict, config: MistralConfig):
    """Convert a state dict file, when a single `nn.Module` is never sharded in different files (usual case)."""
    new_dict = {}

    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_key_value_heads = config.num_key_value_heads
    key_value_dim = head_dim * num_key_value_heads
    query_dim = head_dim * num_attention_heads

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "q_proj" in new_key:
            tensor = tensor.view(num_attention_heads, head_dim, hidden_size).reshape(query_dim, hidden_size)
            tensor = permute_for_rope(tensor, num_attention_heads, query_dim, hidden_size)
        elif "k_proj" in new_key:
            tensor = tensor.view(num_key_value_heads, head_dim, hidden_size).reshape(key_value_dim, hidden_size)
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, hidden_size)
        elif "v_proj" in new_key:
            tensor = tensor.view(num_key_value_heads, head_dim, hidden_size).reshape(key_value_dim, hidden_size)

        new_dict[new_key] = tensor
    return new_dict


def get_concat_dim(key):
    """Return the dimension to concatenate the weights on."""
    concat_dim_1 = [
        r"model.embed_tokens.weight",
        r"model.layers.(\d+).self_attn.o_proj.weight",
        r"model.layers.(\d+).mlp.down_proj.weight",
    ]
    if any(re.search(pattern, key) for pattern in concat_dim_1):
        return 1
    return 0


def convert_state_dict_sharded(loaded_shards: list[dict], config: MistralConfig):
    """Convert the state dict, when a single `nn.Module` is sharded accross different files."""
    new_dict = {}

    num_shards = len(loaded_shards)

    n_heads = config.num_attention_heads
    dim = config.hidden_size
    dims_per_head = dim // n_heads
    num_key_value_heads = config.num_key_value_heads
    n_heads_per_shard = n_heads // num_shards
    num_local_key_value_heads = num_key_value_heads // num_shards
    key_value_dim = dim if n_heads == num_key_value_heads else dims_per_head * num_local_key_value_heads

    original_keys = loaded_shards[0].keys()
    for old_key in original_keys:
        new_key = map_old_key_to_new(old_key)
        cat_dim = get_concat_dim(new_key)

        if "q_proj" in new_key:
            tensor = torch.cat(
                [shard.pop(old_key).view(n_heads_per_shard, dims_per_head, dim) for shard in loaded_shards],
                dim=cat_dim,
            ).reshape(dim, dim)
            tensor = permute_for_rope(tensor, n_heads, dim, dim)
        elif "k_proj" in new_key:
            tensor = torch.cat(
                [shard.pop(old_key).view(num_local_key_value_heads, dims_per_head, dim) for shard in loaded_shards],
                dim=cat_dim,
            ).reshape(key_value_dim, dim)
            tensor = permute_for_rope(tensor, num_key_value_heads, key_value_dim, dim)
        elif "v_proj" in new_key:
            tensor = torch.cat(
                [shard.pop(old_key).view(num_local_key_value_heads, dims_per_head, dim) for shard in loaded_shards],
                dim=cat_dim,
            ).reshape(key_value_dim, dim)
        elif "input_layernorm" in new_key or "post_attention_layernorm" in new_key:
            tensor = loaded_shards[0][old_key].clone()
        elif "model.norm.weight" in new_key:
            tensor = loaded_shards[0][old_key]
        else:
            tensor = torch.cat([shard.pop(old_key) for shard in loaded_shards], dim=cat_dim)

        new_dict[new_key] = tensor

    return new_dict


def convert_config(original_config: dict, max_position_embeddings: int = 32768):
    key_mapping = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "intermediate_size": "hidden_dim",
        "num_attention_heads": "n_heads",
        "rms_norm_eps": "norm_eps",
    }
    similar_keys_to_keep = [
        "head_dim",
        "vocab_size",
    ]

    new_config_kwargs = {k: original_config[v] for k, v in key_mapping.items()}
    new_config_kwargs.update({k: v for k, v in original_config.items() if k in similar_keys_to_keep})

    # These are not always defined depending on `params.json`
    new_config_kwargs["sliding_window"] = original_config.get("sliding_window", None)
    new_config_kwargs["num_key_value_heads"] = original_config.get(
        "n_kv_heads", new_config_kwargs["num_attention_heads"]
    )
    new_config_kwargs["rope_theta"] = original_config.get("rope_theta", 10000.0)
    new_config_kwargs["max_position_embeddings"] = original_config.get("max_seq_len", max_position_embeddings)

    # This may sometimes be a string in `params.json`
    if new_config_kwargs["sliding_window"] is not None:
        new_config_kwargs["sliding_window"] = int(new_config_kwargs["sliding_window"])

    new_config = MistralConfig(**new_config_kwargs)
    return new_config


def convert_and_write_model(input_dir: str, output_dir: str, max_position_embeddings: int, modules_are_split: bool):
    """Convert the model and save it (this implicitly save the config as well)."""
    params = read_json(os.path.join(input_dir, "params.json"))
    config = convert_config(params, max_position_embeddings)

    full_state_dict = {}
    # The model may be split between different files, but a single nn.Module is always fully present in a single file
    if not modules_are_split:
        shards = [file for file in os.listdir(input_dir) if file.endswith(".safetensors")]
        for shard_file in shards:
            original_state_dict = load_file(os.path.join(input_dir, shard_file))
            new_dict = convert_state_dict(original_state_dict, config)
            full_state_dict.update(new_dict)
    # A single nn.Module is split between different checkpoint files
    else:
        shards = [file for file in os.listdir(input_dir) if re.match(r"consolidated.\d+.pth", file)]
        shards = sorted(shards, key=lambda x: int(x.split(".")[1]))
        loaded_shards = [torch.load(os.path.join(input_dir, file), map_location="cpu") for file in shards]
        full_state_dict = convert_state_dict_sharded(loaded_shards, config)

    # Load weights into model and resave them
    with torch.device("meta"):
        model = MistralForCausalLM(config)
    model.load_state_dict(full_state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


class MistralConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, vocab: str):
        bpe_ranks = vocab
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in self.additional_special_tokens:
                vocab[token_bytes_to_string(token)] = idx
                if len(token) == 1:
                    continue
                local = []
                for index in range(1, len(token)):
                    piece_l, piece_r = token[:index], token[index:]
                    if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                        local.append((piece_l, piece_r, rank))
                local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
                merges.extend(local)
            else:
                vocab[token] = idx
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


def convert_and_write_tokenizer(input_dir: str, output_dir: str, tokenizer_template_name: str = ""):
    """Convert the tokenizer and save it."""
    # Tekken format -- need to use the Converter
    if "tekken.json" in os.listdir(input_dir):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tokenizer_file = os.path.join(input_dir, "tekken.json")
        mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)

        # Extract vocab and special tokens
        vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
        all_special = [
            token.value if hasattr(token, "value") else token
            for token in mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens
        ]
        specials_tokens = {token: all_special.index(token) for token in all_special}
        specials_tokens.update(vocab)
        vocab = specials_tokens

        # Convert
        tokenizer = LlamaTokenizerFast(
            tokenizer_object=MistralConverter(vocab=vocab, additional_special_tokens=all_special).converted(),
        )

        # Post-process
        tokenizer.add_special_tokens({"additional_special_tokens": all_special})

    else:
        # May have .v3 or .v7 at the end
        tokenizer_file = [file for file in os.listdir(input_dir) if "tokenizer.model" in file][0]
        tokenizer = LlamaTokenizerFast(os.path.join(input_dir, tokenizer_file))

    # Load a chat template from another model
    if tokenizer_template_name != "":
        template_tok = AutoTokenizer.from_pretrained(tokenizer_template_name)
        tokenizer.chat_template = template_tok.chat_template

    # Finally save it
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="",
        help="Another model name from which to copy the chat template.",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=32768,
        help="`max_position_embeddings` field in the config. This needs to be manually passed (not present anywhere otherwise).",
    )
    parser.add_argument(
        "--modules_are_split",
        action="store_true",
        help="If passed, then the weights of a single `nn.Module` are assumed to be split between different files.",
    )
    parser.add_argument(
        "--tokenizer_only",
        action="store_true",
        help="If passed, will only convert the tokenizer.",
    )

    args = parser.parse_args()

    if not args.tokenizer_only:
        convert_and_write_model(args.input_dir, args.output_dir, args.max_position_embeddings, args.modules_are_split)
    convert_and_write_tokenizer(args.input_dir, args.output_dir, args.template_name)


if __name__ == "__main__":
    main()
