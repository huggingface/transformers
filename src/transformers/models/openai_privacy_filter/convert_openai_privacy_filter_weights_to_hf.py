# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import regex as re
import tiktoken
import torch
from safetensors.torch import load_file as safe_load

from transformers import OpenAIPrivacyFilterConfig, OpenAIPrivacyFilterForTokenClassification, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^embedding\.weight$":                         r"model.embed_tokens.weight",
    r"^norm\.scale$":                               r"model.norm.weight",
    r"^unembedding\.weight$":                       r"score.weight",

    r"^block\.(\d+)\.attn\.qkv\.(weight|bias)$":   r"model.layers.\1.self_attn.qkv_proj.\2",
    r"^block\.(\d+)\.attn\.norm\.scale$":          r"model.layers.\1.input_layernorm.weight",
    r"^block\.(\d+)\.attn\.out\.(weight|bias)$":   r"model.layers.\1.self_attn.o_proj.\2",
    r"^block\.(\d+)\.attn\.sinks$":                r"model.layers.\1.self_attn.sinks",

    r"^block\.(\d+)\.mlp\.norm\.scale$":           r"model.layers.\1.post_attention_layernorm.weight",
    r"^block\.(\d+)\.mlp\.gate\.(weight|bias)$":   r"model.layers.\1.mlp.router.\2",
    r"^block\.(\d+)\.mlp\.swiglu\.weight$":        r"model.layers.\1.mlp.experts.gate_up_proj",
    r"^block\.(\d+)\.mlp\.swiglu\.bias$":          r"model.layers.\1.mlp.experts.gate_up_proj_bias",
    r"^block\.(\d+)\.mlp\.out\.weight$":           r"model.layers.\1.mlp.experts.down_proj",
    r"^block\.(\d+)\.mlp\.out\.bias$":             r"model.layers.\1.mlp.experts.down_proj_bias",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: Iterable[str] | None = None) -> dict[str, str]:
    output_dict = {}
    if state_dict_keys is None:
        return output_dict

    for old_key in state_dict_keys:
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            converted_key, count = re.subn(pattern, replacement, old_key)
            if count:
                output_dict[old_key] = converted_key
                break
    return output_dict


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


def get_o200k_harmony_special_tokens() -> list[str]:
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
    used_ids = set(special_tokens_map.values())
    for token_id in range(199999, 200018):
        if token_id in used_ids:
            continue
        special_tokens_map.setdefault(f"<|reserved_{token_id}|>", token_id)
    return [token for token, _ in sorted(special_tokens_map.items(), key=lambda item: item[1])]


class OpenAIPrivacyFilterConverter(TikTokenConverter):
    def extract_vocab_merges_from_model(self, tiktoken_url: str):
        tokenizer = tiktoken.get_encoding(tiktoken_url)
        self.pattern = tokenizer._pat_str
        bpe_ranks = tokenizer._mergeable_ranks
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(token_bytes):
            return "".join([byte_encoder[ord(char)] for char in token_bytes.decode("latin-1")])

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
            local = sorted(local, key=lambda item: (bpe_ranks[item[0]], bpe_ranks[item[1]]), reverse=False)
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def __init__(self, vocab_file: str, model_max_length: int | None, **kwargs):
        super().__init__(vocab_file, pattern=None, extra_special_tokens=get_o200k_harmony_special_tokens())
        tokenizer = self.converted()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )


def build_config(original_config: dict[str, object], state_dict: dict[str, torch.Tensor]) -> OpenAIPrivacyFilterConfig:
    initial_context_length = int(original_config["initial_context_length"])
    bidirectional_left_context = int(original_config["bidirectional_left_context"])
    bidirectional_right_context = int(original_config["bidirectional_right_context"])
    if bidirectional_left_context != bidirectional_right_context:
        raise ValueError(
            "Privacy Filter conversion expects equal bidirectional context sizes; got "
            f"left={bidirectional_left_context}, right={bidirectional_right_context}."
        )
    rope_theta = float(original_config["rope_theta"])
    rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": rope_theta,
        "factor": float(original_config["rope_scaling_factor"]),
        "beta_fast": float(original_config["rope_ntk_beta"]),
        "beta_slow": float(original_config["rope_ntk_alpha"]),
        "truncate": False,
        "original_max_position_embeddings": initial_context_length,
    }

    config = OpenAIPrivacyFilterConfig(
        vocab_size=int(state_dict["embedding.weight"].shape[0]),
        num_labels=int(state_dict["unembedding.weight"].shape[0]),
        hidden_size=int(original_config["hidden_size"]),
        intermediate_size=int(original_config["intermediate_size"]),
        num_hidden_layers=int(original_config["num_hidden_layers"]),
        num_local_experts=int(original_config["num_experts"]),
        num_experts_per_tok=int(original_config["experts_per_token"]),
        head_dim=int(original_config["head_dim"]),
        num_attention_heads=int(original_config["num_attention_heads"]),
        num_key_value_heads=int(original_config["num_key_value_heads"]),
        sliding_window=bidirectional_left_context,
        initial_context_length=initial_context_length,
        max_position_embeddings=int(original_config["max_position_embeddings"]),
        default_n_ctx=int(original_config["default_n_ctx"]),
        rope_theta=rope_theta,
        rope_parameters=rope_parameters,
    )
    config.architectures = ["OpenAIPrivacyFilterForTokenClassification"]
    return config


def split_qkv_tensor(
    tensor: torch.Tensor, config: OpenAIPrivacyFilterConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len = config.num_attention_heads * config.head_dim
    kv_len = config.num_key_value_heads * config.head_dim
    q = tensor[:q_len]
    k = tensor[q_len : q_len + kv_len]
    v = tensor[q_len + kv_len : q_len + 2 * kv_len]
    return q, k, v


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], config: OpenAIPrivacyFilterConfig
) -> dict[str, torch.Tensor]:
    new_keys = convert_old_keys_to_new_keys(original_state_dict.keys())
    converted = {}

    for key, tensor in original_state_dict.items():
        if key not in new_keys:
            raise KeyError(f"Unrecognized Privacy Filter checkpoint tensor: {key}")
        new_key = new_keys[key]

        if re.search("qkv_proj", new_key):
            q, k, v = split_qkv_tensor(tensor, config)
            q_key = re.sub(r"qkv_proj", "q_proj", new_key)
            k_key = re.sub(r"qkv_proj", "k_proj", new_key)
            v_key = re.sub(r"qkv_proj", "v_proj", new_key)
            converted[q_key] = q.contiguous()
            converted[k_key] = k.contiguous()
            converted[v_key] = v.contiguous()
        elif re.search(r"sinks", new_key):
            converted[new_key] = (tensor.float() * math.log(2.0)).contiguous()
        else:
            converted[new_key] = tensor.contiguous()

    if "score.bias" not in converted:
        converted["score.bias"] = torch.zeros(config.num_labels, dtype=converted["score.weight"].dtype)

    return converted


def write_tokenizer(save_dir: str, model_max_length: int | None):
    converter = OpenAIPrivacyFilterConverter(vocab_file="o200k_base", model_max_length=model_max_length)
    converter.tokenizer.save_pretrained(save_dir)


def write_model(input_dir: str, output_dir: str):
    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    os.makedirs(output_path, exist_ok=True)

    original_config = json.loads((input_path / "config.json").read_text())
    original_state_dict = {}
    for file in os.listdir(input_path):
        if file.endswith(".safetensors"):
            original_state_dict.update(safe_load(str(input_path / file)))

    config = build_config(original_config, original_state_dict)
    state_dict = convert_state_dict(original_state_dict, config)
    del original_state_dict
    gc.collect()

    with torch.device("meta"):
        model = OpenAIPrivacyFilterForTokenClassification(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(output_path)
    del model, state_dict
    gc.collect()

    write_tokenizer(str(output_path), model_max_length=config.default_n_ctx)
    calibration_path = input_path / "viterbi_calibration.json"
    if calibration_path.exists():
        shutil.copy2(calibration_path, output_path / "viterbi_calibration.json")

    OpenAIPrivacyFilterForTokenClassification.from_pretrained(output_path).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="Path to the original Privacy Filter/OSPF checkpoint directory."
    )
    parser.add_argument("--output_dir", required=True, help="Path to write the converted Transformers checkpoint.")
    args = parser.parse_args()
    write_model(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
