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
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import regex as re
import torch
from safetensors.torch import load_file as safe_load

from transformers import OpfConfig, OpfForTokenClassification, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenEncodingConverter, get_o200k_harmony_special_tokens


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^embedding\.weight$":                         r"model.embed_tokens.weight",
    r"^embedding_norm\.scale$":                     r"model.embedding_norm.weight",
    r"^norm\.scale$":                               r"model.norm.weight",
    r"^unembedding\.weight$":                       r"score.weight",

    r"^block\.(\d+)\.attn\.qkv\.(weight|bias)$":   r"model.layers.\1.attn.qkv_proj.\2",
    r"^block\.(\d+)\.attn\.norm\.scale$":          r"model.layers.\1.attn.norm.weight",
    r"^block\.(\d+)\.attn\.out\.(weight|bias)$":   r"model.layers.\1.attn.o_proj.\2",
    r"^block\.(\d+)\.attn\.sinks$":                r"model.layers.\1.attn.sinks",

    r"^block\.(\d+)\.mlp\.norm\.scale$":           r"model.layers.\1.mlp.norm.weight",
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


class OpfConverter(TikTokenEncodingConverter):
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


def build_config(original_config: dict[str, object], state_dict: dict[str, torch.Tensor]) -> OpfConfig:
    initial_context_length = int(original_config.get("initial_context_length", 4096))
    rope_theta = float(original_config.get("rope_theta", 150000.0))
    rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": rope_theta,
        "factor": float(original_config.get("rope_scaling_factor", 32.0)),
        "beta_fast": float(original_config.get("rope_ntk_beta", 32.0)),
        "beta_slow": float(original_config.get("rope_ntk_alpha", 1.0)),
        "truncate": False,
        "original_max_position_embeddings": initial_context_length,
    }

    config = OpfConfig(
        vocab_size=int(state_dict["embedding.weight"].shape[0]),
        num_labels=int(original_config.get("num_labels", state_dict["unembedding.weight"].shape[0])),
        hidden_size=int(original_config["hidden_size"]),
        intermediate_size=int(original_config["intermediate_size"]),
        num_hidden_layers=int(original_config["num_hidden_layers"]),
        num_local_experts=int(original_config.get("num_experts", original_config.get("num_local_experts", 128))),
        num_experts_per_tok=int(
            original_config.get("experts_per_token", original_config.get("num_experts_per_tok", 4))
        ),
        head_dim=int(original_config["head_dim"]),
        num_attention_heads=int(original_config["num_attention_heads"]),
        num_key_value_heads=int(original_config["num_key_value_heads"]),
        sliding_window=int(original_config["sliding_window"]),
        bidirectional_left_context=int(original_config.get("bidirectional_left_context", 128)),
        bidirectional_right_context=int(original_config.get("bidirectional_right_context", 128)),
        initial_context_length=initial_context_length,
        max_position_embeddings=int(original_config.get("max_position_embeddings", 131072)),
        default_n_ctx=int(original_config.get("default_n_ctx", 128000)),
        rope_theta=rope_theta,
        rope_parameters=rope_parameters,
    )
    config.architectures = ["OpfForTokenClassification"]
    return config


def split_qkv_tensor(tensor: torch.Tensor, config: OpfConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len = config.num_attention_heads * config.head_dim
    kv_len = config.num_key_value_heads * config.head_dim
    q = tensor[:q_len]
    k = tensor[q_len : q_len + kv_len]
    v = tensor[q_len + kv_len : q_len + 2 * kv_len]
    return q, k, v


def convert_state_dict(original_state_dict: dict[str, torch.Tensor], config: OpfConfig) -> dict[str, torch.Tensor]:
    new_keys = convert_old_keys_to_new_keys(original_state_dict.keys())
    converted = {}

    for key, tensor in original_state_dict.items():
        if key not in new_keys:
            raise KeyError(f"Unrecognized OPF checkpoint tensor: {key}")
        new_key = new_keys[key]

        if re.search("qkv_proj", new_key):
            q, k, v = split_qkv_tensor(tensor, config)
            q_key = re.sub(r"qkv_proj", "q_proj", new_key)
            k_key = re.sub(r"qkv_proj", "k_proj", new_key)
            v_key = re.sub(r"qkv_proj", "v_proj", new_key)
            converted[q_key] = q.contiguous()
            converted[k_key] = k.contiguous()
            converted[v_key] = v.contiguous()
        elif re.search(r"norm\.weight|sinks", new_key):
            converted[new_key] = tensor.float().contiguous()
        else:
            converted[new_key] = tensor.contiguous()

    return converted


def write_tokenizer(save_dir: str, model_max_length: int | None):
    converter = OpfConverter(vocab_file="o200k_base", model_max_length=model_max_length)
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
        model = OpfForTokenClassification(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(output_path)
    del model, state_dict
    gc.collect()

    write_tokenizer(str(output_path), model_max_length=config.default_n_ctx)
    calibration_path = input_path / "viterbi_calibration.json"
    if calibration_path.exists():
        shutil.copy2(calibration_path, output_path / "viterbi_calibration.json")

    OpfForTokenClassification.from_pretrained(output_path).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to the original OPF/OSPF checkpoint directory.")
    parser.add_argument("--output_dir", required=True, help="Path to write the converted Transformers checkpoint.")
    args = parser.parse_args()
    write_model(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
