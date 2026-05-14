# Copyright 2026 Zyphra and the HuggingFace Inc. team. All rights reserved.
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
"""Convert original alternating-layer ZAYA checkpoints to the Transformers-native decoder-layer layout."""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import ZayaConfig


_DEFAULT_ROPE_THETA = 5_000_000.0
_DEFAULT_SWA_ROPE_THETA = 10_000.0
_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LOCAL_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.zaya_block\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight$"
)

_UNUSED_CONFIG_KEYS = (
    "cca",
    "num_query_groups",
    "intermediate_size",
    "ffn_hidden_size",
    "moe_router_topk",
    "norm_epsilon",
    "zaya_mlp_expansion",
    "activation_func",
    "normalization",
    "add_bias_linear",
    "gated_linear_unit",
    "fused_add_norm",
    "apply_rope_fusion",
    "bias_activation_fusion",
    "activation_func_fp8_input_store",
    "clamp_temp",
    "kv_channels",
    "mamba_cache_dtype",
    "residual_in_fp32",
    "rope_scaling",
    "scale_residual_merge",
    "zaya_high_prec",
    "zaya_use_mod",
    "zaya_use_eda",
)


def _rename_common(rest: str) -> str:
    replacements = (
        ("self_attn.qkv.conv_qk.0.", "self_attn.qkv_proj.conv_qk_depthwise."),
        ("self_attn.qkv.conv_qk.1.", "self_attn.qkv_proj.conv_qk_grouped."),
        ("self_attn.qkv.temp", "self_attn.qk_norm.temp"),
        ("self_attn.qkv.linear_q.", "self_attn.qkv_proj.q_proj."),
        ("self_attn.qkv.linear_k.", "self_attn.qkv_proj.k_proj."),
        ("self_attn.qkv.val_proj1.", "self_attn.qkv_proj.v_proj_current."),
        ("self_attn.qkv.val_proj2.", "self_attn.qkv_proj.v_proj_delayed."),
        ("self_attn.qkv.", "self_attn.qkv_proj."),
        ("zaya_block.router.rmsnorm_eda.", "mlp.gate.router_mlp.rmsnorm_eda."),
        ("zaya_block.router.router_mlp.0.", "mlp.gate.router_mlp.fc1."),
        ("zaya_block.router.router_mlp.2.", "mlp.gate.router_mlp.fc2."),
        ("zaya_block.router.router_mlp.4.", "mlp.gate.router_mlp.out_proj."),
        ("zaya_block.router.", "mlp.gate."),
        ("zaya_block.", "mlp."),
    )
    for old, new in replacements:
        if rest.startswith(old):
            return new + rest.removeprefix(old)
    return rest


def _expert_target(name: str) -> tuple[str, int] | None:
    match = _LOCAL_EXPERT_PATTERN.match(name)
    if match is None:
        return None

    old_layer_idx = int(match.group(1))
    if old_layer_idx % 2 != 1:
        raise ValueError(f"Expert weights are expected on odd ZAYA layers, got: {name}")

    new_layer_idx = old_layer_idx // 2
    expert_idx = int(match.group(2))
    projection = "gate_up_proj" if match.group(3) == "1" else "down_proj"
    target = f"model.layers.{new_layer_idx}.mlp.experts.{projection}"
    return target, expert_idx


def convert_weight_name(name: str, old_num_hidden_layers: int | None = None) -> str | None:
    if _expert_target(name) is not None:
        return None

    match = _LAYER_PATTERN.match(name)
    if match is None:
        if old_num_hidden_layers is not None and name.startswith("model.res_scale."):
            new_layer_idx = old_num_hidden_layers // 2 - 1
            return f"model.layers.{new_layer_idx}.post_mlp_residual_scale.{name.removeprefix('model.res_scale.')}"
        return name

    old_layer_idx = int(match.group(1))
    rest = match.group(2)
    new_layer_idx = old_layer_idx // 2

    if old_layer_idx % 2 == 0:
        rest = _rename_common(rest)
        if rest.startswith("self_attn."):
            return f"model.layers.{new_layer_idx}.{rest}"
        if rest.startswith("input_norm."):
            return f"model.layers.{new_layer_idx}.input_layernorm.{rest.removeprefix('input_norm.')}"
        if rest.startswith("res_scale."):
            if old_layer_idx == 0:
                return f"model.input_{rest.removeprefix('res_scale.')}"
            return f"model.layers.{new_layer_idx - 1}.post_mlp_residual_scale.{rest.removeprefix('res_scale.')}"
    else:
        rest = _rename_common(rest)
        if rest.startswith("mlp."):
            return f"model.layers.{new_layer_idx}.{rest}"
        if rest.startswith("input_norm."):
            return f"model.layers.{new_layer_idx}.post_attention_layernorm.{rest.removeprefix('input_norm.')}"
        if rest.startswith("res_scale."):
            return f"model.layers.{new_layer_idx}.post_attention_residual_scale.{rest.removeprefix('res_scale.')}"

    raise ValueError(f"Unexpected ZAYA layer weight name: {name}")


def _to_hybrid_layer_type(layer_type: str) -> str:
    if layer_type == "full_attention":
        return "hybrid"
    if layer_type == "sliding_attention":
        return "hybrid_sliding"
    raise ValueError(f"Unsupported ZAYA layer type: {layer_type}")


def _convert_layer_types(config_dict: dict, old_num_hidden_layers: int, new_num_hidden_layers: int) -> list[str]:
    layer_types = config_dict.get("layer_types")
    if layer_types is not None:
        if len(layer_types) == old_num_hidden_layers:
            return [_to_hybrid_layer_type(layer_type) for layer_type in layer_types[::2]]
        if len(layer_types) == new_num_hidden_layers:
            return [_to_hybrid_layer_type(layer_type) for layer_type in layer_types]
        raise ValueError("`layer_types` must match either the original or converted number of hidden layers.")

    swa_layers = config_dict.get("swa_layers")
    if swa_layers is None:
        return ["hybrid"] * new_num_hidden_layers
    if len(swa_layers) == old_num_hidden_layers:
        swa_layers = swa_layers[::2]
    elif len(swa_layers) != new_num_hidden_layers:
        raise ValueError("`swa_layers` must match either the original or converted number of hidden layers.")
    return ["hybrid" if int(window_size) == 0 else "hybrid_sliding" for window_size in swa_layers]


def convert_config(input_dir: Path, output_dir: Path) -> None:
    config_dict = json.loads((input_dir / "config.json").read_text())
    old_num_hidden_layers = int(config_dict["num_hidden_layers"])
    if old_num_hidden_layers % 2 != 0:
        raise ValueError("Original ZAYA checkpoints must have an even number of alternating attention/MoE layers.")

    new_num_hidden_layers = old_num_hidden_layers // 2
    layer_types = _convert_layer_types(config_dict, old_num_hidden_layers, new_num_hidden_layers)
    partial_rotary_factor = 0.5
    rope_theta = config_dict.get("rope_theta", _DEFAULT_ROPE_THETA)
    swa_rotary_base = config_dict.get("swa_rotary_base", _DEFAULT_SWA_ROPE_THETA)
    rms_norm_eps = config_dict.get("rms_norm_eps", config_dict.get("norm_epsilon", ZayaConfig.rms_norm_eps))
    router_hidden_size = config_dict.get(
        "router_hidden_size", config_dict.get("zaya_mlp_expansion", ZayaConfig.router_hidden_size)
    )
    expert_ffn_size = config_dict.get("intermediate_size", config_dict.get("ffn_hidden_size"))
    moe_intermediate_size = expert_ffn_size // 2 if expert_ffn_size is not None else ZayaConfig.moe_intermediate_size
    num_experts_per_tok = config_dict.get(
        "num_experts_per_tok", config_dict.get("moe_router_topk", ZayaConfig.num_experts_per_tok)
    )

    swa_layers = config_dict.get("swa_layers") or []
    sliding_window = config_dict.get("sliding_window")
    if sliding_window is None:
        positive_windows = [int(window_size) for window_size in swa_layers if int(window_size) > 0]
        # Original ZAYA stores the number of previous tokens attended by SWA layers. Transformers' sliding window
        # is the total local attention span, including the current token.
        sliding_window = max(positive_windows) + 1 if positive_windows else None

    rope_parameters = {
        "hybrid": {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        },
        "hybrid_sliding": {
            "rope_type": "default",
            "rope_theta": swa_rotary_base,
            "partial_rotary_factor": partial_rotary_factor,
        },
    }

    for key in (*_UNUSED_CONFIG_KEYS, "swa_layers", "rope_theta", "swa_rotary_base"):
        config_dict.pop(key, None)

    config_dict.update(
        {
            "architectures": ["ZayaForCausalLM"],
            "num_hidden_layers": new_num_hidden_layers,
            "moe_intermediate_size": moe_intermediate_size,
            "num_experts_per_tok": num_experts_per_tok,
            "rms_norm_eps": rms_norm_eps,
            "router_hidden_size": router_hidden_size,
            "layer_types": layer_types,
            "sliding_window": sliding_window,
            "rope_parameters": rope_parameters,
        }
    )
    ZayaConfig(**config_dict).save_pretrained(output_dir)


def copy_non_weight_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.name == "config.json":
            continue
        if path.name.endswith(".safetensors") or path.name.endswith(".bin"):
            continue
        if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}:
            continue

        output_path = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(path, output_path)


def _build_weight_plan(input_dir: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str], dict]:
    index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    old_weight_map = index["weight_map"]
    old_num_hidden_layers = int(json.loads((input_dir / "config.json").read_text())["num_hidden_layers"])
    converted_weight_map = {}
    normal_sources_by_output_file = defaultdict(list)
    expert_sources_by_target = defaultdict(list)
    output_file_by_target = {}

    for source_key, filename in old_weight_map.items():
        expert_info = _expert_target(source_key)
        if expert_info is not None:
            target_key, expert_idx = expert_info
            expert_sources_by_target[target_key].append((expert_idx, source_key))
            output_file_by_target.setdefault(target_key, filename)
            converted_weight_map[target_key] = output_file_by_target[target_key]
            continue

        target_key = convert_weight_name(source_key, old_num_hidden_layers)
        if target_key in converted_weight_map:
            raise ValueError(f"Duplicate converted weight name: {target_key}")
        converted_weight_map[target_key] = filename
        normal_sources_by_output_file[filename].append((source_key, target_key))

    index["weight_map"] = converted_weight_map
    return normal_sources_by_output_file, expert_sources_by_target, output_file_by_target, index


def _load_sources(input_dir: Path, source_keys: list[str], old_weight_map: dict[str, str]) -> dict[str, torch.Tensor]:
    sources_by_file = defaultdict(list)
    for source_key in source_keys:
        sources_by_file[old_weight_map[source_key]].append(source_key)

    tensors = {}
    for filename, keys in sources_by_file.items():
        with safe_open(input_dir / filename, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def convert_safetensors(input_dir: Path, output_dir: Path) -> None:
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors_path = input_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError("Only safetensors ZAYA checkpoints are supported by this converter.")

        old_num_hidden_layers = int(json.loads((input_dir / "config.json").read_text())["num_hidden_layers"])
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            state_dict = {}
            expert_groups = defaultdict(list)
            for key in f.keys():
                expert_info = _expert_target(key)
                if expert_info is not None:
                    target_key, expert_idx = expert_info
                    expert_groups[target_key].append((expert_idx, f.get_tensor(key)))
                    continue
                state_dict[convert_weight_name(key, old_num_hidden_layers)] = f.get_tensor(key)
            for target_key, expert_tensors in expert_groups.items():
                state_dict[target_key] = torch.stack([tensor for _, tensor in sorted(expert_tensors)], dim=0)
        save_file(state_dict, output_dir / "model.safetensors", metadata=metadata)
        return

    old_index = json.loads(index_path.read_text())
    old_weight_map = old_index["weight_map"]
    normal_sources_by_output_file, expert_sources_by_target, output_file_by_target, converted_index = (
        _build_weight_plan(input_dir)
    )
    output_filenames = sorted(set(converted_index["weight_map"].values()))

    metadata_by_file = {}
    for filename in sorted(set(old_weight_map.values())):
        with safe_open(input_dir / filename, framework="pt", device="cpu") as f:
            metadata_by_file[filename] = f.metadata()

    for output_filename in output_filenames:
        shard = {}
        normal_sources = normal_sources_by_output_file.get(output_filename, [])
        source_keys = [source_key for source_key, _ in normal_sources]

        expert_groups_for_shard = {
            target_key: sorted(sources)
            for target_key, sources in expert_sources_by_target.items()
            if output_file_by_target[target_key] == output_filename
        }
        for sources in expert_groups_for_shard.values():
            source_keys.extend(source_key for _, source_key in sources)

        loaded_tensors = _load_sources(input_dir, source_keys, old_weight_map)
        for source_key, target_key in normal_sources:
            shard[target_key] = loaded_tensors[source_key]
        for target_key, sources in expert_groups_for_shard.items():
            shard[target_key] = torch.stack([loaded_tensors[source_key] for _, source_key in sources], dim=0)

        save_file(shard, output_dir / output_filename, metadata=metadata_by_file.get(output_filename))

    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(converted_index, indent=2, sort_keys=True) + "\n"
    )


def convert_checkpoint(input_dir: str, output_dir: str, overwrite: bool = False) -> None:
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Please write the converted checkpoint to a different output directory.")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    copy_non_weight_files(input_path, output_path)
    convert_config(input_path, output_path)
    convert_safetensors(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Path to the original alternating-layer ZAYA checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Path where the converted checkpoint should be written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite a non-empty output directory.")
    args = parser.parse_args()
    convert_checkpoint(args.input_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
