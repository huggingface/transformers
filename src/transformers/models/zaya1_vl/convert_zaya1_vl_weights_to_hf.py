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
"""Convert original ZAYA1-VL checkpoints to the Transformers-native layout."""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import Zaya1VLConfig, Zaya1VLTextConfig


_DEFAULT_SWA_ROPE_THETA = 10_000.0
_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.zaya_block\.experts\.local_experts\.(\d+)\."
    r"(linear_fc[12]|lora_fc[12]\.[01])\.weight$"
)

_UNUSED_CONFIG_KEYS = (
    "activation_func",
    "activation_func_fp8_input_store",
    "add_bias_linear",
    "apply_rope_fusion",
    "ar_threshold",
    "bias_activation_fusion",
    "cca",
    "clamp_temp",
    "ffn_hidden_size",
    "fused_add_norm",
    "gated_linear_unit",
    "lora_rank",
    "moe_router_topk",
    "norm_epsilon",
    "normalization",
    "num_query_groups",
    "projector_hidden_act",
    "residual_in_fp32",
    "rotary_base",
    "scale_residual_merge",
    "temporal_patch_size",
    "use_lora_att",
    "use_rope_scaling",
    "zaya_mlp_expansion",
    "zaya_use_eda",
    "zaya_use_mod",
)

_VISION_CONFIG_UNUSED_KEYS = ("_attn_implementation_autoset", "in_chans", "model_type", "torch_dtype")


def _rename_common(rest: str) -> str:
    replacements = (
        ("self_attn.qkv.conv_qk.0.", "self_attn.qkv_proj.conv_qk_depthwise."),
        ("self_attn.qkv.conv_qk.1.", "self_attn.qkv_proj.conv_qk_grouped."),
        ("self_attn.qkv.temp", "self_attn.qk_norm.temp"),
        ("self_attn.qkv.linear_q.", "self_attn.qkv_proj.q_proj."),
        ("self_attn.qkv.linear_k.", "self_attn.qkv_proj.k_proj."),
        ("self_attn.qkv.val_proj1.", "self_attn.qkv_proj.v_proj_current."),
        ("self_attn.qkv.val_proj2.", "self_attn.qkv_proj.v_proj_delayed."),
        ("self_attn.qkv.lora_linear_q.0.", "self_attn.qkv_proj.q_lora_a."),
        ("self_attn.qkv.lora_linear_q.1.", "self_attn.qkv_proj.q_lora_b."),
        ("self_attn.qkv.lora_linear_k.0.", "self_attn.qkv_proj.k_lora_a."),
        ("self_attn.qkv.lora_linear_k.1.", "self_attn.qkv_proj.k_lora_b."),
        ("self_attn.qkv.lora_val_proj1.0.", "self_attn.qkv_proj.v_current_lora_a."),
        ("self_attn.qkv.lora_val_proj1.1.", "self_attn.qkv_proj.v_current_lora_b."),
        ("self_attn.qkv.lora_val_proj2.0.", "self_attn.qkv_proj.v_delayed_lora_a."),
        ("self_attn.qkv.lora_val_proj2.1.", "self_attn.qkv_proj.v_delayed_lora_b."),
        ("self_attn.lora_linear_o.0.", "self_attn.o_lora_a."),
        ("self_attn.lora_linear_o.1.", "self_attn.o_lora_b."),
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
    match = _EXPERT_PATTERN.match(name)
    if match is None:
        return None

    layer_idx = int(match.group(1))
    expert_idx = int(match.group(2))
    projection = match.group(3)
    target_projection = {
        "linear_fc1": "gate_up_proj",
        "linear_fc2": "down_proj",
        "lora_fc1.0": "lora_gate_up_proj_a",
        "lora_fc1.1": "lora_gate_up_proj_b",
        "lora_fc2.0": "lora_down_proj_a",
        "lora_fc2.1": "lora_down_proj_b",
    }[projection]
    target = f"model.language_model.layers.{layer_idx}.mlp.experts.{target_projection}"
    return target, expert_idx


def convert_weight_name(name: str, num_hidden_layers: int) -> str | None:
    if _expert_target(name) is not None:
        return None
    if name.startswith("vision_tower."):
        return f"model.visual.{name.removeprefix('vision_tower.')}"
    if name.startswith("model.embed_tokens."):
        return f"model.language_model.embed_tokens.{name.removeprefix('model.embed_tokens.')}"
    if name.startswith("model.final_norm."):
        return f"model.language_model.final_norm.{name.removeprefix('model.final_norm.')}"
    if name.startswith("model.res_scale."):
        return (
            f"model.language_model.layers.{num_hidden_layers - 1}.post_mlp_residual_scale."
            f"{name.removeprefix('model.res_scale.')}"
        )

    match = _LAYER_PATTERN.match(name)
    if match is None:
        return name

    layer_idx = int(match.group(1))
    rest = match.group(2)
    if rest.startswith("attn."):
        rest = _rename_common(rest.removeprefix("attn."))
        if rest.startswith("self_attn."):
            return f"model.language_model.layers.{layer_idx}.{rest}"
        if rest.startswith("input_norm."):
            return f"model.language_model.layers.{layer_idx}.input_layernorm.{rest.removeprefix('input_norm.')}"
        if rest.startswith("res_scale."):
            if layer_idx == 0:
                return f"model.language_model.input_{rest.removeprefix('res_scale.')}"
            return (
                f"model.language_model.layers.{layer_idx - 1}.post_mlp_residual_scale."
                f"{rest.removeprefix('res_scale.')}"
            )
    if rest.startswith("mlp."):
        rest = _rename_common(rest.removeprefix("mlp."))
        if rest.startswith("mlp."):
            return f"model.language_model.layers.{layer_idx}.{rest}"
        if rest.startswith("input_norm."):
            return (
                f"model.language_model.layers.{layer_idx}.post_attention_layernorm.{rest.removeprefix('input_norm.')}"
            )
        if rest.startswith("res_scale."):
            return (
                f"model.language_model.layers.{layer_idx}.post_attention_residual_scale."
                f"{rest.removeprefix('res_scale.')}"
            )

    raise ValueError(f"Unexpected ZAYA1-VL layer weight name: {name}")


def convert_config(input_dir: Path, output_dir: Path) -> None:
    config_dict = json.loads((input_dir / "config.json").read_text())
    num_hidden_layers = int(config_dict["num_hidden_layers"])
    rms_norm_eps = config_dict.get("rms_norm_eps", config_dict.get("norm_epsilon", Zaya1VLTextConfig.rms_norm_eps))
    router_hidden_size = config_dict.get("router_hidden_size", config_dict.get("zaya_mlp_expansion", 256))
    expert_ffn_size = config_dict.get("intermediate_size", config_dict.get("ffn_hidden_size"))
    moe_intermediate_size = (
        expert_ffn_size // 2 if expert_ffn_size is not None else Zaya1VLTextConfig.moe_intermediate_size
    )
    num_experts_per_tok = config_dict.get("num_experts_per_tok", config_dict.get("moe_router_topk", 1))
    rope_theta = config_dict.get("rope_theta", config_dict.get("rotary_base", 1_000_000.0))
    swa_rotary_base = config_dict.get("swa_rotary_base", _DEFAULT_SWA_ROPE_THETA)

    vision_config = dict(config_dict.pop("vision_config", {}))
    vision_config["in_channels"] = vision_config.get("in_channels", vision_config.get("in_chans", 3))
    vision_config.setdefault("depth", 32)
    vision_config.setdefault("hidden_size", 1280)
    vision_config.setdefault("intermediate_size", 3420)
    vision_config.setdefault("num_heads", 16)
    vision_config.setdefault("hidden_act", "silu")
    vision_config.setdefault("patch_size", vision_config.pop("spatial_patch_size", 14))
    vision_config.setdefault("spatial_merge_size", 2)
    vision_config.setdefault("temporal_patch_size", config_dict.get("temporal_patch_size", 1))
    vision_config.setdefault("tokens_per_second", 2)
    vision_config.setdefault("window_size", 112)
    vision_config.setdefault("out_hidden_size", config_dict["hidden_size"])
    vision_config.setdefault("fullatt_block_indexes", [7, 15, 23, 31])
    for key in _VISION_CONFIG_UNUSED_KEYS:
        vision_config.pop(key, None)

    rope_parameters = {
        "hybrid": {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": config_dict.get("rope_pct", 0.5),
        },
        "hybrid_sliding": {
            "rope_type": "default",
            "rope_theta": swa_rotary_base,
            "partial_rotary_factor": config_dict.get("rope_pct", 0.5),
        },
    }

    for key in (*_UNUSED_CONFIG_KEYS, "rope_pct", "swa_layers", "swa_rotary_base"):
        config_dict.pop(key, None)

    image_token_id = config_dict.pop("image_token_id", Zaya1VLConfig.image_token_id)
    vision_start_token_id = config_dict.pop("vision_start_token_id", Zaya1VLConfig.vision_start_token_id)
    vision_end_token_id = config_dict.pop("vision_end_token_id", Zaya1VLConfig.vision_end_token_id)
    tie_word_embeddings = config_dict.get("tie_word_embeddings", Zaya1VLConfig.tie_word_embeddings)
    output_router_logits = config_dict.get("output_router_logits", Zaya1VLConfig.output_router_logits)

    text_config = {
        **config_dict,
        "model_type": "zaya1_vl_text",
        "rms_norm_eps": rms_norm_eps,
        "moe_intermediate_size": moe_intermediate_size,
        "router_hidden_size": router_hidden_size,
        "num_experts_per_tok": num_experts_per_tok,
        "layer_types": ["hybrid"] * num_hidden_layers,
        "rope_parameters": rope_parameters,
    }
    text_config.pop("architectures", None)
    text_config.pop("model_type", None)
    text_config = Zaya1VLTextConfig(**text_config).to_dict()

    config = Zaya1VLConfig(
        architectures=["Zaya1VLForConditionalGeneration"],
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=image_token_id,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
        tie_word_embeddings=tie_word_embeddings,
        output_router_logits=output_router_logits,
    )
    config.save_pretrained(output_dir)


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
            if path.name == "preprocessor_config.json":
                preprocessor_config = json.loads(path.read_text())
                preprocessor_config["processor_class"] = "Zaya1VLProcessor"
                output_path.write_text(json.dumps(preprocessor_config, indent=2, sort_keys=True) + "\n")
            else:
                shutil.copy2(path, output_path)


def _build_weight_plan(input_dir: Path):
    index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    old_weight_map = index["weight_map"]
    num_hidden_layers = int(json.loads((input_dir / "config.json").read_text())["num_hidden_layers"])
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

        target_key = convert_weight_name(source_key, num_hidden_layers)
        if target_key in converted_weight_map:
            raise ValueError(f"Duplicate converted weight name: {target_key}")
        converted_weight_map[target_key] = filename
        normal_sources_by_output_file[filename].append((source_key, target_key))

    index["weight_map"] = converted_weight_map
    return normal_sources_by_output_file, expert_sources_by_target, output_file_by_target, old_weight_map, index


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


def convert_weights(input_dir: Path, output_dir: Path) -> None:
    (
        normal_sources_by_output_file,
        expert_sources_by_target,
        output_file_by_target,
        old_weight_map,
        index,
    ) = _build_weight_plan(input_dir)
    expert_tensors_by_output_file = defaultdict(dict)

    for target_key, indexed_sources in expert_sources_by_target.items():
        indexed_sources = sorted(indexed_sources)
        source_keys = [source_key for _, source_key in indexed_sources]
        sources = _load_sources(input_dir, source_keys, old_weight_map)
        expert_tensors_by_output_file[output_file_by_target[target_key]][target_key] = torch.stack(
            [sources[source_key] for _, source_key in indexed_sources], dim=0
        ).contiguous()

    for filename, source_and_target_keys in normal_sources_by_output_file.items():
        tensors = {}
        with safe_open(input_dir / filename, framework="pt", device="cpu") as f:
            for source_key, target_key in source_and_target_keys:
                tensors[target_key] = f.get_tensor(source_key)
        tensors.update(expert_tensors_by_output_file.pop(filename, {}))
        save_file(tensors, output_dir / filename, metadata={"format": "pt"})

    for filename, tensors in expert_tensors_by_output_file.items():
        save_file(tensors, output_dir / filename, metadata={"format": "pt"})

    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def convert_checkpoint(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_config(input_dir, output_dir)
    copy_non_weight_files(input_dir, output_dir)
    convert_weights(input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    convert_checkpoint(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
