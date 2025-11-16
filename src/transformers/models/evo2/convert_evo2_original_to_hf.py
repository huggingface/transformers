"""Conversion script for original Evo2 checkpoints to Hugging Face format."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Optional

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from transformers import Evo2Config, Evo2ForCausalLM


def _load_original_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    if isinstance(checkpoint, (list, tuple)):
        raise ValueError("Unexpected checkpoint structure. Expected a dictionary with model weights.")
    return checkpoint


def _load_config(path: str) -> Evo2Config:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to parse YAML configuration files. Please install pyyaml.")
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        config_dict = raw.get("model", raw)
        layer_specs = config_dict.get("layers", [])
        layer_types = []
        for layer in layer_specs:
            if isinstance(layer, dict):
                layer_type = layer.get("type") or layer.get("layer_type") or layer.get("block_type")
                if layer_type is None:
                    layer_type = "attention"
                layer_types.append(layer_type.lower())
            else:
                layer_types.append(str(layer).lower())
        config_dict["layer_types"] = layer_types or config_dict.get("layer_types")
        return Evo2Config(**config_dict)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        return Evo2Config(**config_kwargs)
    if os.path.isdir(path):
        return Evo2Config.from_pretrained(path)
    raise ValueError(f"Unsupported config format for '{path}'. Expected directory, JSON, or YAML file.")


def _match_first_available(state_dict: Dict[str, torch.Tensor], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in state_dict:
            return candidate
    return None


def _map_attention_key(layer_idx: int, target_suffix: str, original_state: Dict[str, torch.Tensor]) -> Optional[str]:
    suffix_map = {
        "q_proj.weight": ["attn.wq.weight", "attention.wq.weight", "attention.q_proj.weight"],
        "k_proj.weight": ["attn.wk.weight", "attention.wk.weight", "attention.k_proj.weight"],
        "v_proj.weight": ["attn.wv.weight", "attention.wv.weight", "attention.v_proj.weight"],
        "o_proj.weight": ["attn.wo.weight", "attention.wo.weight", "attention.out_proj.weight"],
        "q_proj.bias": ["attn.wq.bias", "attention.wq.bias", "attention.q_proj.bias"],
        "k_proj.bias": ["attn.wk.bias", "attention.wk.bias", "attention.k_proj.bias"],
        "v_proj.bias": ["attn.wv.bias", "attention.wv.bias", "attention.v_proj.bias"],
        "o_proj.bias": ["attn.wo.bias", "attention.wo.bias", "attention.out_proj.bias"],
        "input_layernorm.weight": ["attn_norm.weight", "input_layernorm.weight"],
        "post_attention_layernorm.weight": ["mlp_norm.weight", "post_attention_layernorm.weight"],
        "input_layernorm.bias": ["attn_norm.bias", "input_layernorm.bias"],
        "post_attention_layernorm.bias": ["mlp_norm.bias", "post_attention_layernorm.bias"],
    }
    candidates = []
    for variant in suffix_map.get(target_suffix, []):
        candidates.append(f"layers.{layer_idx}.{variant}")
        candidates.append(f"model.layers.{layer_idx}.{variant}")
    return _match_first_available(original_state, candidates)


def _map_mlp_key(layer_idx: int, target_suffix: str, original_state: Dict[str, torch.Tensor]) -> Optional[str]:
    suffix_map = {
        "gate_proj.weight": ["mlp.gate_proj.weight", "mlp.w1.weight", "ffn.gate_proj.weight"],
        "up_proj.weight": ["mlp.up_proj.weight", "mlp.w3.weight", "ffn.up_proj.weight"],
        "down_proj.weight": ["mlp.down_proj.weight", "mlp.w2.weight", "ffn.down_proj.weight"],
        "gate_proj.bias": ["mlp.gate_proj.bias", "mlp.w1.bias", "ffn.gate_proj.bias"],
        "up_proj.bias": ["mlp.up_proj.bias", "mlp.w3.bias", "ffn.up_proj.bias"],
        "down_proj.bias": ["mlp.down_proj.bias", "mlp.w2.bias", "ffn.down_proj.bias"],
    }
    candidates = []
    for variant in suffix_map.get(target_suffix, []):
        candidates.append(f"layers.{layer_idx}.{variant}")
        candidates.append(f"model.layers.{layer_idx}.{variant}")
    return _match_first_available(original_state, candidates)


def _map_hyena_key(layer_idx: int, target_suffix: str, original_state: Dict[str, torch.Tensor]) -> Optional[str]:
    suffix_map = {
        "input_layernorm.weight": ["hyena_norm.weight", "input_layernorm.weight"],
        "input_layernorm.bias": ["hyena_norm.bias", "input_layernorm.bias"],
        "post_attention_layernorm.weight": ["mlp_norm.weight", "post_layernorm.weight"],
        "post_attention_layernorm.bias": ["mlp_norm.bias", "post_layernorm.bias"],
        "filter.in_proj.weight": ["hyena.filter.in_proj.weight", "hyena.in_proj.weight"],
        "filter.out_proj.weight": ["hyena.filter.out_proj.weight", "hyena.out_proj.weight"],
        "filter.conv.weight": ["hyena.filter.conv.weight"],
    }
    candidates = []
    for variant in suffix_map.get(target_suffix, []):
        candidates.append(f"layers.{layer_idx}.{variant}")
        candidates.append(f"model.layers.{layer_idx}.{variant}")
    return _match_first_available(original_state, candidates)


def _map_key(target_key: str, config: Evo2Config, original_state: Dict[str, torch.Tensor]) -> Optional[str]:
    if target_key == "model.embed_tokens.weight":
        return _match_first_available(
            original_state,
            [
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "tok_embeddings.weight",
                "embedding.weight",
                "embeddings.word_embeddings.weight",
            ],
        )
    if target_key == "model.norm.weight":
        return _match_first_available(
            original_state,
            ["model.norm.weight", "norm.weight", "final_layer_norm.weight", "rms_norm.weight"],
        )
    if target_key == "lm_head.weight":
        return _match_first_available(original_state, ["lm_head.weight", "output.weight", "head.weight"])

    if target_key.startswith("model.layers."):
        parts = target_key.split(".")
        layer_idx = int(parts[2])
        layer_type = config.layer_types[layer_idx]
        suffix = ".".join(parts[4:]) if parts[3] == "block" else ".".join(parts[3:])
        if layer_type == "attention":
            if suffix.startswith("attention."):
                attn_suffix = suffix[len("attention.") :]
                return _map_attention_key(layer_idx, attn_suffix, original_state)
            if suffix.startswith("mlp."):
                mlp_suffix = suffix[len("mlp.") :]
                return _map_mlp_key(layer_idx, mlp_suffix, original_state)
            if suffix.startswith("hidden_dropout"):
                return None
            if suffix.startswith("input_layernorm") or suffix.startswith("post_attention_layernorm"):
                return _map_attention_key(layer_idx, suffix, original_state)
        else:
            if suffix.startswith("filter."):
                filter_suffix = suffix
                return _map_hyena_key(layer_idx, filter_suffix, original_state)
            if suffix.startswith("mlp."):
                mlp_suffix = suffix[len("mlp.") :]
                return _map_mlp_key(layer_idx, mlp_suffix, original_state)
            if suffix.startswith("input_layernorm") or suffix.startswith("post_attention_layernorm"):
                return _map_hyena_key(layer_idx, suffix, original_state)
        return None
    return None


def convert_checkpoint(original_checkpoint: str, config_path: str, output_dir: str) -> None:
    original_state = _load_original_state_dict(original_checkpoint)
    config = _load_config(config_path)
    model = Evo2ForCausalLM(config)

    target_state = model.state_dict()
    new_state = {}
    missing_keys = []

    for key in target_state.keys():
        source_key = _map_key(key, config, original_state)
        if source_key is None:
            missing_keys.append(key)
            continue
        new_state[key] = original_state[source_key]

    if missing_keys:
        raise KeyError(
            "The following keys could not be mapped from the original checkpoint: " + ", ".join(missing_keys)
        )

    model.load_state_dict(new_state, strict=True)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an original Evo2 checkpoint to Hugging Face format.")
    parser.add_argument("checkpoint", type=str, help="Path to the original .pt checkpoint file")
    parser.add_argument("config", type=str, help="Path to the Evo2 YAML/JSON config or HF directory")
    parser.add_argument("output", type=str, help="Output directory for the converted model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
