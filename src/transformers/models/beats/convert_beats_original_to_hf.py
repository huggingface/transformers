# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the MIT License.

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from transformers.models.beats import BEATsConfig, BEATsModel


def remap_key(key: str):
    """Map Microsoft BEATs checkpoint keys to HuggingFace BEATsModel keys."""
    # Skip weight_g — we'll compute the actual weight manually
    if key == "encoder.pos_conv.0.weight_g":
        return None
    if key == "encoder.pos_conv.0.weight_v":
        return None  # handled separately

    if key.startswith("post_extract_proj."):
        return "patch_embedding." + key
    if key == "patch_embedding.weight":
        return "patch_embedding.patch_embedding.weight"
    if key == "patch_embedding.bias":
        return "patch_embedding.patch_embedding.bias"
    if key == "layer_norm.weight":
        return "patch_embedding.layer_norm.weight"
    if key == "layer_norm.bias":
        return "patch_embedding.layer_norm.bias"

    return key


def compute_weight_from_weight_norm(weight_v: torch.Tensor, weight_g: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    Reconstruct the actual weight from weight_norm parameters.
    weight = weight_g * (weight_v / ||weight_v||)
    dim=2 means normalization is along dim 2.
    """
    norm = weight_v.norm(dim=dim, keepdim=True)
    weight = weight_g * (weight_v / norm)
    return weight


def convert_beats_checkpoint(checkpoint_path: str, output_dir: str):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    cfg = checkpoint["cfg"]
    config = BEATsConfig(
        input_patch_size=cfg.get("input_patch_size", 16),
        embed_dim=cfg.get("embed_dim", 512),
        conv_bias=cfg.get("conv_bias", False),
        encoder_layers=cfg.get("encoder_layers", 12),
        encoder_embed_dim=cfg.get("encoder_embed_dim", 768),
        encoder_ffn_embed_dim=cfg.get("encoder_ffn_embed_dim", 3072),
        encoder_attention_heads=cfg.get("encoder_attention_heads", 12),
        activation_fn=cfg.get("activation_fn", "gelu"),
        dropout=cfg.get("dropout", 0.1),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        activation_dropout=cfg.get("activation_dropout", 0.0),
        dropout_input=cfg.get("dropout_input", 0.1),
        encoder_layerdrop=cfg.get("encoder_layerdrop", 0.05),
        layer_norm_first=cfg.get("layer_norm_first", False),
        deep_norm=cfg.get("deep_norm", True),
        relative_position_embedding=cfg.get("relative_position_embedding", True),
        num_buckets=cfg.get("num_buckets", 320),
        max_distance=cfg.get("max_distance", 800),
        gru_rel_pos=cfg.get("gru_rel_pos", True),
        conv_pos=cfg.get("conv_pos", 128),
        conv_pos_groups=cfg.get("conv_pos_groups", 16),
        grep_linear_units=8,
    )

    print("Config loaded:")
    print(f"  encoder_layers: {config.encoder_layers}")
    print(f"  encoder_embed_dim: {config.encoder_embed_dim}")

    print("\nBuilding HuggingFace BEATsModel...")
    model = BEATsModel(config)
    model.eval()

    # Remap keys
    original_state_dict = checkpoint["model"]
    hf_state_dict = {}

    # Compute actual pos_conv weight from weight_norm params
    weight_v = original_state_dict["encoder.pos_conv.0.weight_v"]
    weight_g = original_state_dict["encoder.pos_conv.0.weight_g"]
    actual_weight = compute_weight_from_weight_norm(weight_v, weight_g, dim=2)
    hf_state_dict["encoder.pos_conv.0.weight"] = actual_weight
    print(f"Computed pos_conv weight from weight_norm: {actual_weight.shape}")

    # Remap all other keys
    for key, value in original_state_dict.items():
        new_key = remap_key(key)
        if new_key is not None:
            hf_state_dict[new_key] = value

    # Load weights
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    print(f"\nMissing keys ({len(missing)}):")
    for k in missing:
        print(f"  {k}")

    print(f"\nUnexpected keys ({len(unexpected)}):")
    for k in unexpected:
        print(f"  {k}")

    if not missing and not unexpected:
        print("All keys matched perfectly!")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Verify
    print("\nVerifying saved model...")
    loaded_model = BEATsModel.from_pretrained(output_dir)
    loaded_model.eval()

    torch.manual_seed(42)
    fbank = torch.randn(1, 100, 128)
    with torch.no_grad():
        output1 = model(fbank)
        output2 = loaded_model(fbank)

    diff = (output1 - output2).abs().max().item()
    print(f"Max difference between original and loaded: {diff:.2e}")
    assert diff < 1e-4, f"Outputs don't match! diff={diff}"
    print("Verification PASSED!")
    print(f"\nDone! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_beats_checkpoint(args.checkpoint_path, args.output_dir)