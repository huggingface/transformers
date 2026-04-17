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

"""Convert a Cisco Time Series Model (CTSM) 1.0 checkpoint to the transformers format.

Sample usage:

```
python src/transformers/models/ctsm/convert_ctsm_original_to_hf.py \
    --output_dir /output/path \
    --huggingface_repo_id cisco-ai/cisco-time-series-model-1.0
```
"""

import argparse
import os

import torch
from huggingface_hub import snapshot_download

from transformers import CtsmConfig, CtsmModelForPrediction


CTSM_CHECKPOINT_FILENAME = "torch_model.pt"

# CTSM 1.0 public checkpoint ships 15 quantiles spanning [0.01, 0.99].
CTSM_1_0_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]


def _layer_mapping(num_layers: int, hidden_size: int) -> dict[str, str | tuple[str, int]]:
    """Return a mapping `old_key -> new_key` (or `(new_prefix, split_idx)` for fused QKV)."""
    mapping: dict[str, str | tuple[str, int]] = {
        # input tokenizer (residual block)
        "input_ff_layer.hidden_layer.0.weight": "model.input_ff_layer.input_layer.weight",
        "input_ff_layer.hidden_layer.0.bias": "model.input_ff_layer.input_layer.bias",
        "input_ff_layer.output_layer.weight": "model.input_ff_layer.output_layer.weight",
        "input_ff_layer.output_layer.bias": "model.input_ff_layer.output_layer.bias",
        "input_ff_layer.residual_layer.weight": "model.input_ff_layer.residual_layer.weight",
        "input_ff_layer.residual_layer.bias": "model.input_ff_layer.residual_layer.bias",
        # frequency, resolution and special token embeddings
        "freq_emb.weight": "model.freq_emb.weight",
        "multi_resolution.weight": "model.multi_resolution.weight",
        "special_token": "model.special_token",
        # horizon head (residual block)
        "horizon_ff_layer.hidden_layer.0.weight": "horizon_ff_layer.input_layer.weight",
        "horizon_ff_layer.hidden_layer.0.bias": "horizon_ff_layer.input_layer.bias",
        "horizon_ff_layer.output_layer.weight": "horizon_ff_layer.output_layer.weight",
        "horizon_ff_layer.output_layer.bias": "horizon_ff_layer.output_layer.bias",
        "horizon_ff_layer.residual_layer.weight": "horizon_ff_layer.residual_layer.weight",
        "horizon_ff_layer.residual_layer.bias": "horizon_ff_layer.residual_layer.bias",
    }

    layer_template = {
        # fused qkv -> split into q, k, v below
        "stacked_transformer.layers.{i}.self_attn.qkv_proj.weight": ("model.layers.{i}.self_attn", "qkv_weight"),
        "stacked_transformer.layers.{i}.self_attn.qkv_proj.bias": ("model.layers.{i}.self_attn", "qkv_bias"),
        "stacked_transformer.layers.{i}.self_attn.o_proj.weight": "model.layers.{i}.self_attn.o_proj.weight",
        "stacked_transformer.layers.{i}.self_attn.o_proj.bias": "model.layers.{i}.self_attn.o_proj.bias",
        "stacked_transformer.layers.{i}.self_attn.scaling": "model.layers.{i}.self_attn.scaling",
        "stacked_transformer.layers.{i}.mlp.gate_proj.weight": "model.layers.{i}.mlp.gate_proj.weight",
        "stacked_transformer.layers.{i}.mlp.gate_proj.bias": "model.layers.{i}.mlp.gate_proj.bias",
        "stacked_transformer.layers.{i}.mlp.down_proj.weight": "model.layers.{i}.mlp.down_proj.weight",
        "stacked_transformer.layers.{i}.mlp.down_proj.bias": "model.layers.{i}.mlp.down_proj.bias",
        "stacked_transformer.layers.{i}.mlp.layer_norm.weight": "model.layers.{i}.mlp.layer_norm.weight",
        "stacked_transformer.layers.{i}.mlp.layer_norm.bias": "model.layers.{i}.mlp.layer_norm.bias",
        "stacked_transformer.layers.{i}.input_layernorm.weight": "model.layers.{i}.input_layernorm.weight",
    }
    for i in range(num_layers):
        for old, new in layer_template.items():
            mapping[old.format(i=i)] = new.format(i=i) if isinstance(new, str) else (new[0].format(i=i), new[1])
    return mapping


def convert_state_dict(original_sd: dict[str, torch.Tensor], hidden_size: int) -> dict[str, torch.Tensor]:
    """Rewrite the original CTSM state dict into the transformers key layout."""
    num_layers = 0
    for key in original_sd:
        if key.startswith("stacked_transformer.layers."):
            idx = int(key.split(".")[2])
            num_layers = max(num_layers, idx + 1)
    if num_layers == 0:
        raise ValueError("No transformer layers found in the original checkpoint.")

    mapping = _layer_mapping(num_layers, hidden_size)
    new_sd: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for old_key, target in mapping.items():
        if old_key not in original_sd:
            missing.append(old_key)
            continue
        tensor = original_sd[old_key]
        if isinstance(target, tuple):
            prefix, kind = target
            if kind == "qkv_weight":
                q, k, v = tensor.split(hidden_size, dim=0)
                new_sd[f"{prefix}.q_proj.weight"] = q.clone()
                new_sd[f"{prefix}.k_proj.weight"] = k.clone()
                new_sd[f"{prefix}.v_proj.weight"] = v.clone()
            elif kind == "qkv_bias":
                q, k, v = tensor.split(hidden_size, dim=0)
                new_sd[f"{prefix}.q_proj.bias"] = q.clone()
                new_sd[f"{prefix}.k_proj.bias"] = k.clone()
                new_sd[f"{prefix}.v_proj.bias"] = v.clone()
            else:
                raise ValueError(f"Unknown fused projection kind: {kind}")
        else:
            new_sd[target] = tensor.clone()
    if missing:
        print(f"[warn] {len(missing)} expected key(s) missing from the original checkpoint (first 5): {missing[:5]}")
    return new_sd


def _infer_config_from_state_dict(original_sd: dict[str, torch.Tensor]) -> CtsmConfig:
    """Infer a `CtsmConfig` from an original CTSM 1.0 state dict."""
    num_layers = 1 + max(
        (int(k.split(".")[2]) for k in original_sd if k.startswith("stacked_transformer.layers.")),
        default=-1,
    )
    hidden_size = original_sd["input_ff_layer.output_layer.weight"].shape[0]
    qkv_out = original_sd["stacked_transformer.layers.0.self_attn.qkv_proj.weight"].shape[0]
    # qkv is [3 * num_heads * head_dim, hidden_size] — split evenly.
    num_heads = 16
    head_dim = qkv_out // (3 * num_heads)
    horizon_out = original_sd["horizon_ff_layer.output_layer.weight"].shape[0]
    horizon_length = 128
    num_outputs = horizon_out // horizon_length
    quantiles = (
        CTSM_1_0_QUANTILES if num_outputs - 1 == len(CTSM_1_0_QUANTILES) else [0.1 * i for i in range(1, num_outputs)]
    )

    return CtsmConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        patch_length=32,
        context_length=512,
        horizon_length=horizon_length,
        quantiles=quantiles,
        use_positional_embedding=False,
        use_resolution_embeddings="multi_resolution.weight" in original_sd,
        use_special_token="special_token" in original_sd,
        agg_factor=60,
        max_position_embeddings=1025,
    )


def write_model(output_dir: str, huggingface_repo_id: str, safe_serialization: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    local_dir = snapshot_download(repo_id=huggingface_repo_id, allow_patterns=[CTSM_CHECKPOINT_FILENAME])
    checkpoint_path = os.path.join(local_dir, CTSM_CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{CTSM_CHECKPOINT_FILENAME} not found in {huggingface_repo_id}")

    print(f"Loading original checkpoint from {checkpoint_path}")
    original_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    config = _infer_config_from_state_dict(original_sd)
    print(
        f"Inferred CtsmConfig: layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} head_dim={config.head_dim} quantiles={len(config.quantiles)}"
    )
    config.save_pretrained(output_dir)

    model = CtsmModelForPrediction(config)
    converted_sd = convert_state_dict(original_sd, hidden_size=config.hidden_size)

    incompatible = model.load_state_dict(converted_sd, strict=False)
    if incompatible.missing_keys:
        print(f"[warn] missing keys after load: {incompatible.missing_keys[:10]}")
    if incompatible.unexpected_keys:
        print(f"[warn] unexpected keys after load: {incompatible.unexpected_keys[:10]}")

    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    print(f"Saved transformers checkpoint to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Where to write the converted HF checkpoint.")
    parser.add_argument(
        "--huggingface_repo_id",
        default="cisco-ai/cisco-time-series-model-1.0",
        help="Original CTSM repo on the Hub.",
    )
    parser.add_argument("--safe_serialization", type=bool, default=True)
    args = parser.parse_args()

    write_model(
        output_dir=args.output_dir,
        huggingface_repo_id=args.huggingface_repo_id,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
