# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Convert a research ESMC checkpoint to the transformers layout.

The research repos store the encoder under ``esmc.*`` with the upstream module names, pack q/k/v and
the SwiGLU gate/up into single tensors, and carry TransformerEngine ``_extra_state`` blobs. This
script rewrites all of that ahead of time so ``from_pretrained`` needs no runtime conversion.

``convert_esmc_state_dict`` and ``build_esmc_config_dict`` are also imported by
``convert_esmfold2_checkpoint.py``, which embeds an ESMC backbone under ``esmc.*``.
"""

import argparse
import glob
import json
import os

import torch
from huggingface_hub import HfApi, save_torch_state_dict, snapshot_download
from safetensors.torch import load_file

from transformers import EsmcConfig, EsmcTokenizer


# EsmcConfig fields spelled the same in the research checkpoint's config.json.
_LEGACY_FIELDS = (
    "vocab_size",
    "initializer_range",
    "pad_token_id",
    "tie_word_embeddings",
    "mask_token_id",
    "classifier_dropout",
)

# EsmcConfig field -> its name in the research checkpoint's config.json, for the renamed fields only.
_LEGACY_RENAMES = {
    "hidden_size": "d_model",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
}

# Not carried over: re-derived by ``EsmcConfig``, or naming a class the port removed.
_LEGACY_DROP_KEYS = {
    "architectures",  # research repos name a now-removed "ESMCForMaskedLM"
    "model_type",
    "transformers_version",
}

# Literal substring rewrites from the research checkpoint's keys to the port's module names. Applied
# in order to every key; the packed projections below are split afterwards, off the rewritten name.
_WEIGHT_KEY_RENAMES = (
    ("embed.", "embed_tokens."),
    ("transformer.blocks.", "layers."),
    ("transformer.norm.", "norm."),
    ("attn.layernorm_qkv.layer_norm_weight", "input_layernorm.weight"),
    ("attn.layernorm_qkv.layer_norm_bias", "input_layernorm.bias"),
    ("attn.q_ln.", "self_attn.q_norm."),
    ("attn.k_ln.", "self_attn.k_norm."),
    ("attn.out_proj.", "self_attn.o_proj."),
    ("ffn.layer_norm_weight", "post_attention_layernorm.weight"),
    ("ffn.layer_norm_bias", "post_attention_layernorm.bias"),
    ("ffn.fc2_weight", "mlp.down_proj.weight"),
    # The masked-LM head is a plain nn.Sequential upstream.
    ("lm_head.0.", "lm_head.dense."),
    ("lm_head.2.", "lm_head.layer_norm."),
    ("lm_head.3.", "lm_head.decoder."),
)

# Packed tensors the port splits into separate projections, keyed by the rewritten suffix. The split
# order matches the order the model concatenates them in.
_PACKED_QKV_SUFFIX = "attn.layernorm_qkv.weight"
_PACKED_GATE_UP_SUFFIX = "ffn.fc1_weight"

# TransformerEngine quantization state, meaningless outside the upstream fused kernels.
_WEIGHT_KEY_DROP_SUFFIX = "_extra_state"


def _read_json(directory: str) -> dict:
    with open(os.path.join(directory, "config.json")) as f:
        return json.load(f)


def build_esmc_config_dict(esmc_dir: str) -> dict:
    """The backbone repo's config.json, reshaped onto ``EsmcConfig``'s field names.

    Raises on any source field that is neither ported nor explicitly dropped, so a new release
    variant cannot silently lose a setting.
    """
    old = _read_json(esmc_dir)
    config = {field: old[field] for field in _LEGACY_FIELDS if field in old}
    for port_field, legacy_field in _LEGACY_RENAMES.items():
        config[port_field] = old[legacy_field]
    if "dtype" in old:
        config["dtype"] = old["dtype"]
    mapped = {*_LEGACY_FIELDS, *_LEGACY_RENAMES.values(), "dtype"}
    unexpected = set(old) - (mapped | _LEGACY_DROP_KEYS)
    if unexpected:
        raise ValueError(f"unmapped fields in the source ESMC config: {sorted(unexpected)}")
    return config


def build_config(esmc_dir: str) -> EsmcConfig:
    config = build_esmc_config_dict(esmc_dir)
    config["architectures"] = ["EsmcForMaskedLM"]
    return EsmcConfig.from_dict(config)


def convert_esmc_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite upstream ESMC keys onto the port's module names, splitting the packed projections.

    Substring-based, so it works whether the keys are top-level (a standalone ESMC checkpoint) or
    nested under a parent prefix (the ESMFold2 bundle's ``esmc.*`` backbone).
    """
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_KEY_DROP_SUFFIX):
            continue
        for old, new in _WEIGHT_KEY_RENAMES:
            key = key.replace(old, new)
        # .clone(): the chunks are views sharing the packed tensor's storage, which safetensors refuses to save.
        if key.endswith(_PACKED_QKV_SUFFIX):
            base = key[: -len(_PACKED_QKV_SUFFIX)]
            q, k, v = (chunk.clone() for chunk in torch.chunk(tensor, 3, dim=0))
            converted.update(
                {
                    base + "self_attn.q_proj.weight": q,
                    base + "self_attn.k_proj.weight": k,
                    base + "self_attn.v_proj.weight": v,
                }
            )
        elif key.endswith(_PACKED_GATE_UP_SUFFIX):
            base = key[: -len(_PACKED_GATE_UP_SUFFIX)]
            gate, up = (chunk.clone() for chunk in torch.chunk(tensor, 2, dim=0))
            converted.update({base + "mlp.gate_proj.weight": gate, base + "mlp.up_proj.weight": up})
        else:
            converted[key] = tensor
    return converted


def _resolve_dir(path_or_repo: str) -> str:
    return path_or_repo if os.path.isdir(path_or_repo) else snapshot_download(path_or_repo)


def _load_state_dict(directory: str) -> dict[str, torch.Tensor]:
    shards = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors weights found in {directory}")
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    return state_dict


def save_tokenizer(output_dir: str) -> None:
    """Write a fresh ESMC tokenizer next to the converted weights.

    Built rather than copied from the source directory on purpose: the ESMC vocabulary is fixed
    biology notation, so there is nothing checkpoint-specific to carry over, and the research repos
    predate the port -- their ``tokenizer_class`` names a class that no longer exists, which
    ``AutoTokenizer`` resolves to the plain backend instead of raising, silently dropping the
    ``chain_break_token`` registration.
    """
    EsmcTokenizer().save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esmc", default="biohub/ESMC-6B", help="research ESMC checkpoint (repo id or dir)")
    parser.add_argument("--output_dir", required=True, help="where to write the converted checkpoint")
    parser.add_argument("--push_to_hub", default=None, help="optional repo id to upload the result to")
    args = parser.parse_args()

    esmc_dir = _resolve_dir(args.esmc)
    config = build_config(esmc_dir)
    state_dict = convert_esmc_state_dict(_load_state_dict(esmc_dir))

    os.makedirs(args.output_dir, exist_ok=True)
    config.save_pretrained(args.output_dir)
    save_torch_state_dict(state_dict, args.output_dir)
    save_tokenizer(args.output_dir)
    print(f"converted {len(state_dict)} tensors to {args.output_dir}")

    if args.push_to_hub:
        HfApi().upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="model")


if __name__ == "__main__":
    main()
