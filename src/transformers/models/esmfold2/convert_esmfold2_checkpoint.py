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
"""Bundle a research ESMFold2 checkpoint and its ESMC backbone into one transformers checkpoint."""

import argparse
import glob
import json
import os

import torch
from huggingface_hub import HfApi, save_torch_state_dict, snapshot_download
from safetensors.torch import load_file

from transformers import EsmcTokenizer, EsmFold2Config
from transformers.models.esmc.convert_esmc_checkpoint import build_esmc_config_dict, convert_esmc_state_dict


# EsmFold2Config paths spelled the same in the research checkpoint's config.json; renamed ones live
# in ``_LEGACY_RENAMES`` below.
_LEGACY_FIELDS = (
    "n_relative_residx_bins",
    "n_relative_chain_bins",
    "num_loops",
    "num_diffusion_samples",
    "structure_head.diffusion_module.sigma_data",
    "structure_head.diffusion_module.fourier_dim",
    "structure_head.diffusion_module.token_num_blocks",
    "structure_head.diffusion_module.token_num_heads",
    "structure_head.diffusion_module.transition_multiplier",
    "structure_head.distogram_bins",
    "structure_head.gamma_0",
    "structure_head.gamma_min",
    "structure_head.noise_scale",
    "structure_head.step_scale",
    "structure_head.inference_s_max",
    "structure_head.inference_s_min",
    "structure_head.inference_p",
    "structure_head.inference_num_steps",
    "confidence_head.num_plddt_bins",
    "confidence_head.num_pde_bins",
    "confidence_head.num_pae_bins",
    "confidence_head.min_dist",
    "confidence_head.max_dist",
    "confidence_head.distogram_bins",
    "lm_encoder.lm_dropout",
    "lm_encoder.per_loop_lm_dropout",
)

# Port config path -> dotted path in the research checkpoint's config.json, for the renamed fields only.
_LEGACY_RENAMES = {
    "hidden_size": "d_single",
    "pairwise_hidden_size": "d_pair",
    "single_inputs_size": "inputs.d_inputs",
    "sliding_window": "inputs.atom_encoder.swa_window_size",
    "msa_encoder.overwrite": "msa_encoder_overwrite",
    "folding_trunk_num_hidden_layers": "folding_trunk.n_layers",
    "parcae_num_coda_layers": "parcae.coda_n_layers",
    "atom_encoder.hidden_size": "inputs.atom_encoder.d_atom",
    "atom_encoder.num_hidden_layers": "inputs.atom_encoder.n_blocks",
    "atom_encoder.num_attention_heads": "inputs.atom_encoder.n_heads",
    "atom_encoder.expansion_ratio": "inputs.atom_encoder.expansion_ratio",
    "atom_encoder.spatial_rope_base_frequency": "inputs.atom_encoder.spatial_rope_base_frequency",
    "atom_encoder.n_spatial_rope_pairs_per_axis": "inputs.atom_encoder.n_spatial_rope_pairs_per_axis",
    "atom_encoder.n_uid_rope_pairs": "inputs.atom_encoder.n_uid_rope_pairs",
    "atom_encoder.uid_rope_base_frequency": "inputs.atom_encoder.uid_rope_base_frequency",
    "structure_head.diffusion_module.atom_encoder.hidden_size": "structure_head.diffusion_module.c_atom",
    "structure_head.diffusion_module.atom_encoder.num_hidden_layers": "structure_head.diffusion_module.atom_num_blocks",
    "structure_head.diffusion_module.atom_encoder.num_attention_heads": "structure_head.diffusion_module.atom_num_heads",
    "structure_head.diffusion_module.token_hidden_size": "structure_head.diffusion_module.c_token",
    "confidence_head.num_hidden_layers": "confidence_head.folding_trunk.n_layers",
    "msa_encoder.hidden_size": "msa_encoder.d_msa",
    "msa_encoder.outer_hidden_size": "msa_encoder.d_hidden",
    "msa_encoder.num_hidden_layers": "msa_encoder.n_layers",
    "msa_encoder.num_attention_heads": "msa_encoder.n_heads_msa",
    "msa_encoder.head_width": "msa_encoder.msa_head_width",
    "lm_encoder.num_hidden_layers": "lm_encoder.n_layers",
}

_LEGACY_PORT_PATHS = (*_LEGACY_FIELDS, *_LEGACY_RENAMES)

# Leaves not carried over: backbone id/size, re-derived fields, always-on head flags, training knobs.
_LEGACY_DROP_PATHS = {
    "architectures",
    "model_type",
    "transformers_version",
    "type",  # only the release variant is ported, so the field was dropped entirely
    "esmc_id",
    "inputs.atom_encoder.d_token",  # derived from ``single_inputs_size``, not from this half-width
    "lm_d_model",
    "lm_num_layers",
    "lm_dropout",
    "disable_msa_features",
    "force_lm_dropout_during_inference",
    "folding_trunk.n_heads",
    "folding_trunk.dropout",
    "structure_head.train_noise_log_mean",
    "structure_head.train_noise_log_std",
    "structure_head.diffusion_module.c_z",
    "structure_head.diffusion_module.c_s_inputs",
    "structure_head.diffusion_module.relpos_r_max",
    "structure_head.diffusion_module.relpos_s_max",
    "msa_encoder.enabled",  # always built now (every release enables it)
    "lm_encoder.enabled",  # always built now (every release enables it)
    "confidence_head.enabled",
    "confidence_head.folding_trunk.n_heads",
    "confidence_head.folding_trunk.dropout",
    "parcae.enabled",
    "parcae.max_steps",
    "parcae.min_steps",
    "parcae.poisson_mean",
}

# Literal substring rewrites from the research checkpoint's keys to the port's module names, so
# from_pretrained needs no runtime conversion. Shapes and order are unchanged.
_WEIGHT_KEY_RENAMES = (
    ("inputs_embedder.atom_attention_encoder.", "inputs_atom_encoder."),
    (".atom_transformer.", "."),
    ("._engine.", "."),
    (".blocks.", ".layers."),
    (".w_up.", ".gate_up_proj."),
    (".w_down.", ".down_proj."),
    (".lin_swish.", ".ffn.gate_up_proj."),
    (".lin_out.", ".ffn.down_proj."),
    # The pair/msa transitions are already fused as w12/w3, unlike the blocks above.
    (".ffn.w12.", ".ffn.gate_up_proj."),
    (".ffn.w3.", ".ffn.down_proj."),
    ("fourier.w", "fourier.frequencies"),  # fixed Fourier freq/phase buffers
    ("fourier.b", "fourier.phases"),
    ("parcae_", "parcae."),  # loose attributes, now grouped under an ``EsmFold2Parcae`` submodule
    ("output_mlp.0.", "output_fc1."),
    ("output_mlp.2.", "output_fc2."),
    ("adaln_modulation.1.", "adaln_linear."),
    # EsmFold2AdaptiveLayerNorm's conditioning scale and gate/shift projections.
    ("adaln.s_gate.", "adaln.gate_proj."),
    ("adaln.s_shift.", "adaln.shift_proj."),
    ("adaln.s_scale", "adaln.cond_norm.weight"),
    ("base_z_linear.0.", "base_z_input_norm."),
    ("base_z_linear.1.", "base_z_proj."),
    ("base_z_mlp.0.", "base_z_to_pair."),
    ("base_z_mlp.1.", "base_z_output_norm."),
    ("compute_bias.0.", "bias_norm."),
    ("compute_bias.1.", "bias_proj."),
    # Grouped under an input_embedder submodule; the trailing dots keep the prefixes distinct.
    ("confidence_head.s_inputs_norm.", "confidence_head.input_embedder.s_inputs_norm."),
    ("confidence_head.z_norm.", "confidence_head.input_embedder.z_norm."),
    ("confidence_head.s_to_z.", "confidence_head.input_embedder.s_to_z."),
    ("confidence_head.s_to_z_transpose.", "confidence_head.input_embedder.s_to_z_transpose."),
    ("confidence_head.s_to_z_prod_in1.", "confidence_head.input_embedder.s_to_z_prod_in1."),
    ("confidence_head.s_to_z_prod_in2.", "confidence_head.input_embedder.s_to_z_prod_in2."),
    ("confidence_head.s_to_z_prod_out.", "confidence_head.input_embedder.s_to_z_prod_out."),
)
# The SWA attention packed q/k/v into one Wqkv; the port uses separate projections.
_PACKED_QKV_SUFFIX = "attn.Wqkv.weight"

# Dead research-checkpoint tensors, vestigial in the fork too; the port never allocates them.
_WEIGHT_KEY_DROPS = (
    "confidence_head.s_norm.",
    "confidence_head.s_inputs_to_single.",
    "confidence_head.s_input_to_s.",
)


def _read_json(directory: str) -> dict:
    with open(os.path.join(directory, "config.json")) as f:
        return json.load(f)


def _get_path(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        node = node[part]
    return node


def _leaf_paths(cfg: dict, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        paths |= _leaf_paths(value, f"{dotted}.") if isinstance(value, dict) else {dotted}
    return paths


def build_legacy_config(old: dict) -> dict:
    """Reshape the research checkpoint's nested config into the port's nested EsmFold2Config layout.

    Each port path in ``_LEGACY_PORT_PATHS`` reads from the same path in the source config unless
    ``_LEGACY_RENAMES`` overrides it.
    """
    config: dict = {}
    for port_path in _LEGACY_PORT_PATHS:
        node = config
        parts = port_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _get_path(old, _LEGACY_RENAMES.get(port_path, port_path))
    if "dtype" in old:
        config["dtype"] = old["dtype"]
    mapped = {_LEGACY_RENAMES.get(port_path, port_path) for port_path in _LEGACY_PORT_PATHS}
    unexpected = _leaf_paths(old) - (mapped | _LEGACY_DROP_PATHS | {"dtype"})
    if unexpected:
        raise ValueError(f"unmapped fields in the source ESMFold2 config: {sorted(unexpected)}")
    return config


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


def build_config(esmfold2_dir: str, esmc_dir: str) -> EsmFold2Config:
    config = build_legacy_config(_read_json(esmfold2_dir))
    config["architectures"] = ["EsmFold2Model"]  # experimental repos ship a now-removed architecture string
    config["esmc_config"] = build_esmc_config_dict(esmc_dir)
    return EsmFold2Config.from_dict(config)


def rename_trunk_keys(trunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed: dict[str, torch.Tensor] = {}
    for key, tensor in trunk.items():
        if any(drop in key for drop in _WEIGHT_KEY_DROPS):
            continue
        for old, new in _WEIGHT_KEY_RENAMES:
            key = key.replace(old, new)
        if key.endswith(_PACKED_QKV_SUFFIX):
            base = key[: -len("Wqkv.weight")]
            # .clone(): the chunks are views sharing the packed tensor's storage, which safetensors refuses to save.
            q, k, v = (chunk.clone() for chunk in torch.chunk(tensor, 3, dim=0))
            renamed.update({base + "q_proj.weight": q, base + "k_proj.weight": k, base + "v_proj.weight": v})
        else:
            renamed[key] = tensor
    return _standardize_attention_keys(_fuse_transition_swiglu(renamed))


def _standardize_attention_keys(renamed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Bring the two multi-head attention modules onto the standard transformers layout: split the
    pair-bias attention's fused ``kv_proj`` into ``k_proj``/``v_proj`` (k first, matching the model's
    ``chunk`` order) and rename the attention output projection ``out_proj`` -> ``o_proj``. Scoped to
    the atom ``.attn.`` modules and the token ``attn_blocks`` so the (non-MHA) row-attention-pooling
    ``out_proj`` is left untouched."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in renamed.items():
        if "attn_blocks." in key and key.endswith(".kv_proj.weight"):
            base = key[: -len("kv_proj.weight")]
            k, v = (chunk.clone() for chunk in torch.chunk(tensor, 2, dim=0))
            out[base + "k_proj.weight"] = k
            out[base + "v_proj.weight"] = v
            continue
        if key.endswith(".attn.out_proj.weight") or ("attn_blocks." in key and key.endswith(".out_proj.weight")):
            key = key.replace(".out_proj.weight", ".o_proj.weight")
        out[key] = tensor
    return out


def _fuse_transition_swiglu(renamed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Consolidate the diffusion-conditioning transitions (``z_transitions``/``s_transitions``) onto the
    shared ``EsmFold2SwiGLU``: fuse the unfused gate/up projections into ``ffn.gate_up_proj`` (gate first,
    matching ``EsmFold2SwiGLU``'s split) and rename the output projection to ``ffn.down_proj``. Scoped to
    the transitions so it never touches the attention ``out_proj``."""
    for key in list(renamed):
        if (".z_transitions." in key or ".s_transitions." in key) and key.endswith(".a_proj.weight"):
            base = key[: -len("a_proj.weight")]
            gate = renamed.pop(base + "a_proj.weight")
            up = renamed.pop(base + "b_proj.weight")
            renamed[base + "ffn.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
        elif (".z_transitions." in key or ".s_transitions." in key) and key.endswith(".out_proj.weight"):
            renamed[key.replace(".out_proj.weight", ".ffn.down_proj.weight")] = renamed.pop(key)
    return renamed


def merge_state_dict(esmfold2_dir: str, esmc_dir: str) -> dict[str, torch.Tensor]:
    trunk = _load_state_dict(esmfold2_dir)
    if any(k.startswith("esmc.") for k in trunk):
        raise RuntimeError("the ESMFold2 checkpoint already contains esmc.* keys — already bundled?")
    trunk = rename_trunk_keys(trunk)

    # A standalone ESMC checkpoint already stores its encoder under esmc.*; keep only those, dropping
    # the masked-LM head the trunk does not use, then run them through the same conversion the
    # standalone ESMC script applies so the bundle needs no runtime renaming either.
    esmc = _load_state_dict(esmc_dir)
    kept = {k: v for k, v in esmc.items() if k.startswith("esmc.")}
    if not kept:
        raise RuntimeError(f"no esmc.* tensors found in {esmc_dir}")
    return {**trunk, **convert_esmc_state_dict(kept)}


def save_tokenizer(output_dir: str) -> None:
    """Write a fresh ESMC tokenizer next to the bundled weights.

    Built rather than copied from the backbone directory on purpose: the ESMC vocabulary is fixed
    biology notation, so there is nothing checkpoint-specific to carry over, and the published
    backbone repos predate the port -- their ``tokenizer_class`` names a class that no longer
    exists, which ``AutoTokenizer`` resolves to the plain backend instead of raising, silently
    dropping the ``chain_break_token`` registration.
    """
    EsmcTokenizer().save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esmfold2", default="biohub/ESMFold2", help="ESMFold2 trunk checkpoint (repo id or dir)")
    parser.add_argument("--esmc", default="biohub/ESMC-6B", help="ESMC backbone checkpoint (repo id or dir)")
    parser.add_argument("--output_dir", required=True, help="where to write the bundled checkpoint")
    parser.add_argument("--push_to_hub", default=None, help="optional repo id to upload the result to")
    args = parser.parse_args()

    esmfold2_dir, esmc_dir = _resolve_dir(args.esmfold2), _resolve_dir(args.esmc)
    config = build_config(esmfold2_dir, esmc_dir)
    state_dict = merge_state_dict(esmfold2_dir, esmc_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    config.save_pretrained(args.output_dir)
    save_torch_state_dict(state_dict, args.output_dir)
    save_tokenizer(args.output_dir)
    print(f"bundled {len(state_dict)} tensors to {args.output_dir}")

    if args.push_to_hub:
        HfApi().upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="model")


if __name__ == "__main__":
    main()
