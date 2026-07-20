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

from transformers import AutoTokenizer, EsmcTokenizer, EsmFold2Config


# Flat EsmFold2Config field -> dotted path in the research checkpoint's nested config.json.
_LEGACY_FIELD_MAP = {
    "hidden_size": "d_single",
    "pairwise_hidden_size": "d_pair",
    "single_inputs_size": "inputs.d_inputs",
    "sliding_window": "inputs.atom_encoder.swa_window_size",
    "n_relative_residx_bins": "n_relative_residx_bins",
    "n_relative_chain_bins": "n_relative_chain_bins",
    "num_loops": "num_loops",
    "num_diffusion_samples": "num_diffusion_samples",
    "msa_encoder_overwrite": "msa_encoder_overwrite",
    "folding_trunk_num_hidden_layers": "folding_trunk.n_layers",
    "atom_encoder_hidden_size": "inputs.atom_encoder.d_atom",
    "atom_encoder_token_hidden_size": "inputs.atom_encoder.d_token",
    "atom_encoder_num_hidden_layers": "inputs.atom_encoder.n_blocks",
    "atom_encoder_num_attention_heads": "inputs.atom_encoder.n_heads",
    "atom_encoder_expansion_ratio": "inputs.atom_encoder.expansion_ratio",
    "atom_encoder_spatial_rope_base_frequency": "inputs.atom_encoder.spatial_rope_base_frequency",
    "atom_encoder_n_spatial_rope_pairs_per_axis": "inputs.atom_encoder.n_spatial_rope_pairs_per_axis",
    "atom_encoder_n_uid_rope_pairs": "inputs.atom_encoder.n_uid_rope_pairs",
    "atom_encoder_uid_rope_base_frequency": "inputs.atom_encoder.uid_rope_base_frequency",
    "diffusion_sigma_data": "structure_head.diffusion_module.sigma_data",
    "diffusion_atom_hidden_size": "structure_head.diffusion_module.c_atom",
    "diffusion_token_hidden_size": "structure_head.diffusion_module.c_token",
    "diffusion_fourier_dim": "structure_head.diffusion_module.fourier_dim",
    "diffusion_atom_num_blocks": "structure_head.diffusion_module.atom_num_blocks",
    "diffusion_atom_num_heads": "structure_head.diffusion_module.atom_num_heads",
    "diffusion_token_num_blocks": "structure_head.diffusion_module.token_num_blocks",
    "diffusion_token_num_heads": "structure_head.diffusion_module.token_num_heads",
    "diffusion_transition_multiplier": "structure_head.diffusion_module.transition_multiplier",
    "structure_head_distogram_bins": "structure_head.distogram_bins",
    "structure_head_gamma_0": "structure_head.gamma_0",
    "structure_head_gamma_min": "structure_head.gamma_min",
    "structure_head_noise_scale": "structure_head.noise_scale",
    "structure_head_step_scale": "structure_head.step_scale",
    "structure_head_inference_s_max": "structure_head.inference_s_max",
    "structure_head_inference_s_min": "structure_head.inference_s_min",
    "structure_head_inference_p": "structure_head.inference_p",
    "structure_head_inference_num_steps": "structure_head.inference_num_steps",
    "confidence_head_num_hidden_layers": "confidence_head.folding_trunk.n_layers",
    "confidence_head_num_plddt_bins": "confidence_head.num_plddt_bins",
    "confidence_head_num_pde_bins": "confidence_head.num_pde_bins",
    "confidence_head_num_pae_bins": "confidence_head.num_pae_bins",
    "confidence_head_min_dist": "confidence_head.min_dist",
    "confidence_head_max_dist": "confidence_head.max_dist",
    "confidence_head_distogram_bins": "confidence_head.distogram_bins",
    "msa_encoder_hidden_size": "msa_encoder.d_msa",
    "msa_encoder_outer_hidden_size": "msa_encoder.d_hidden",
    "msa_encoder_num_hidden_layers": "msa_encoder.n_layers",
    "msa_encoder_num_attention_heads": "msa_encoder.n_heads_msa",
    "msa_encoder_head_width": "msa_encoder.msa_head_width",
    "lm_encoder_num_hidden_layers": "lm_encoder.n_layers",
    "lm_encoder_lm_dropout": "lm_encoder.lm_dropout",
    "lm_encoder_per_loop_lm_dropout": "lm_encoder.per_loop_lm_dropout",
    "parcae_num_coda_layers": "parcae.coda_n_layers",
}

# Leaves intentionally not carried over: backbone id/size (now in esmc_config), fields the flat
# config re-derives or forces equal to a canonical field, always-on head flags, and training knobs.
_LEGACY_DROP_PATHS = {
    "architectures",
    "model_type",
    "transformers_version",
    "type",  # only the release variant is ported, so the field was dropped entirely
    "esmc_id",
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

# Trunk weight-key rewrites: turn the research checkpoint's keys into the port's module names so
# from_pretrained needs no runtime conversion. Literal substring rewrites; shapes/order unchanged.
_WEIGHT_KEY_RENAMES = (
    ("inputs_embedder.atom_attention_encoder.", "inputs_atom_encoder."),
    (".atom_transformer.", "."),
    ("._engine.", "."),
    (".blocks.", ".layers."),
    (".w_up.", ".gate_up_proj."),
    (".w_down.", ".down_proj."),
    (".lin_swish.", ".ffn.gate_up_proj."),
    (".lin_out.", ".ffn.down_proj."),
    # pair/msa-transition SwiGLU is already fused as w12/w3 in the research checkpoint (unlike the
    # w_up/w_down and lin_swish/lin_out blocks above); the port names every SwiGLU gate_up_proj/down_proj.
    (".ffn.w12.", ".ffn.gate_up_proj."),
    (".ffn.w3.", ".ffn.down_proj."),
    ("fourier.w", "fourier.frequencies"),  # fixed Fourier freq/phase buffers
    ("fourier.b", "fourier.phases"),
    ("output_mlp.0.", "output_fc1."),
    ("output_mlp.2.", "output_fc2."),
    ("adaln_modulation.1.", "adaln_linear."),
    ("base_z_linear.0.", "base_z_input_norm."),
    ("base_z_linear.1.", "base_z_proj."),
    ("base_z_mlp.0.", "base_z_to_pair."),
    ("base_z_mlp.1.", "base_z_output_norm."),
    ("compute_bias.0.", "bias_norm."),
    ("compute_bias.1.", "bias_proj."),
    # ConfidenceHead loose input projections grouped under an input_embedder submodule (dotted
    # suffixes so ``s_to_z.`` does not also match ``s_to_z_transpose.`` / ``s_to_z_prod_*``).
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

# Dead research-checkpoint tensors the port never wired up (vestigial in the fork too — see the PR
# discussion); the port doesn't allocate them, so drop them rather than emit unexpected keys.
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


def flatten_legacy_config(old: dict) -> dict:
    flat = {flat_key: _get_path(old, old_path) for flat_key, old_path in _LEGACY_FIELD_MAP.items()}
    if "dtype" in old:
        flat["dtype"] = old["dtype"]
    unexpected = _leaf_paths(old) - (set(_LEGACY_FIELD_MAP.values()) | _LEGACY_DROP_PATHS | {"dtype"})
    if unexpected:
        raise ValueError(f"unmapped fields in the source ESMFold2 config: {sorted(unexpected)}")
    return flat


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
    flat = flatten_legacy_config(_read_json(esmfold2_dir))
    flat["architectures"] = ["EsmFold2Model"]  # experimental repos ship a now-removed architecture string
    flat["esmc_config"] = _read_json(esmc_dir)
    return EsmFold2Config.from_dict(flat)


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
    return renamed


def merge_state_dict(esmfold2_dir: str, esmc_dir: str) -> dict[str, torch.Tensor]:
    trunk = _load_state_dict(esmfold2_dir)
    if any(k.startswith("esmc.") for k in trunk):
        raise RuntimeError("the ESMFold2 checkpoint already contains esmc.* keys — already bundled?")
    trunk = rename_trunk_keys(trunk)

    # A standalone ESMC checkpoint already stores its encoder under esmc.*; keep those, drop the
    # standalone LM head and TransformerEngine _extra_state blobs.
    esmc = _load_state_dict(esmc_dir)
    kept = {k: v for k, v in esmc.items() if k.startswith("esmc.") and not k.endswith("_extra_state")}
    if not kept:
        raise RuntimeError(f"no esmc.* tensors found in {esmc_dir}")
    return {**trunk, **kept}


def save_tokenizer(esmc_dir: str, output_dir: str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(esmc_dir)
    except Exception:  # backbone dir ships no tokenizer files
        tokenizer = EsmcTokenizer()
    tokenizer.save_pretrained(output_dir)


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
    save_tokenizer(esmc_dir, args.output_dir)
    print(f"bundled {len(state_dict)} tensors to {args.output_dir}")

    if args.push_to_hub:
        HfApi().upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="model")


if __name__ == "__main__":
    main()
