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
"""
Converts an original GR00T N1.7 checkpoint (e.g. `nvidia/GR00T-N1.7-3B`) to the transformers format: a `Gr00tConfig`, a
`Gr00tProcessor` (with the tokenizer and chat template of the Cosmos-Reason2 backbone and the per-embodiment statistics)
and the weights.

The original weights are renamed on load by the `gr00t` entry of `conversion_mapping.py`, given the converted config;
the original config and processor files use the Isaac-GR00T format and are converted here.

Usage:

```bash
python src/transformers/models/gr00t/convert_gr00t_to_hf.py \
    --checkpoint nvidia/GR00T-N1.7-3B --output_dir gr00t-n1.7-3b-hf

# Repositories holding several checkpoints in subfolders
python src/transformers/models/gr00t/convert_gr00t_to_hf.py \
    --checkpoint nvidia/GR00T-N1.7-LIBERO --subfolder libero_10 --output_dir gr00t-n1.7-libero-10-hf
```

The original files are read with the semantics of the Isaac-GR00T `n1.7-release` tag. Settings the port can't reproduce
faithfully (other architectures, other action or normalization schemes, opt-in masking modes) raise an error instead of
converting to a model that silently behaves differently.

The tokenizer is loaded from the backbone repository named in the checkpoint config (`nvidia/Cosmos-Reason2-2B`), which
is gated: accept its terms on the Hub and log in first.
"""

import argparse
import json
import os

import torch
from huggingface_hub import list_repo_files, snapshot_download

from transformers import (
    AutoConfig,
    AutoProcessor,
    Gr00tConfig,
    Gr00tForConditionalGeneration,
    Gr00tImageProcessor,
    Gr00tProcessor,
)


# Tag -> projector index table of the original processor, merged into `embodiment_id.json` for the tags it lacks (e.g.
# `new_embodiment`, the slot used to fine-tune a new robot): `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` in
# https://github.com/NVIDIA/Isaac-GR00T/blob/n1.7-release/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
ORIGINAL_EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}

# Defaults of the original `Gr00tN1d7Processor.__init__`, used for keys a `processor_config.json` doesn't store (the
# original `from_pretrained` only backfills `clip_outliers`). Released checkpoints store all of them explicitly, which is
# why `Gr00tProcessor` defaults to the released values instead.
ORIGINAL_PROCESSOR_DEFAULTS = {
    "use_percentiles": False,
    "clip_outliers": True,
    "use_relative_action": False,
    "formalize_language": True,
    "use_albumentations": False,
    "apply_sincos_state_encoding": False,
    "use_mean_std": False,
    "exclude_state": False,
}

ORIGINAL_ACTION_REPRESENTATIONS = {"ABSOLUTE", "RELATIVE", "DELTA"}
ORIGINAL_ACTION_TYPES = {"EEF", "NON_EEF"}
ORIGINAL_ACTION_FORMATS = {"DEFAULT", "XYZ_ROT6D", "XYZ_ROTVEC"}

PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


def load_json(directory: str, name: str) -> dict:
    with open(os.path.join(directory, name), encoding="utf-8") as f:
        return json.load(f)


def processor_kwarg(kwargs: dict, key: str):
    value = kwargs.get(key)
    return ORIGINAL_PROCESSOR_DEFAULTS[key] if value is None else value


def validate_original_config(original: dict) -> None:
    """Raises if the original model config describes a model `Gr00tForConditionalGeneration` doesn't implement."""
    problems = []
    architectures = original.get("architectures") or [original.get("model_type")]
    if architectures != ["Gr00tN1d7"] or original.get("model_type", "Gr00tN1d7") != "Gr00tN1d7":
        problems.append(
            f"architecture {architectures} / model_type {original.get('model_type')!r}: only the original "
            "`Gr00tN1d7` architecture is supported, not variants with extra or different modules"
        )
    if original.get("backbone_model_type", "qwen") != "qwen":
        problems.append(f"backbone_model_type {original['backbone_model_type']!r}: only the Qwen3-VL backbone")

    # Architecture choices the port implements (defaults of the original `Gr00tN1d7Config` for missing keys)
    expected_flags = {
        "use_alternate_vl_dit": (True, "the DiT alternates text and image cross-attention"),
        "use_vlln": (True, "the backbone features go through a LayerNorm"),
        "add_pos_embed": (True, "the action tokens get learned position embeddings"),
        "use_vl_self_attention": (True, "the backbone features go through the VL refiner"),
        "exclude_state": (False, "the state is used"),
    }
    for key, (expected, description) in expected_flags.items():
        if original.get(key, expected) != expected:
            problems.append(f"{key}={original[key]}: the port requires {expected} ({description})")
    for key in ("strict_vl_padding_mask", "strict_action_padding_mask"):
        if original.get(key):
            problems.append(f"{key}=True: only the default (legacy) masking of the released checkpoints is supported")
    for key in ("use_tactile", "use_lora"):
        if original.get(key):
            problems.append(f"{key}=True: extra modules of forks of the original code are not supported")
    if original.get("state_history_length", 1) != 1:
        problems.append(f"state_history_length={original['state_history_length']}: only a single state step")

    diffusion = original.get("diffusion_model_cfg") or {}
    refiner = original.get("vl_self_attention_cfg")
    if not refiner or not refiner.get("num_layers"):
        problems.append(
            "vl_self_attention_cfg is empty: models without the VL refiner (e.g. GR00T-H) are not supported"
        )
        refiner = {}
    dit_hidden_size = diffusion.get("num_attention_heads", 0) * diffusion.get("attention_head_dim", 0)
    expected_diffusion = {"norm_type": "ada_norm", "final_dropout": True, "interleave_self_attention": True}
    for key, expected in expected_diffusion.items():
        if diffusion.get(key, expected) != expected:
            problems.append(f"diffusion_model_cfg.{key}={diffusion[key]!r}: the port requires {expected!r}")
    block_indices = diffusion.get("block_indices")
    if block_indices is not None and list(block_indices) != list(range(diffusion.get("num_layers", 0))):
        problems.append(
            "diffusion_model_cfg.block_indices: pruned DiTs are not supported, as the port alternates the "
            "cross-attention targets by block position"
        )
    if diffusion.get("positional_embeddings") is not None:
        problems.append("diffusion_model_cfg.positional_embeddings: the DiT blocks have no positional embeddings")
    if original.get("input_embedding_dim", dit_hidden_size) != dit_hidden_size:
        problems.append(
            f"input_embedding_dim={original['input_embedding_dim']}: must equal the DiT width ({dit_hidden_size})"
        )
    if diffusion.get("output_dim", original.get("hidden_size")) != original.get("hidden_size"):
        problems.append(f"diffusion_model_cfg.output_dim={diffusion['output_dim']}: must equal hidden_size")
    if refiner:
        # Missing keys take the defaults of the original `SelfAttentionTransformer`
        if refiner.get("positional_embeddings", "sinusoidal") is not None:
            problems.append("vl_self_attention_cfg.positional_embeddings: the VL refiner has no positional embeddings")
        if not refiner.get("final_dropout", True):
            problems.append("vl_self_attention_cfg.final_dropout=False: the port requires True")
        refiner_width = refiner.get("num_attention_heads", 0) * refiner.get("attention_head_dim", 0)
        if refiner_width != original.get("backbone_embedding_dim", 2048):
            problems.append(f"the VL refiner width {refiner_width} must equal backbone_embedding_dim")
    if problems:
        raise ValueError("This checkpoint can't be converted faithfully:\n- " + "\n- ".join(problems))


def validate_original_processor(kwargs: dict, embodiment_id_mapping: dict, config: dict) -> None:
    """Raises if the original processor settings use preprocessing `Gr00tProcessor` doesn't implement."""
    problems = []
    if processor_kwarg(kwargs, "apply_sincos_state_encoding"):
        problems.append("apply_sincos_state_encoding=True: sin/cos state encoding is not supported")
    if processor_kwarg(kwargs, "use_mean_std"):
        problems.append("use_mean_std=True: global mean/std normalization is not supported")
    if processor_kwarg(kwargs, "exclude_state"):
        problems.append("exclude_state=True: the port always uses the state")
    if not processor_kwarg(kwargs, "use_albumentations"):
        problems.append(
            "use_albumentations=False: only the albumentations image pipeline (area resize, fractional center crop) "
            "is implemented, not the torchvision one"
        )
    if kwargs.get("strict_action_padding_mask"):
        problems.append("strict_action_padding_mask=True: only the default (legacy) masking is supported")
    for key in ("max_state_dim", "max_action_dim"):
        if key in config and kwargs.get(key) != config[key]:
            problems.append(f"{key}: the processor pads to {kwargs.get(key)} but the model expects {config[key]}")
    use_relative_action = processor_kwarg(kwargs, "use_relative_action")

    for tag, modalities in kwargs["modality_configs"].items():
        for modality in ("state", "action"):
            modality_config = modalities.get(modality) or {}
            # Per-key `sin_cos_embedding_keys` are not checked: the original processor only applies them with
            # `apply_sincos_state_encoding`, rejected above, and otherwise min/max-normalizes those keys like the port.
            # `mean_std_embedding_keys` are supported by `Gr00tProcessor`
            unsupported_keys = ("pass_through_keys", "min_max_embedding_keys")
            for key in unsupported_keys:
                if modality_config.get(key):
                    problems.append(f"{tag}.{modality}.{key}={modality_config[key]}: per-key {key} is not supported")
        action_config = modalities.get("action") or {}
        for key, group in zip(action_config.get("modality_keys", []), action_config.get("action_configs") or []):
            rep, action_type, action_format = group.get("rep"), group.get("type"), group.get("format")
            if (
                rep not in ORIGINAL_ACTION_REPRESENTATIONS
                or action_type not in ORIGINAL_ACTION_TYPES
                or action_format not in ORIGINAL_ACTION_FORMATS
            ):
                problems.append(f"{tag}.action.{key}: unknown action config {rep}/{action_type}/{action_format}")
            elif use_relative_action and rep == "RELATIVE":
                if not (action_type == "NON_EEF" or (action_type == "EEF" and action_format == "XYZ_ROT6D")):
                    problems.append(
                        f"{tag}.action.{key}: relative {action_type} actions in {action_format} format are not "
                        "supported (only joint actions and XYZ_ROT6D end-effector poses)"
                    )
    max_num_embodiments = config.get("max_num_embodiments", 32)
    for tag, index in embodiment_id_mapping.items():
        if not 0 <= index < max_num_embodiments:
            problems.append(f"embodiment {tag!r} maps to projector {index}, outside [0, {max_num_embodiments})")
    if problems:
        raise ValueError("This checkpoint's processor can't be converted faithfully:\n- " + "\n- ".join(problems))


def convert_embodiment_id_mapping(original_mapping: dict | None) -> dict:
    """Merges the original tag table into `embodiment_id.json` for the tags it lacks, as the original processor does."""
    mapping = dict(original_mapping or {})
    for tag, index in ORIGINAL_EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
        mapping.setdefault(tag, index)
    return mapping


def convert_config(original: dict) -> Gr00tConfig:
    validate_original_config(original)
    vlm_config = AutoConfig.from_pretrained(original["model_name"])
    if vlm_config.text_config.hidden_size != original.get("backbone_embedding_dim", 2048):
        raise ValueError(
            f"backbone_embedding_dim {original.get('backbone_embedding_dim')} doesn't match the hidden size of "
            f"{original['model_name']} ({vlm_config.text_config.hidden_size})."
        )
    # The checkpoint only stores the decoder layers the action head reads from
    vlm_config.text_config.num_hidden_layers = original["select_layer"]
    # The action head reads the last decoder layer before the final norm
    vlm_config.text_config.tie_last_hidden_states = False
    diffusion = original["diffusion_model_cfg"]
    refiner = original["vl_self_attention_cfg"]
    dit_hidden_size = diffusion["num_attention_heads"] * diffusion["attention_head_dim"]
    refiner_hidden_size = refiner["num_attention_heads"] * refiner["attention_head_dim"]
    # Even blocks cross-attend, to the text tokens every `attend_text_every_n_blocks`-th time and to the image tokens
    # otherwise, odd blocks self-attend
    attend_text_every_n_blocks = original.get("attend_text_every_n_blocks", 2)
    layer_types = [
        "full_attention"
        if layer_idx % 2
        else ("text_cross_attention" if layer_idx % (2 * attend_text_every_n_blocks) == 0 else "image_cross_attention")
        for layer_idx in range(diffusion["num_layers"])
    ]
    return Gr00tConfig(
        vlm_config=vlm_config.to_dict(),
        action_horizon=original["action_horizon"],
        max_state_dim=original["max_state_dim"],
        max_action_dim=original["max_action_dim"],
        max_num_embodiments=original["max_num_embodiments"],
        projector_hidden_size=original["hidden_size"],
        refiner_config={
            "hidden_size": refiner_hidden_size,
            "intermediate_size": refiner.get("ff_inner_dim") or 4 * refiner_hidden_size,
            "num_hidden_layers": refiner["num_layers"],
            "num_attention_heads": refiner["num_attention_heads"],
            "dropout": refiner.get("dropout", 0.1),
        },
        action_head_config={
            "hidden_size": dit_hidden_size,
            "intermediate_size": diffusion.get("ff_inner_dim") or 4 * dit_hidden_size,
            "num_hidden_layers": diffusion["num_layers"],
            "num_attention_heads": diffusion["num_attention_heads"],
            "layer_types": layer_types,
            "backbone_hidden_size": vlm_config.text_config.hidden_size,
            "output_size": original["hidden_size"],
            "dropout": diffusion["dropout"],
        },
        num_timestep_buckets=original["num_timestep_buckets"],
        max_position_embeddings=original["max_seq_len"],
        num_inference_steps=original["num_inference_timesteps"],
        time_sampling_beta_alpha=original["noise_beta_alpha"],
        time_sampling_beta_beta=original["noise_beta_beta"],
        time_sampling_scale=original["noise_s"],
        state_dropout_prob=original["state_dropout_prob"],
    )


def load_original_processor_files(processor_dir: str, config: dict) -> tuple[dict, dict, dict]:
    """Reads and validates the original processor kwargs, statistics and embodiment mapping."""
    kwargs = load_json(processor_dir, "processor_config.json")["processor_kwargs"]
    embodiment_id_path = os.path.join(processor_dir, "embodiment_id.json")
    original_mapping = load_json(processor_dir, "embodiment_id.json") if os.path.exists(embodiment_id_path) else None
    embodiment_id_mapping = convert_embodiment_id_mapping(original_mapping)
    validate_original_processor(kwargs, embodiment_id_mapping, config)
    return kwargs, load_json(processor_dir, "statistics.json"), embodiment_id_mapping


def convert_processor(processor_dir: str, config: dict) -> Gr00tProcessor:
    kwargs, statistics, embodiment_id_mapping = load_original_processor_files(processor_dir, config)
    backbone_processor = AutoProcessor.from_pretrained(config["model_name"])
    backbone_image_processor = backbone_processor.image_processor
    # As in the original albumentations pipeline: fall back to the target and crop sizes when these aren't stored
    image_resolution = kwargs.get("shortest_image_edge") or kwargs["image_target_size"][0]
    crop_fraction = kwargs.get("crop_fraction") or kwargs["image_crop_size"][0] / kwargs["image_target_size"][0]
    image_processor = Gr00tImageProcessor(
        image_resolution=image_resolution,
        crop_fraction=crop_fraction,
        size=dict(backbone_image_processor.size),
        patch_size=backbone_image_processor.patch_size,
        temporal_patch_size=backbone_image_processor.temporal_patch_size,
        merge_size=backbone_image_processor.merge_size,
        image_mean=backbone_image_processor.image_mean,
        image_std=backbone_image_processor.image_std,
    )
    return Gr00tProcessor(
        image_processor=image_processor,
        tokenizer=backbone_processor.tokenizer,
        chat_template=backbone_processor.chat_template,
        modality_configs=kwargs["modality_configs"],
        statistics=statistics,
        embodiment_id_mapping=embodiment_id_mapping,
        max_state_dim=kwargs["max_state_dim"],
        max_action_dim=kwargs["max_action_dim"],
        max_action_horizon=kwargs["max_action_horizon"],
        formalize_language=processor_kwarg(kwargs, "formalize_language"),
        use_percentiles=processor_kwarg(kwargs, "use_percentiles"),
        clip_outliers=processor_kwarg(kwargs, "clip_outliers"),
        use_relative_action=processor_kwarg(kwargs, "use_relative_action"),
    )


def find_processor_dir(model_dir: str) -> str:
    """The processor files sit next to the model config, or in a `processor/` subfolder."""
    for directory in (model_dir, os.path.join(model_dir, "processor")):
        if os.path.exists(os.path.join(directory, "processor_config.json")):
            return directory
    raise FileNotFoundError(f"No processor_config.json in {model_dir} or {os.path.join(model_dir, 'processor')}.")


def download_original_files(checkpoint: str, subfolder: str | None, revision: str | None, weights: bool) -> str:
    """Downloads the files of the checkpoint in `subfolder` only, and returns the local checkpoint directory."""
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    wanted = {f"{prefix}config.json"} | {f"{prefix}{d}{name}" for d in ("", "processor/") for name in PROCESSOR_FILES}
    files = []
    for file in list_repo_files(checkpoint, revision=revision):
        is_weight = file.startswith(prefix) and "/" not in file[len(prefix) :] and ".safetensors" in file
        if file in wanted or (weights and is_weight):
            files.append(file)
    if f"{prefix}config.json" not in files:
        raise FileNotFoundError(f"No {prefix}config.json in {checkpoint}: pass the `--subfolder` of a checkpoint.")
    if weights and not any(".safetensors" in file for file in files):
        raise FileNotFoundError(
            f"No safetensors weights in {checkpoint}/{prefix}: pass the `--subfolder` of a checkpoint."
        )
    local_dir = snapshot_download(checkpoint, revision=revision, allow_patterns=files)
    return os.path.join(local_dir, subfolder) if subfolder else local_dir


def convert_gr00t_checkpoint(
    checkpoint: str,
    output_dir: str,
    subfolder: str | None = None,
    revision: str | None = None,
    push_to_hub: str | None = None,
):
    is_local = os.path.isdir(checkpoint)
    if is_local:
        model_dir = os.path.join(checkpoint, subfolder) if subfolder else checkpoint
    else:
        # Validate the small config files before downloading the weights
        model_dir = download_original_files(checkpoint, subfolder, revision, weights=False)
    original_config = load_json(model_dir, "config.json")
    processor_dir = find_processor_dir(model_dir)
    # Validate everything before loading the gated backbone config and tokenizer
    validate_original_config(original_config)
    load_original_processor_files(processor_dir, original_config)

    config = convert_config(original_config)
    processor = convert_processor(processor_dir, original_config)
    if not is_local:
        model_dir = download_original_files(checkpoint, subfolder, revision, weights=True)
    model, loading_info = Gr00tForConditionalGeneration.from_pretrained(
        model_dir, config=config, dtype=torch.bfloat16, output_loading_info=True
    )
    if loading_info["missing_keys"] or loading_info["unexpected_keys"] or loading_info["mismatched_keys"]:
        raise ValueError(f"The checkpoint weights don't match the converted architecture: {loading_info}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    if push_to_hub:
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B", help="Original checkpoint repo or directory.")
    parser.add_argument(
        "--subfolder", default=None, help="Subfolder holding the checkpoint, e.g. `libero_10` for GR00T-N1.7-LIBERO."
    )
    parser.add_argument("--revision", default=None, help="Revision of the original checkpoint repo.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model and processor.")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push the converted files to.")
    args = parser.parse_args()
    convert_gr00t_checkpoint(args.checkpoint, args.output_dir, args.subfolder, args.revision, args.push_to_hub)


if __name__ == "__main__":
    main()
