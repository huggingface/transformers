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

The weights of the original checkpoints also load directly with `Gr00tForConditionalGeneration.from_pretrained`; this
script is needed for the config and the processor, whose original files use the Isaac-GR00T format.

Usage:

```bash
python src/transformers/models/gr00t/convert_gr00t_to_hf.py \
    --checkpoint nvidia/GR00T-N1.7-3B --output_dir gr00t-n1.7-3b-hf
```

The tokenizer is loaded from the backbone repository named in the checkpoint config (`nvidia/Cosmos-Reason2-2B`), which
is gated: accept its terms on the Hub and log in first.
"""

import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download

from transformers import (
    AutoConfig,
    AutoProcessor,
    Gr00tConfig,
    Gr00tForConditionalGeneration,
    Gr00tImageProcessor,
    Gr00tProcessor,
)


def load_json(directory: str, name: str) -> dict:
    with open(os.path.join(directory, name), encoding="utf-8") as f:
        return json.load(f)


def convert_config(original: dict) -> Gr00tConfig:
    vlm_config = AutoConfig.from_pretrained(original["model_name"])
    # The checkpoint only stores the decoder layers the action head reads from
    vlm_config.text_config.num_hidden_layers = original["select_layer"]
    diffusion = original["diffusion_model_cfg"]
    refiner = original["vl_self_attention_cfg"]
    return Gr00tConfig(
        vlm_config=vlm_config.to_dict(),
        action_horizon=original["action_horizon"],
        max_state_dim=original["max_state_dim"],
        max_action_dim=original["max_action_dim"],
        max_num_embodiments=original["max_num_embodiments"],
        projector_hidden_size=original["hidden_size"],
        action_head_config={
            "hidden_size": diffusion["num_attention_heads"] * diffusion["attention_head_dim"],
            "num_hidden_layers": diffusion["num_layers"],
            "num_attention_heads": diffusion["num_attention_heads"],
            "attend_text_every_n_blocks": original.get("attend_text_every_n_blocks", 2),
            "refiner_num_hidden_layers": refiner["num_layers"],
            "refiner_num_attention_heads": refiner["num_attention_heads"],
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


def convert_processor(original_dir: str, model_name: str) -> Gr00tProcessor:
    kwargs = load_json(original_dir, "processor_config.json")["processor_kwargs"]
    if kwargs.get("apply_sincos_state_encoding") or kwargs.get("use_mean_std"):
        raise NotImplementedError("Sin/cos state encoding and global mean/std normalization are not supported.")
    backbone_processor = AutoProcessor.from_pretrained(model_name)
    backbone_image_processor = backbone_processor.image_processor
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
    embodiment_id_path = os.path.join(original_dir, "embodiment_id.json")
    return Gr00tProcessor(
        image_processor=image_processor,
        tokenizer=backbone_processor.tokenizer,
        chat_template=backbone_processor.chat_template,
        modality_configs=kwargs["modality_configs"],
        statistics=load_json(original_dir, "statistics.json"),
        embodiment_id_mapping=load_json(original_dir, "embodiment_id.json")
        if os.path.exists(embodiment_id_path)
        else {},
        max_state_dim=kwargs["max_state_dim"],
        max_action_horizon=kwargs["max_action_horizon"],
        formalize_language=kwargs.get("formalize_language", True),
        use_percentiles=kwargs.get("use_percentiles", False),
        clip_outliers=kwargs.get("clip_outliers", True),
        use_relative_action=kwargs.get("use_relative_action", False),
    )


def convert_gr00t_checkpoint(checkpoint: str, output_dir: str, push_to_hub: str | None = None):
    original_dir = checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint)
    original_config = load_json(original_dir, "config.json")

    config = convert_config(original_config)
    processor = convert_processor(original_dir, original_config["model_name"])
    model = Gr00tForConditionalGeneration.from_pretrained(original_dir, config=config, dtype=torch.bfloat16)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    if push_to_hub:
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B", help="Original checkpoint repo or directory.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model and processor.")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push the converted files to.")
    args = parser.parse_args()
    convert_gr00t_checkpoint(args.checkpoint, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
