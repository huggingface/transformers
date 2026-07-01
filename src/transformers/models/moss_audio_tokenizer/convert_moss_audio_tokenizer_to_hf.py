# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Convert original MOSS Audio Tokenizer checkpoints to the Transformers format."""

import argparse
import gc
import json
import os
from typing import Any

import torch

from transformers import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderConfig,
    MossAudioTokenizerEncoderConfig,
    MossAudioTokenizerFeatureExtractor,
    MossAudioTokenizerModel,
    MossAudioTokenizerQuantizerConfig,
)
from transformers.utils import logging
from transformers.utils.hub import cached_file


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def _transformer_config_from_module_kwargs(module_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_dimension": module_kwargs["input_dimension"],
        "output_dimension": module_kwargs["output_dimension"],
        "d_model": module_kwargs["d_model"],
        "num_heads": module_kwargs["num_heads"],
        "num_layers": module_kwargs["num_layers"],
        "dim_feedforward": module_kwargs["dim_feedforward"],
        "causal": module_kwargs.get("causal", True),
        "norm": module_kwargs.get("norm", "layer_norm"),
        "positional_embedding": module_kwargs.get("positional_embedding", "rope"),
        "max_period": module_kwargs.get("max_period", 10000),
        "hidden_act": module_kwargs.get("hidden_act", "gelu"),
        "gating": module_kwargs.get("gating", "none"),
        "layer_scale": module_kwargs.get("layer_scale", 0.01),
        "conv_layout": module_kwargs.get("conv_layout", True),
    }


def _backbone_config_from_legacy_kwargs(module_kwargs: list[dict[str, Any]], config_class):
    patch_sizes = []
    transformers = []
    transformer_first = False

    for index, module_config in enumerate(module_kwargs):
        module_config = dict(module_config)
        module_type = module_config.pop("module_type")
        if index == 0:
            transformer_first = module_type == "Transformer"

        if module_type == "PatchedPretransform":
            patch_sizes.append(module_config["patch_size"])
        elif module_type == "Transformer":
            transformers.append(_transformer_config_from_module_kwargs(module_config))
        else:
            raise ValueError(f"Unsupported MOSS Audio Tokenizer module type: {module_type}")

    return config_class(
        patch_sizes=patch_sizes,
        transformer_first=transformer_first,
        input_dimensions=[config["input_dimension"] for config in transformers],
        output_dimensions=[config["output_dimension"] for config in transformers],
        d_models=[config["d_model"] for config in transformers],
        num_heads=[config["num_heads"] for config in transformers],
        num_layers=[config["num_layers"] for config in transformers],
        dim_feedforward=[config["dim_feedforward"] for config in transformers],
        causal=[config["causal"] for config in transformers],
        norm=[config["norm"] for config in transformers],
        positional_embedding=[config["positional_embedding"] for config in transformers],
        max_period=[config["max_period"] for config in transformers],
        hidden_act=[config["hidden_act"] for config in transformers],
        gating=[config["gating"] for config in transformers],
        layer_scale=[config["layer_scale"] for config in transformers],
        conv_layout=[config["conv_layout"] for config in transformers],
    )


def convert_config(config_dict: dict[str, Any]) -> MossAudioTokenizerConfig:
    if "encoder_config" in config_dict:
        encoder_config = config_dict["encoder_config"]
    else:
        encoder_config = _backbone_config_from_legacy_kwargs(
            config_dict["encoder_kwargs"], MossAudioTokenizerEncoderConfig
        )

    if "decoder_config" in config_dict:
        decoder_config = config_dict["decoder_config"]
    else:
        decoder_config = _backbone_config_from_legacy_kwargs(
            config_dict["decoder_kwargs"], MossAudioTokenizerDecoderConfig
        )

    if "quantizer_config" in config_dict:
        quantizer_config = dict(config_dict["quantizer_config"])
    else:
        quantizer_config = dict(config_dict["quantizer_kwargs"])
    quantizer_config.pop("quantizer_type", None)

    return MossAudioTokenizerConfig(
        sampling_rate=config_dict.get("sampling_rate", config_dict.get("sample_rate", 24000)),
        downsample_rate=config_dict.get("downsample_rate", 1920),
        causal_transformer_context_duration=config_dict.get("causal_transformer_context_duration", 10.0),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        quantizer_config=MossAudioTokenizerQuantizerConfig(**quantizer_config),
    )


def _load_config(input_path_or_repo: str, revision: str | None = None) -> dict[str, Any]:
    config_path = (
        input_path_or_repo
        if os.path.isfile(input_path_or_repo)
        else cached_file(input_path_or_repo, "config.json", revision=revision)
    )
    with open(config_path) as f:
        return json.load(f)


def _cached_or_local_file(input_path_or_repo: str, filename: str, revision: str | None = None) -> str:
    if os.path.isdir(input_path_or_repo):
        path = os.path.join(input_path_or_repo, filename)
        if os.path.isfile(path):
            return path
        raise OSError(f"Could not find {filename} in {input_path_or_repo}.")
    return cached_file(input_path_or_repo, filename, revision=revision)


def _load_state_dict(input_path_or_repo: str, revision: str | None = None) -> dict[str, torch.Tensor]:
    if os.path.isfile(input_path_or_repo):
        checkpoint_path = input_path_or_repo
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path)
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            checkpoint_path = _cached_or_local_file(input_path_or_repo, filename, revision=revision)
        except OSError:
            continue

        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path)
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    for index_filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = _cached_or_local_file(input_path_or_repo, index_filename, revision=revision)
        except OSError:
            continue

        with open(index_path) as f:
            index = json.load(f)

        state_dict = {}
        shard_filenames = sorted(set(index["weight_map"].values()))
        for shard_filename in shard_filenames:
            shard_path = _cached_or_local_file(input_path_or_repo, shard_filename, revision=revision)
            if shard_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                shard_state_dict = load_file(shard_path)
            else:
                shard_state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
            state_dict.update(shard_state_dict)

        return state_dict

    raise OSError(
        f"Could not find model.safetensors, pytorch_model.bin, or sharded checkpoint index in {input_path_or_repo}."
    )


def _rewrite_weight_norm_key(key: str) -> str:
    if key.endswith(".weight_g"):
        return key[: -len(".weight_g")] + ".parametrizations.weight.original0"
    if key.endswith(".weight_v"):
        return key[: -len(".weight_v")] + ".parametrizations.weight.original1"
    return key


def _convert_interleaved_projection_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    embed_dim = weight.shape[0] // 3
    head_dim = embed_dim // num_heads
    projected = weight.view(3, num_heads, head_dim, *weight.shape[1:])
    for index in (0, 1):
        projected[index] = torch.cat([projected[index, :, ::2], projected[index, :, 1::2]], dim=1)
    return projected.reshape_as(weight)


def _split_legacy_projection(
    state_dict: dict[str, torch.Tensor],
    source_key: str,
    target_pattern: str,
    mult: int,
):
    if source_key not in state_dict:
        return

    weight = state_dict.pop(source_key)
    weight = weight.view(mult, -1, *weight.shape[1:])
    for index in range(mult):
        state_dict[target_pattern.format(index=index)] = weight[index]


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], model: MossAudioTokenizerModel
) -> dict[str, torch.Tensor]:
    state_dict = {}
    for key, value in original_state_dict.items():
        key = _rewrite_weight_norm_key(key)
        if key.endswith(".alpha"):
            key = key[: -len(".alpha")] + ".weight"
            value = value.reshape(-1)
        state_dict[key] = value

    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "MossAudioTokenizerAttention":
            continue

        prefix = f"{module_name}." if module_name else ""
        _split_legacy_projection(
            state_dict,
            prefix + "in_proj_weight",
            prefix + "in_projs.{index}.weight",
            module.mult,
        )
        _split_legacy_projection(
            state_dict,
            prefix + "in_proj.weight",
            prefix + "in_projs.{index}.weight",
            module.mult,
        )
        _split_legacy_projection(
            state_dict,
            prefix + "out_proj.weight",
            prefix + "out_projs.{index}.weight",
            module.mult,
        )

        for index in range(module.mult):
            key = prefix + f"in_projs.{index}.weight"
            if key in state_dict:
                state_dict[key] = _convert_interleaved_projection_rows(state_dict[key], module.num_heads)

    return state_dict


@torch.no_grad()
def convert_model(input_path_or_repo: str, revision: str | None = None) -> MossAudioTokenizerModel:
    config_dict = _load_config(input_path_or_repo, revision=revision)
    config = convert_config(config_dict)

    logger.info("Loading original checkpoint")
    original_state_dict = _load_state_dict(input_path_or_repo, revision=revision)

    logger.info("Creating Transformers model")
    with torch.device("meta"):
        model = MossAudioTokenizerModel(config)
    model.apply_weight_norm()

    logger.info("Converting state dict")
    state_dict = convert_state_dict(original_state_dict, model)
    del original_state_dict
    gc.collect()

    logger.info("Loading converted state dict")
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.remove_weight_norm()
    del model.config._name_or_path
    return model


def create_feature_extractor(config: MossAudioTokenizerConfig) -> MossAudioTokenizerFeatureExtractor:
    return MossAudioTokenizerFeatureExtractor(
        feature_size=1,
        sampling_rate=config.sampling_rate,
        padding_value=0.0,
        hop_length=config.downsample_rate,
        padding_side="right",
        return_attention_mask=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert MOSS Audio Tokenizer weights to Hugging Face format")
    parser.add_argument("--input_path_or_repo", type=str, default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--input_revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub_path", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None and args.push_to_hub_path is None:
        raise ValueError("Either --output_dir or --push_to_hub_path must be provided.")

    model = convert_model(args.input_path_or_repo, revision=args.input_revision)
    feature_extractor = create_feature_extractor(model.config)

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        feature_extractor.save_pretrained(args.output_dir)
        logger.info(f"Model and feature extractor saved to {args.output_dir}")

    if args.push_to_hub_path is not None:
        model.push_to_hub(args.push_to_hub_path)
        feature_extractor.push_to_hub(args.push_to_hub_path)
        logger.info(f"Model and feature extractor pushed to {args.push_to_hub_path}")


if __name__ == "__main__":
    main()
