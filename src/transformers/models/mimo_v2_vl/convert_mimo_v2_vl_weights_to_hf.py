# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
Convert an original MiMo V2.x checkpoint (`model_type="mimo_v2"`) into a native Transformers checkpoint.

The original checkpoints (`XiaomiMiMo/MiMo-V2.5`, `XiaomiMiMo/MiMo-V2.6-Flash-RL`, `XiaomiMiMo/MiMo-V2.6-Pro-RL` and the
text-only `XiaomiMiMo/MiMo-V2.5-Pro`) all declare `model_type="mimo_v2"` and `architectures=["MiMoV2ForCausalLM"]`,
and store:
    - a flat `config.json`, with the vision/audio settings and the preprocessing settings (`processor_config`)
      nested inside it,
    - TP-interleaved fused `qkv_proj` weights, FP8 (block-scaled) dense weights and FP8 or MXFP4 expert weights,
    - for multimodal checkpoints, the learned audio tokenizer in the `audio_tokenizer` directory.

This script dequantizes every weight, splits the fused QKV projections and renames the tensors to the native layout.
It writes the configuration with a distinct `model_type`/`architectures` per family, together with the processor:
    - multimodal checkpoints: `model_type="mimo_v2_vl"`, [`MiMoV2VLForConditionalGeneration`], [`MiMoV2VLProcessor`],
    - text-only checkpoints: `model_type="mimo_v2_flash"`, [`MiMoV2FlashForCausalLM`], tokenizer.

The conversion is first planned from the safetensors headers and checked against the names and shapes of the model,
before any weight is read. `--dry_run` stops there, which validates a conversion with the configuration, tokenizer
files and headers only (e.g. a copy of a Hub repository without its weights).

Sample usage:
    python src/transformers/models/mimo_v2_vl/convert_mimo_v2_vl_weights_to_hf.py \
        --input_dir /path/to/MiMo-V2.6-Flash-RL --output_dir /path/to/MiMo-V2.6-Flash-RL-hf --device cuda

The converted directory can then be uploaded as a native repository, e.g. with
`hf upload-large-folder <repo_id> /path/to/MiMo-V2.6-Flash-RL-hf --repo-type model`.
"""

import argparse
import json
import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import (
    GenerationConfig,
    MiMoV2FlashConfig,
    MiMoV2FlashForCausalLM,
    MiMoV2VLConfig,
    MiMoV2VLFeatureExtractor,
    MiMoV2VLForConditionalGeneration,
    MiMoV2VLImageProcessor,
    MiMoV2VLProcessor,
    MiMoV2VLVideoProcessor,
    TokenizersBackend,
)
from transformers.models.mimo_v2_flash.convert_mimo_v2_flash_weights_to_hf import convert_config_kwargs


# --------------------------------------------------------------------------------------------------------------------
# Mappings from the original names to the native ones
# --------------------------------------------------------------------------------------------------------------------

# Original text settings that have no native equivalent (quantization is removed by this conversion).
TEXT_KEYS_TO_DROP = {
    "architectures",
    "attention_projection_layout",
    "hybrid_block_size",
    "model_type",
    "num_nextn_predict_layers",
    "quantization_config",
    "transformers_version",
    "vision_model_type",
}
MULTIMODAL_KEYS = {
    "audio_config",
    "vision_config",
    "processor_config",
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "audio_start_token_id",
    "audio_end_token_id",
}

# Original `vision_config` entries -> `MiMoV2VLVisionConfig` arguments (`fullatt_block_indexes` and
# `vit_window_attn_types` are converted to `layer_types` and `patch_orders`).
VISION_CONFIG_MAPPING = {
    "depth": "depth",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "hidden_act": "hidden_act",
    "num_heads": "num_heads",
    "num_key_value_heads": "num_key_value_heads",
    "qk_channels": "head_dim",
    "in_chans": "in_channels",
    "patch_size": "patch_size",
    "temporal_patch_size": "temporal_patch_size",
    "spatial_merge_size": "spatial_merge_size",
    "out_hidden_size": "out_hidden_size",
    "visual_token_window_size": "sliding_window",
    "use_sink": "use_attention_sinks",
}
# Original `audio_config` entries -> `MiMoV2VLAudioConfig` arguments (`rope_theta`, `input_full_attention`,
# `speech_vocab_size` and `speech_zeroemb_idx` are converted separately).
AUDIO_CONFIG_MAPPING = {
    "audio_channels": "num_codebooks",
    "group_size": "group_size",
    "input_local_dim": "hidden_size",
    "input_local_layers": "num_hidden_layers",
    "input_local_attn_heads": "num_attention_heads",
    "input_local_head_dim": "head_dim",
    "input_local_intermediate_size": "intermediate_size",
    "input_local_hidden_dropout": "attention_dropout",
    "add_post_norm": "use_final_norm",
    "out_hidden_size": "out_hidden_size",
    "projection_layers": "projection_layers",
}
# Original `audio_tokenizer/config.json` entries -> `MiMoV2VLAudioTokenizerConfig` arguments (the attention window,
# layer pattern, skip connection, codebook sizes and `rope_theta` are converted separately).
AUDIO_TOKENIZER_CONFIG_MAPPING = {
    "d_model": "hidden_size",
    "n_mels": "num_mel_bins",
    "kernel_size": "conv_kernel_size",
    "stride_size": "conv_stride",
    "encoder_layers": "num_hidden_layers",
    "encoder_attention_heads": "num_attention_heads",
    "encoder_ffn_dim": "intermediate_size",
    "activation_function": "hidden_act",
    "encoder_causal": "is_causal",
    "avg_pooler": "downsample_factor",
}
# Original `processor_config` entries -> `MiMoV2VLFeatureExtractor` arguments.
FEATURE_EXTRACTOR_CONFIG_MAPPING = {
    "audio_sampling_rate": "sampling_rate",
    "audio_nfft": "n_fft",
    "audio_hop_length": "hop_length",
    "audio_window_size": "win_length",
    "audio_fmin": "f_min",
    "audio_fmax": "f_max",
    "audio_n_mels": "feature_size",
    "audio_channels": "num_codebooks",
    "audio_group_size": "group_size",
    "audio_segment_size": "segment_length",
    "audio_kernel_size": "conv_kernel_size",
    "audio_stride_size": "conv_stride",
    "audio_avg_pooler": "downsample_factor",
    "audio_input_id_per_second": "codec_frame_rate",
}
PROCESSOR_TOKEN_NAMES = (
    "image",
    "video",
    "audio",
    "vision_start",
    "vision_end",
    "audio_start",
    "audio_end",
    "video_start",
    "video_end",
)

# Original tensor names -> native tensor names. The first matching pattern is applied.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^model\.": "model.language_model.",
    r"^visual\.merger\.ln_q\.": "model.visual.merger.norm.",
    r"^visual\.merger\.mlp\.0\.": "model.visual.merger.linear_fc1.",
    r"^visual\.merger\.mlp\.2\.": "model.visual.merger.linear_fc2.",
    r"^visual\.": "model.visual.",
    r"^speech_embeddings\.": "model.audio_encoder.embed_tokens.",
    r"^audio_encoder\.input_local_transformer\.": "model.audio_encoder.",
    r"^audio_encoder\.projection\.mlp\.0\.": "model.audio_encoder.projection.linear_fc1.",
    r"^audio_encoder\.projection\.mlp\.2\.": "model.audio_encoder.projection.linear_fc2.",
    r"^audio_encoder\.": "model.audio_encoder.",
}
# Original audio tokenizer tensor names -> native tensor names. Only the encoder is used for understanding.
AUDIO_TOKENIZER_KEY_MAPPING = {
    r"^encoder\.down_sample_layer\.0\.": "model.audio_tokenizer.downsample_conv.",
    r"^encoder\.down_sample_norm\.": "model.audio_tokenizer.downsample_norm.",
    r"^encoder\.quantizer\.vq\.layers\.(\d+)\._codebook\.embed$": r"model.audio_tokenizer.quantizer.codebooks.\1.embed",
    r"^encoder\.": "model.audio_tokenizer.",
}


def convert_key(key: str, mapping: dict[str, str]) -> str:
    for pattern, replacement in mapping.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key, count=1)
    return key


def rename_config_keys(original_config: dict, mapping: dict[str, str]) -> dict:
    return {mapping[key]: value for key, value in original_config.items() if key in mapping}


def parse_per_codebook_values(value: str | int | list[int], num_codebooks: int) -> list[int]:
    """Parses values stored as a hyphen-separated string such as `"1280"` or `"1280-1024-..."`, one per codebook."""
    if isinstance(value, str):
        value = [int(item) for item in value.split("-")]
    elif isinstance(value, int):
        value = [value]
    return value * num_codebooks if len(value) == 1 else value


# --------------------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------------------


def is_multimodal(original_config: dict) -> bool:
    return bool(original_config.get("vision_config") or original_config.get("audio_config"))


def convert_text_config(original_config: dict) -> dict:
    text_config = {
        key: value
        for key, value in original_config.items()
        if key not in TEXT_KEYS_TO_DROP and key not in MULTIMODAL_KEYS
    }
    # Older checkpoints use the legacy `torch_dtype` name, sometimes next to a null `dtype` (or the opposite).
    torch_dtype = text_config.pop("torch_dtype", None)
    text_config["dtype"] = text_config.get("dtype") or torch_dtype
    return convert_config_kwargs(text_config)


def convert_vision_config(original_vision_config: dict) -> dict:
    vision_config = rename_config_keys(original_vision_config, VISION_CONFIG_MAPPING)
    # `vit_window_attn_types`: -1 for full attention, 0 (row-major) or 1 (column-major) for windowed attention.
    window_attn_types = original_vision_config.get("vit_window_attn_types")
    full_attention_blocks = set(original_vision_config.get("fullatt_block_indexes") or [])
    if window_attn_types is not None:
        vision_config["layer_types"] = [
            "full_attention" if i in full_attention_blocks or attn_type == -1 else "sliding_attention"
            for i, attn_type in enumerate(window_attn_types)
        ]
        vision_config["patch_orders"] = [
            "column_major" if attn_type == 1 else "row_major" for attn_type in window_attn_types
        ]
    return vision_config


def convert_audio_config(original_audio_config: dict) -> dict:
    audio_config = rename_config_keys(original_audio_config, AUDIO_CONFIG_MAPPING)
    if original_audio_config.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Only full rotary embeddings are supported in the audio encoder.")
    if "input_full_attention" in original_audio_config:
        audio_config["is_causal"] = not original_audio_config["input_full_attention"]
    if "rope_theta" in original_audio_config:
        audio_config["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": float(original_audio_config["rope_theta"]),
        }
    num_codebooks = audio_config.get("num_codebooks", 20)
    if "speech_vocab_size" in original_audio_config:
        audio_config["codebook_vocab_sizes"] = parse_per_codebook_values(
            original_audio_config["speech_vocab_size"], num_codebooks
        )
    if "speech_zeroemb_idx" in original_audio_config:
        audio_config["codebook_pad_token_ids"] = parse_per_codebook_values(
            original_audio_config["speech_zeroemb_idx"], num_codebooks
        )
    return audio_config


def convert_audio_tokenizer_config(original_audio_tokenizer_config: dict, segment_length: int | None) -> dict:
    original = original_audio_tokenizer_config
    audio_tokenizer_config = rename_config_keys(original, AUDIO_TOKENIZER_CONFIG_MAPPING)
    num_layers = original.get("encoder_layers", 24)

    # `encoder_attn_window_size` is `[left, right]`, -1 meaning unbounded. With `hybrid_attention`, the last layer of
    # every cycle of `swa_per_block` layers attends without a window.
    left, right = original.get("encoder_attn_window_size", [128, 0])
    is_causal = original.get("encoder_causal", True)
    if right not in (0, -1) and not is_causal:
        raise ValueError("Only causal or left-only attention windows are supported in the audio tokenizer.")
    if left < 0:
        layer_types = ["full_attention"] * num_layers
    elif original.get("hybrid_attention", False):
        swa_per_block = original.get("swa_per_block", 2)
        layer_types = [
            "sliding_attention" if i % swa_per_block < swa_per_block - 1 else "full_attention"
            for i in range(num_layers)
        ]
    else:
        layer_types = ["sliding_attention"] * num_layers
    audio_tokenizer_config["layer_types"] = layer_types
    if left >= 0:
        audio_tokenizer_config["sliding_window"] = left

    # `encoder_skip_layer_id` is one-based.
    skip_layer_id = original.get("encoder_skip_layer_id")
    audio_tokenizer_config["skip_layer_idx"] = skip_layer_id - 1 if skip_layer_id else None
    if "codebook_size" in original:
        num_quantizers = original.get("num_quantizers", len(original["codebook_size"]))
        codebook_sizes = list(original["codebook_size"])[:num_quantizers]
        audio_tokenizer_config["codebook_sizes"] = codebook_sizes + codebook_sizes[-1:] * (
            num_quantizers - len(codebook_sizes)
        )
    if "rope_theta" in original:
        audio_tokenizer_config["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": float(original["rope_theta"]),
        }
    if segment_length is not None:
        audio_tokenizer_config["segment_length"] = segment_length
    return audio_tokenizer_config


def load_original_config(input_dir: Path) -> dict:
    return json.loads((input_dir / "config.json").read_text(encoding="utf-8"))


def convert_config(input_dir: Path) -> MiMoV2VLConfig | MiMoV2FlashConfig:
    original_config = load_original_config(input_dir)
    text_config = convert_text_config(original_config)
    if not is_multimodal(original_config):
        return MiMoV2FlashConfig(**text_config)

    original_audio_tokenizer_config = json.loads(
        (input_dir / "audio_tokenizer" / "config.json").read_text(encoding="utf-8")
    )
    processor_config = original_config.get("processor_config") or {}
    token_ids = {
        name: original_config.get(name, processor_config.get(name))
        for name in MULTIMODAL_KEYS
        if name.endswith("_token_id") and original_config.get(name, processor_config.get(name)) is not None
    }
    return MiMoV2VLConfig(
        text_config=text_config,
        vision_config=convert_vision_config(original_config["vision_config"]),
        audio_config=convert_audio_config(original_config["audio_config"]),
        audio_tokenizer_config=convert_audio_tokenizer_config(
            original_audio_tokenizer_config, original_config["audio_config"].get("audio_segment_size")
        ),
        tie_word_embeddings=original_config.get("tie_word_embeddings", False),
        dtype=text_config.get("dtype"),
        **token_ids,
    )


def convert_generation_config(input_dir: Path) -> GenerationConfig | None:
    path = input_dir / "generation_config.json"
    if not path.is_file():
        return None
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    generation_config.pop("transformers_version", None)
    # Greedy configs of the original checkpoints still list sampling options, which cannot be saved.
    if not generation_config.get("do_sample", False):
        for name in ("temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"):
            generation_config.pop(name, None)
    return GenerationConfig(**generation_config)


# --------------------------------------------------------------------------------------------------------------------
# Processor
# --------------------------------------------------------------------------------------------------------------------


def convert_processor(input_dir: Path, config: MiMoV2VLConfig) -> MiMoV2VLProcessor:
    original_config = load_original_config(input_dir)
    processor_config = original_config.get("processor_config") or {}
    vision_config = config.vision_config
    audio_tokenizer_config = config.audio_tokenizer_config

    # The original tokenizer config declares a slow tokenizer class: load the `tokenizer.json` backend directly.
    tokenizer = TokenizersBackend.from_pretrained(input_dir)

    spatial_kwargs = {
        "patch_size": processor_config.get("patch_size", vision_config.patch_size),
        "merge_size": processor_config.get("merge_size", vision_config.spatial_merge_size),
        "temporal_patch_size": processor_config.get("temporal_patch_size", vision_config.temporal_patch_size),
    }
    factor = spatial_kwargs["patch_size"] * spatial_kwargs["merge_size"]

    def pixel_budget(modality):
        return {
            "shortest_edge": processor_config.get(f"{modality}_min_pixels", 4 * factor**2),
            "longest_edge": processor_config.get(f"{modality}_max_pixels", 4096 * factor**2),
        }

    image_processor = MiMoV2VLImageProcessor(**spatial_kwargs, size=pixel_budget("image"))
    video_kwargs = {
        "temporal_compression_ratio": processor_config.get("temporal_compression_ratio"),
        "total_max_pixels": processor_config.get("video_total_max_pixels"),
        "fps": processor_config.get("fps"),
        "num_frames": processor_config.get("num_frames"),
        "min_frames": processor_config.get("min_frames"),
        "max_frames": processor_config.get("max_frames"),
    }
    video_processor = MiMoV2VLVideoProcessor(
        **spatial_kwargs,
        size=pixel_budget("video"),
        **{key: value for key, value in video_kwargs.items() if value is not None},
    )

    # The audio token counts computed by the feature extractor follow the audio tokenizer and audio encoder.
    feature_extractor_kwargs = {
        "feature_size": audio_tokenizer_config.num_mel_bins,
        "conv_kernel_size": audio_tokenizer_config.conv_kernel_size,
        "conv_stride": audio_tokenizer_config.conv_stride,
        "downsample_factor": audio_tokenizer_config.downsample_factor,
        "segment_length": audio_tokenizer_config.segment_length,
        "num_codebooks": config.audio_config.num_codebooks,
        "group_size": config.audio_config.group_size,
    }
    feature_extractor_kwargs.update(rename_config_keys(processor_config, FEATURE_EXTRACTOR_CONFIG_MAPPING))
    feature_extractor = MiMoV2VLFeatureExtractor(**feature_extractor_kwargs)

    # Use the special tokens of the checkpoint's token IDs.
    token_kwargs = {}
    for name in PROCESSOR_TOKEN_NAMES:
        token_id = original_config.get(f"{name}_token_id", processor_config.get(f"{name}_token_id"))
        if token_id is not None:
            token_kwargs[f"{name}_token"] = tokenizer.convert_ids_to_tokens(token_id)

    processor_kwargs = {
        key: processor_config[key]
        for key in ("video_audio_interleave_length", "use_per_grid_t_timestamps")
        if key in processor_config
    }
    return MiMoV2VLProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        **token_kwargs,
        **processor_kwargs,
    )


# --------------------------------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------------------------------


def dequantize_mxfp4(weight: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Unpack two E2M1 values per byte (low nibble first) and apply the per-block E8M0 scales."""
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=weight.device, dtype=torch.float32
    )
    indices = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    if tuple(scales.shape) != (weight.shape[0], indices.shape[1] // block_size):
        raise ValueError(f"MXFP4 scales of shape {tuple(scales.shape)} do not match weights {tuple(weight.shape)}.")
    scale_values = torch.exp2(scales.float() - 127).repeat_interleave(block_size, dim=-1)
    return values[indices] * scale_values


def dequantize_fp8(weight: torch.Tensor, scales: torch.Tensor, block_size: tuple[int, int] = (128, 128)):
    """Multiply FP8 weights by their per-block inverse scales."""
    expanded = scales.float().repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], 1)
    if expanded.shape[0] < weight.shape[0] or expanded.shape[1] < weight.shape[1]:
        raise ValueError("FP8 scales do not cover the weight matrix.")
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]


def get_attention_shape(text_config, layer_idx: int) -> tuple[int, int, int, int]:
    """Returns the query heads, KV heads, query/key head dim and value head dim of a decoder layer."""
    if text_config.layer_types[layer_idx] == "sliding_attention":
        return (
            text_config.swa_num_attention_heads,
            text_config.swa_num_key_value_heads,
            text_config.swa_head_dim,
            text_config.swa_v_head_dim,
        )
    return (
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
        text_config.v_head_dim,
    )


def is_valid_tp_size(tp_size: int, num_heads: int, num_kv_heads: int) -> bool:
    # KV heads are either sharded evenly across ranks, or replicated when there are fewer KV heads than ranks.
    kv_ok = num_kv_heads % tp_size == 0 if num_kv_heads >= tp_size else tp_size % num_kv_heads == 0
    return num_heads % tp_size == 0 and kv_ok


def get_tp_size_candidates(text_config, layer_idx, num_rows, num_scale_rows=None, block_rows=128) -> set[int]:
    """
    Returns the TP sizes compatible with a fused QKV weight of `num_rows` rows. When the weight is FP8, the number of
    rows of its scales (padded to `block_rows` on every rank) also depends on the TP size and narrows the candidates.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim = get_attention_shape(text_config, layer_idx)
    candidates = set()
    for tp_size in range(1, num_heads + 1):
        if not is_valid_tp_size(tp_size, num_heads, num_kv_heads):
            continue
        rows_per_rank = num_heads // tp_size * head_dim + max(1, num_kv_heads // tp_size) * (head_dim + v_head_dim)
        if rows_per_rank * tp_size != num_rows:
            continue
        if num_scale_rows is not None and math.ceil(rows_per_rank / block_rows) * tp_size != num_scale_rows:
            continue
        candidates.add(tp_size)
    return candidates


def infer_tp_size(checkpoint, text_config, block_rows=128) -> int:
    """Infers the TP size used to save the fused QKV weights, which must be consistent across all layers."""
    candidates = None
    for name in checkpoint.names():
        if match := re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight", name):
            scale_name = name + "_scale_inv"
            num_scale_rows = checkpoint.shape(scale_name)[0] if scale_name in checkpoint else None
            layer_candidates = get_tp_size_candidates(
                text_config, int(match[1]), checkpoint.shape(name)[0], num_scale_rows, block_rows
            )
            candidates = layer_candidates if candidates is None else candidates & layer_candidates
    if candidates is None or len(candidates) != 1:
        raise ValueError(
            f"Cannot infer the TP size of the fused QKV weights (candidates: {sorted(candidates or [])}), please pass "
            "`--tp_size`."
        )
    return candidates.pop()


def split_interleaved_qkv(weight, text_config, layer_idx, tp_size, scales=None, block_size=(128, 128)):
    """
    Splits a `[Q_rank0, K_rank0, V_rank0, Q_rank1, ...]` fused weight into the full Q, K and V weights.
    FP8 scales are applied per rank, since every rank shard is padded to the scale block size independently.
    """
    num_heads, num_kv_heads, head_dim, v_head_dim = get_attention_shape(text_config, layer_idx)
    if not is_valid_tp_size(tp_size, num_heads, num_kv_heads):
        raise ValueError(f"TP size {tp_size} is incompatible with {num_heads} query and {num_kv_heads} KV heads.")
    kv_heads_per_rank = max(1, num_kv_heads // tp_size)
    q_size = num_heads // tp_size * head_dim
    k_size = kv_heads_per_rank * head_dim
    v_size = kv_heads_per_rank * v_head_dim
    shard_size = q_size + k_size + v_size
    if weight.shape[0] != shard_size * tp_size:
        raise ValueError(f"QKV weight of shape {tuple(weight.shape)} does not match a TP={tp_size} layout.")

    scale_shards = scales.chunk(tp_size, dim=0) if scales is not None else [None] * tp_size
    queries, keys, values = [], [], []
    for rank, (shard, shard_scales) in enumerate(zip(weight.split(shard_size), scale_shards)):
        if shard_scales is not None:
            shard = dequantize_fp8(shard, shard_scales, block_size)
        query, key, value = shard.split((q_size, k_size, v_size), dim=0)
        queries.append(query)
        # When there are fewer KV heads than ranks, consecutive ranks hold copies of the same KV head.
        if num_kv_heads >= tp_size or rank % (tp_size // num_kv_heads) == 0:
            keys.append(key)
            values.append(value)
    return torch.cat(queries), torch.cat(keys), torch.cat(values)


class ShardWriter:
    """Writes tensors to safetensors shards of at most `max_shard_size` bytes, followed by the weight index."""

    def __init__(self, output_dir: Path, max_shard_size: int):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size
        self.buffer, self.buffer_size = {}, 0
        self.shards = []
        self.total_size = 0

    def add(self, name: str, tensor: torch.Tensor):
        tensor = tensor.contiguous().cpu()
        size = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_size + size > self.max_shard_size:
            self.flush()
        self.buffer[name] = tensor
        self.buffer_size += size
        self.total_size += size

    def flush(self):
        if not self.buffer:
            return
        filename = f"model-{len(self.shards) + 1:05d}.safetensors"
        save_file(self.buffer, self.output_dir / filename, metadata={"format": "pt"})
        self.shards.append((filename, list(self.buffer)))
        print(f"Saved {filename} ({len(self.buffer)} tensors)", flush=True)
        self.buffer, self.buffer_size = {}, 0

    def finalize(self):
        self.flush()
        num_shards = len(self.shards)
        weight_map = {}
        for index, (filename, names) in enumerate(self.shards, start=1):
            final_name = f"model-{index:05d}-of-{num_shards:05d}.safetensors"
            (self.output_dir / filename).rename(self.output_dir / final_name)
            weight_map.update(dict.fromkeys(names, final_name))
        index = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        (self.output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


class SafetensorsCheckpoint:
    """
    Lazy access to a (sharded) safetensors checkpoint. Names, dtypes and shapes come from the shard headers only, so
    that a conversion can be planned and validated without reading (or downloading) the weights.
    """

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            self.directory, self.weight_map = path, index["weight_map"]
            self.metadata = index.get("metadata") or {}
        else:
            self.directory, self.metadata = path.parent, {}
            self.weight_map = dict.fromkeys(self.read_header(path), path.name)
        self.headers = {}
        for filename in sorted(set(self.weight_map.values())):
            self.headers.update(self.read_header(self.directory / filename))
        self.files = {}

    @staticmethod
    def read_header(path: Path) -> dict:
        with open(path, "rb") as reader:
            (header_size,) = struct.unpack("<Q", reader.read(8))
            header = json.loads(reader.read(header_size))
        header.pop("__metadata__", None)
        return header

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> list[str]:
        # Grouping by shard keeps the reads sequential.
        return sorted(self.weight_map, key=lambda name: (self.weight_map[name], name))

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.headers[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.headers[name]["dtype"]

    def get(self, name: str, device: str = "cpu") -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self.files:
            self.files[filename] = safe_open(self.directory / filename, framework="pt", device="cpu")
        return self.files[filename].get_tensor(name).to(device)


def load_tensor(checkpoint, name, device="cpu"):
    return [checkpoint.get(name, device)]


def load_fp8(checkpoint, name, scale_name, block_size, device="cpu"):
    return [dequantize_fp8(checkpoint.get(name, device), checkpoint.get(scale_name, device), block_size)]


def load_mxfp4(checkpoint, name, scale_name, block_size, device="cpu"):
    return [dequantize_mxfp4(checkpoint.get(name, device), checkpoint.get(scale_name, device), block_size)]


def load_qkv(checkpoint, name, scale_name, text_config, layer_idx, tp_size, block_size, device="cpu"):
    scales = checkpoint.get(scale_name, device) if scale_name is not None else None
    return list(
        split_interleaved_qkv(checkpoint.get(name, device), text_config, layer_idx, tp_size, scales, block_size)
    )


@dataclass
class ConversionStep:
    """Produces the tensors `names` (with shapes `shapes`) of the converted checkpoint from the original one."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    convert: Callable[[], list[torch.Tensor]]


def plan_conversion(
    checkpoint: SafetensorsCheckpoint,
    audio_tokenizer: SafetensorsCheckpoint | None,
    config,
    original_config: dict,
    tp_size: int | None = None,
    device: str = "cpu",
) -> list[ConversionStep]:
    """Lists the conversion of every tensor, using only the checkpoint headers."""
    multimodal = audio_tokenizer is not None
    text_config = config.get_text_config()
    quantization_config = original_config.get("quantization_config") or {}
    fp8_block_size = tuple(quantization_config.get("weight_block_size", (128, 128)))
    mxfp4_block_size = quantization_config.get("mxfp4_block_size", 32)
    tp_size = (
        tp_size or checkpoint.metadata.get("tp_size") or infer_tp_size(checkpoint, text_config, fp8_block_size[0])
    )
    print(f"Splitting the fused QKV weights with TP={tp_size}.", flush=True)
    qkv_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight")

    steps = []
    for name in checkpoint.names():
        # Scales are consumed with their weight. Multi-token prediction layers and rotary buffers are not used.
        if name.endswith((".weight_scale_inv", ".weight_scale", "rotary_emb.inv_freq")) or name.startswith(
            "model.mtp."
        ):
            continue
        shape = checkpoint.shape(name)
        target = convert_key(name, ORIGINAL_TO_CONVERTED_KEY_MAPPING) if multimodal else name
        stem = name.removesuffix(".weight")
        fp8_scale_name = stem + ".weight_scale_inv" if name.endswith(".weight") else None
        if fp8_scale_name not in checkpoint:
            fp8_scale_name = None

        if qkv_match := qkv_pattern.fullmatch(name):
            layer_idx = int(qkv_match[1])
            layer_tp_size = tp_size
            num_heads, num_kv_heads, head_dim, v_head_dim = get_attention_shape(text_config, layer_idx)
            # Each rank stores its query heads and its (possibly replicated) key/value heads.
            kv_heads_per_rank = max(1, num_kv_heads // layer_tp_size)
            expected_rows = num_heads * head_dim + layer_tp_size * kv_heads_per_rank * (head_dim + v_head_dim)
            if shape[0] != expected_rows:
                raise ValueError(f"{name} has {shape[0]} rows, expected {expected_rows} for TP={layer_tp_size}.")
            prefix = target.removesuffix("qkv_proj.weight")
            names = [f"{prefix}{projection}_proj.weight" for projection in "qkv"]
            shapes = [
                (num_heads * head_dim, shape[1]),
                (num_kv_heads * head_dim, shape[1]),
                (num_kv_heads * v_head_dim, shape[1]),
            ]
            convert = partial(
                load_qkv, checkpoint, name, fp8_scale_name, text_config, layer_idx, layer_tp_size, fp8_block_size
            )
        elif checkpoint.dtype(name) == "U8":
            # MXFP4 stores two 4-bit values per byte.
            names, shapes = [target], [(shape[0], 2 * shape[1])]
            convert = partial(load_mxfp4, checkpoint, name, stem + ".weight_scale", mxfp4_block_size)
        elif fp8_scale_name is not None:
            names, shapes = [target], [shape]
            convert = partial(load_fp8, checkpoint, name, fp8_scale_name, fp8_block_size)
        else:
            names, shapes = [target], [shape]
            convert = partial(load_tensor, checkpoint, name)
        steps.append(ConversionStep(names, shapes, partial(convert, device=device)))

    if multimodal:
        num_quantizers = len(config.audio_tokenizer_config.codebook_sizes)
        for name in audio_tokenizer.names():
            # Only the encoder and the codebooks of the quantizer stages used by the model are needed: the decoder
            # and the training statistics of the codebooks (`cluster_size`, `embed_avg`, `inited`) are dropped.
            if not name.startswith("encoder.") or name.endswith(("cluster_size", "embed_avg", "inited")):
                continue
            if (match := re.search(r"\.quantizer\.vq\.layers\.(\d+)\.", name)) and int(match[1]) >= num_quantizers:
                continue
            steps.append(
                ConversionStep(
                    [convert_key(name, AUDIO_TOKENIZER_KEY_MAPPING)],
                    [audio_tokenizer.shape(name)],
                    partial(load_tensor, audio_tokenizer, name),
                )
            )
    return steps


def get_expected_checkpoint_shapes(model_class, config) -> dict[str, tuple[int, ...]]:
    """
    Returns the names and shapes of a saved native checkpoint: the model state dict, with the fused experts and the
    attention sinks in the layout they are saved in (`conversion_mapping.py` converts them when loading).
    """
    with torch.device("meta"):
        model = model_class(config)
    shapes = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            num_experts, double_intermediate_size, hidden_size = tensor.shape
            prefix = name.removesuffix("gate_up_proj")
            for expert in range(num_experts):
                for projection in ("gate_proj", "up_proj"):
                    shapes[f"{prefix}{expert}.{projection}.weight"] = (double_intermediate_size // 2, hidden_size)
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert in range(tensor.shape[0]):
                shapes[f"{prefix}{expert}.down_proj.weight"] = tuple(tensor.shape[1:])
        elif name.endswith(".self_attn.sinks"):
            shapes[name.replace(".self_attn.sinks", ".self_attn.attention_sink_bias")] = tuple(tensor.shape)
        else:
            shapes[name] = tuple(tensor.shape)
    return shapes


def check_conversion_plan(steps: list[ConversionStep], model_class, config):
    """Checks that the planned tensors exactly match the names and shapes expected by the model."""
    planned = {name: shape for step in steps for name, shape in zip(step.names, step.shapes)}
    expected = get_expected_checkpoint_shapes(model_class, config)
    missing = sorted(set(expected) - set(planned))
    unexpected = sorted(set(planned) - set(expected))
    mismatched = sorted(
        f"{name}: {planned[name]} != {expected[name]}"
        for name in set(planned) & set(expected)
        if tuple(planned[name]) != tuple(expected[name])
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"The converted weights do not match {model_class.__name__}:\n"
            f"missing={missing[:20]}\nunexpected={unexpected[:20]}\nmismatched shapes={mismatched[:20]}"
        )
    print(f"Conversion plan matches {model_class.__name__}: {len(planned)} tensors.", flush=True)


def write_converted_weights(
    steps: list[ConversionStep], output_dir: Path, dtype=torch.bfloat16, max_shard_size=5 * 1024**3
):
    writer = ShardWriter(output_dir, max_shard_size)
    with torch.no_grad():
        for step in steps:
            for name, shape, tensor in zip(step.names, step.shapes, step.convert()):
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}.")
                # The audio tokenizer quantizer and the router correction bias are kept in float32.
                keep_float32 = ".quantizer." in name or name.endswith("e_score_correction_bias")
                if tensor.is_floating_point() and not keep_float32:
                    tensor = tensor.to(dtype)
                writer.add(name, tensor)
    writer.finalize()


def convert_checkpoint(
    input_dir,
    output_dir=None,
    tp_size=None,
    device="cpu",
    dtype=torch.bfloat16,
    max_shard_size=5 * 1024**3,
    dry_run=False,
):
    """
    Converts the checkpoint in `input_dir` to `output_dir`. With `dry_run=True`, only the shard headers are read: the
    configuration, processor and the names and shapes of all the converted weights are checked, and nothing is
    written. This allows validating a conversion of a Hub repository without downloading its weights.
    """
    input_dir = Path(input_dir)
    original_config = load_original_config(input_dir)
    config = convert_config(input_dir)
    multimodal = isinstance(config, MiMoV2VLConfig)
    model_class = MiMoV2VLForConditionalGeneration if multimodal else MiMoV2FlashForCausalLM
    config.architectures = [model_class.__name__]
    processor = convert_processor(input_dir, config) if multimodal else TokenizersBackend.from_pretrained(input_dir)
    generation_config = convert_generation_config(input_dir)

    checkpoint = SafetensorsCheckpoint(input_dir)
    audio_tokenizer = (
        SafetensorsCheckpoint(input_dir / "audio_tokenizer" / "model.safetensors") if multimodal else None
    )
    steps = plan_conversion(checkpoint, audio_tokenizer, config, original_config, tp_size, device)
    check_conversion_plan(steps, model_class, config)
    if dry_run:
        return config, processor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)
    write_converted_weights(steps, output_dir, dtype, max_shard_size)
    print(f"Saved the converted checkpoint to {output_dir}")
    return config, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Directory of the original checkpoint.")
    parser.add_argument("--output_dir", help="Directory to write the converted checkpoint to.")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="TP size of the fused QKV weights. Read from the weight index metadata, or inferred from the shapes.",
    )
    parser.add_argument("--device", default="cpu", help="Device used to dequantize the weights, e.g. `cuda`.")
    parser.add_argument("--dtype", default="bfloat16", help="Dtype of the converted weights.")
    parser.add_argument("--max_shard_size_gb", type=float, default=5.0, help="Maximum size of each output shard.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate the conversion from the configuration, tokenizer and safetensors headers.",
    )
    args = parser.parse_args()
    if args.output_dir is None and not args.dry_run:
        parser.error("--output_dir is required unless --dry_run is set.")
    convert_checkpoint(
        args.input_dir,
        args.output_dir,
        tp_size=args.tp_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        max_shard_size=int(args.max_shard_size_gb * 1024**3),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
