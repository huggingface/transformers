# Copyright 2026 The HuggingFace Team and NVIDIA CORPORATION. All rights reserved.
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

"""Convert AudioVisualFlamingo checkpoints into a Hugging Face repository layout.

Like the AudioFlamingo3 converter, this script:
1) reads source component configs to build an AudioVisualFlamingoConfig programmatically,
2) constructs processor and model objects with those configs,
3) lets the standard HF serialization APIs emit config and safetensors artifacts.

No JSON files are copied or manually edited.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import safe_open, save_model

from transformers import (
    AudioVisualFlamingoConfig,
    AudioVisualFlamingoForConditionalGeneration,
    AudioVisualFlamingoProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig,
    WhisperFeatureExtractor,
)
from transformers.initialization import no_init_weights


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SRC_PATH = Path("/fs/nexus-projects/JSALT_workshop/lasha/Dev/audiovisualflamingo")
DEFAULT_DST_PATH = Path("/fs/nexus-projects/JSALT_workshop/lasha/Dev/comni")
LEGACY_CHECKPOINT_KEY_MAPPING = {
    r"^vision_tower\.vision_tower\.vision_model\.": "vision_tower.vision_tower.",
    r"^sound_tower\.audio_tower\.": "sound_tower.",
}

# Maps legacy component sub-directories to the weight-key prefix expected by
# AudioVisualFlamingoForConditionalGeneration.
COMPONENT_TO_PREFIX = {
    "llm": "llm",
    "vision_tower": "vision_tower.vision_tower",
    "mm_projector": "mm_projector",
    "sound_tower": "sound_tower",
    "sound_mm_projector": "sound_mm_projector",
}

# Non-standard keys injected into the LLM (Qwen2) config by quantization or
# pruning toolchains.  These are never consumed by the HF Qwen2 model and
# bloat the serialised config (channel_order_list alone is ~60 KB).
LLM_CFG_KEYS_TO_STRIP = {
    "channel_order_list",
    "head_order_list",
    "head_dim_list",
    "head_dim_original",
    "hidden_size_list",
    "intermediate_size_list",
    "kv_repeat_original",
    "num_attention_heads_list",
    "num_key_value_heads_list",
    "model_max_length",
    "tokenizer_model_max_length",
    "tokenizer_padding_side",
    "_name_or_path",
    "transformers_version",
}

# Keys stripped from every component sub-config (vision tower, projectors, etc.).
COMPONENT_CFG_KEYS_TO_STRIP = {
    "_name_or_path",
    "transformers_version",
    "torch_dtype",
}

# Additional keys stripped from the sound tower config.  The source Qwen2AudioConfig
# embeds a redundant nested ``audio_config`` (duplicate of top-level fields) and a
# ``text_config`` for its unused text decoder.
SOUND_TOWER_EXTRA_KEYS_TO_STRIP = {
    "audio_config",
    "text_config",
    "vocab_size",
    "audio_token_index",
    "ignore_index",
}

# AudioVisualFlamingoConfig.__init__ explicit parameters that we extract from
# the source top-level config.json (excludes training-only params like *_lr).
AVF_CONFIG_FIELDS = {
    "mm_vision_select_layer",
    "mm_vision_select_feature",
    "dynamic_s2",
    "s2_scales",
    "s2_max_split_size",
    "s2_resize_output_to_scale_idx",
    "image_encoder",
    "video_encoder",
    "sound_encoder",
    "load_audio_in_video",
    "interleaved_vis_aud_in_video",
}

PROCESSOR_CONFIG_FIELDS = {
    "image_aspect_ratio",
    "num_video_frames",
    "max_tiles",
    "interleaved_video_segment_duration",
    "audio_sampling_rate",
    "audio_chunk_length",
    "audio_hop_length",
    "mm_use_bos_eos_tokens",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_s2_scales(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(",")
    return [int(value) for value in values]


def _normalize_encoder_config(config, default_target: str):
    if config is None:
        return {"_target_": default_target}
    if isinstance(config, str):
        config = json.loads(config)
    config = dict(config)
    target = config.get("_target_", default_target)
    if isinstance(target, str):
        config["_target_"] = target.rsplit(".", maxsplit=1)[-1]
    return config


# ---------------------------------------------------------------------------
# Weight collection
# ---------------------------------------------------------------------------


def _resolve_component_dir(dirpath: Path):
    if not dirpath.is_dir():
        return None
    idx = dirpath / "model.safetensors.index.json"
    mono = dirpath / "model.safetensors"
    if idx.exists():
        wm = _load_json(idx).get("weight_map") or {}
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in wm.items():
            by_shard[shard].append(key)
        return ("sharded", dirpath, {shard: sorted(keys) for shard, keys in sorted(by_shard.items())})
    if mono.exists():
        return ("file", mono)
    cands = sorted([x for x in dirpath.iterdir() if x.suffix == ".safetensors"])
    if len(cands) == 1:
        return ("file", cands[0])
    return None


def _collect_component_state(src_root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for component, out_prefix in COMPONENT_TO_PREFIX.items():
        comp = _resolve_component_dir(src_root / component)
        if not comp:
            logger.info("No weights found for optional component: %s", component)
            continue
        if comp[0] == "file":
            fp: Path = comp[1]
            with safe_open(str(fp), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key == "__metadata__":
                        continue
                    state[f"{out_prefix}.{key}"] = f.get_tensor(key)
        else:
            base: Path = comp[1]
            shard_map: dict[str, list[str]] = comp[2]
            for shard, keys in shard_map.items():
                sp = base / shard
                with safe_open(str(sp), framework="pt", device="cpu") as f:
                    for key in keys:
                        state[f"{out_prefix}.{key}"] = f.get_tensor(key)
        logger.info("Collected %s weights under prefix '%s'", component, out_prefix)
    return state


def _normalize_state_dict_keys(state: dict[str, Any]) -> dict[str, Any]:
    normalized_state = dict(state)
    for pattern, replacement in LEGACY_CHECKPOINT_KEY_MAPPING.items():
        renamed_keys = [key for key in normalized_state if re.match(pattern, key)]
        for key in renamed_keys:
            normalized_state[re.sub(pattern, replacement, key)] = normalized_state.pop(key)
    return normalized_state


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def collect_encoder_boundary_tokens(config: AudioVisualFlamingoConfig) -> list[str]:
    token_keys = {"start_tokens", "end_tokens", "sep_tokens"}
    collected: list[str] = []
    seen: set[str] = set()

    def _maybe_add(token):
        if not isinstance(token, str) or token == "None" or token in seen:
            return
        seen.add(token)
        collected.append(token)

    def _visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_keys:
                    _maybe_add(value)
                _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    _maybe_add("\n")
    for attr in ("image_encoder", "video_encoder", "sound_encoder"):
        encoder_cfg = getattr(config, attr, None)
        if isinstance(encoder_cfg, str):
            try:
                encoder_cfg = json.loads(encoder_cfg)
            except Exception:
                continue
        _visit(encoder_cfg)
    return collected


def _build_config(src_root: Path, tokenizer) -> AudioVisualFlamingoConfig:
    """Build an AudioVisualFlamingoConfig programmatically from the source checkpoint."""
    top_cfg = _load_json(src_root / "config.json")

    # Read and clean component sub-configs.
    def _read_component(name: str) -> dict[str, Any] | None:
        p = src_root / name / "config.json"
        return _load_json(p) if p.is_file() else None

    text_config = _read_component("llm")
    if text_config:
        text_config = {k: v for k, v in text_config.items() if k not in LLM_CFG_KEYS_TO_STRIP}

    def _clean_component(cfg, extra_strip=None):
        if cfg is None:
            return None
        cfg = {k: v for k, v in cfg.items() if k not in COMPONENT_CFG_KEYS_TO_STRIP}
        if extra_strip:
            cfg = {k: v for k, v in cfg.items() if k not in extra_strip}
        return cfg

    vision_config = _clean_component(_read_component("vision_tower"))
    audio_config = _clean_component(_read_component("sound_tower"), extra_strip=SOUND_TOWER_EXTRA_KEYS_TO_STRIP)

    # Extract only the fields AudioVisualFlamingoConfig cares about.
    avf_kwargs = {k: top_cfg[k] for k in AVF_CONFIG_FIELDS if k in top_cfg}
    avf_kwargs["s2_scales"] = _normalize_s2_scales(avf_kwargs.get("s2_scales"))
    avf_kwargs["image_encoder"] = _normalize_encoder_config(avf_kwargs.get("image_encoder"), "BasicImageEncoder")
    avf_kwargs["video_encoder"] = _normalize_encoder_config(avf_kwargs.get("video_encoder"), "TSPVideoEncoder")
    avf_kwargs["sound_encoder"] = _normalize_encoder_config(avf_kwargs.get("sound_encoder"), "BasicSoundEncoder")

    config = AudioVisualFlamingoConfig(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        **avf_kwargs,
    )

    # Populate media token IDs.
    media_token_ids = {}
    for name, token in AudioVisualFlamingoConfig.media_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            tokenized = tokenizer(token, add_special_tokens=False).input_ids
            if len(tokenized) != 1:
                raise ValueError(f"Media token `{token}` must map to a single tokenizer id.")
            token_id = tokenized[0]
        media_token_ids[name] = int(token_id)
    config.media_token_ids = media_token_ids

    # Populate encoder boundary token IDs.
    config.encoder_text_token_ids = {
        txt: [int(tid) for tid in tokenizer(txt).input_ids] for txt in collect_encoder_boundary_tokens(config)
    }

    return config


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


def write_processor(
    src_root: Path,
    dst_root: Path,
    config: AudioVisualFlamingoConfig,
) -> AudioVisualFlamingoProcessor:
    """Build and save the processor from source sub-components."""
    # Tokenizer: prefer llm/ subdir, fall back to root.
    tokenizer_dir = src_root / "llm" if (src_root / "llm" / "tokenizer_config.json").exists() else src_root
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

    # Image processor: from the vision_tower preprocessor config.
    vision_dir = src_root / "vision_tower"
    image_processor = AutoImageProcessor.from_pretrained(str(vision_dir), use_fast=False)

    top_cfg = _load_json(src_root / "config.json")
    processor_kwargs = {key: top_cfg[key] for key in PROCESSOR_CONFIG_FIELDS if key in top_cfg}

    # Feature extractor: construct directly (like AF3) with feature_size from the sound tower config.
    feature_size = 128
    if isinstance(config.audio_config, dict):
        feature_size = config.audio_config.get("num_mel_bins", feature_size)
    else:
        feature_size = getattr(config.audio_config, "num_mel_bins", feature_size)
    audio_sampling_rate = processor_kwargs.get("audio_sampling_rate", 16_000)
    audio_chunk_length = processor_kwargs.get("audio_chunk_length", 120)
    audio_hop_length = processor_kwargs.get("audio_hop_length", 60)
    feature_extractor = WhisperFeatureExtractor(
        feature_size=feature_size,
        chunk_length=audio_chunk_length if isinstance(audio_chunk_length, int) else 30,
        sampling_rate=audio_sampling_rate,
        hop_length=audio_hop_length,
        return_attention_mask=True,
    )

    processor = AudioVisualFlamingoProcessor(
        image_processor=image_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        image_aspect_ratio=processor_kwargs.get("image_aspect_ratio"),
        s2_scales=config.s2_scales,
        max_tiles=processor_kwargs.get("max_tiles", 12),
        num_video_frames=processor_kwargs.get("num_video_frames"),
        load_audio_in_video=config.load_audio_in_video,
        interleaved_vis_aud_in_video=config.interleaved_vis_aud_in_video,
        interleaved_video_segment_duration=processor_kwargs.get("interleaved_video_segment_duration", 30),
        mm_use_bos_eos_tokens=processor_kwargs.get("mm_use_bos_eos_tokens", False),
        audio_sampling_rate=audio_sampling_rate,
        audio_chunk_length=audio_chunk_length,
        audio_hop_length=audio_hop_length,
    )
    processor.save_pretrained(str(dst_root))
    logger.info("processor (tokenizer + preprocessors)")
    return processor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def write_model(
    src_root: Path,
    dst_root: Path,
    config: AudioVisualFlamingoConfig,
    tokenizer,
) -> AudioVisualFlamingoForConditionalGeneration:
    """Collect weights, instantiate model, load state dict, and save."""
    state = _normalize_state_dict_keys(_collect_component_state(src_root))
    if not state:
        raise FileNotFoundError("No component safetensors found under source component directories.")

    with no_init_weights():
        model = AudioVisualFlamingoForConditionalGeneration(config)

    load_res = model.load_state_dict(state, strict=True, assign=True)
    if load_res.missing_keys:
        mk = load_res.missing_keys
        raise ValueError(f"Missing keys when loading: {mk[:10]}{' ...' if len(mk) > 10 else ''}")
    if load_res.unexpected_keys:
        uk = load_res.unexpected_keys
        raise ValueError(f"Unexpected keys when loading: {uk[:10]}{' ...' if len(uk) > 10 else ''}")

    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    model.config.save_pretrained(str(dst_root))
    model.generation_config.save_pretrained(str(dst_root))
    save_model(model, str(dst_root / "model.safetensors"), metadata={"format": "pt"}, force_contiguous=False)
    logger.info("model (config + safetensors)")
    return model


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


"""
Reproducible Usage
==================

1) Download the original AudioVisualFlamingo weights (requires Git LFS):

```
git lfs install
git clone <source-repo-url>
```

This will create a folder containing the original components:
``llm/``, ``vision_tower/``, ``mm_projector/``, ``sound_tower/``, and ``sound_mm_projector/``.

2) Convert to the Hugging Face Transformers format (locally):

```
python src/transformers/models/audiovisualflamingo/convert_audiovisualflamingo_to_hf.py \\
  --src_path <source-dir> \\
  --dst_path <destination-dir>
```

3) Convert and push directly to the Hub (requires ``huggingface-cli login`` or ``HF_TOKEN``):

```
python src/transformers/models/audiovisualflamingo/convert_audiovisualflamingo_to_hf.py \\
  --src_path <source-dir> \\
  --dst_path <destination-dir> \\
  --push_to_hub <username-or-org>/audiovisualflamingo
```

This command uploads both the processor (tokenizer + image processor + feature extractor)
and the converted model (sharded safetensors + configs) to the specified Hub repository.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioVisualFlamingo to Hugging Face format.")
    ap.add_argument("--src_path", type=Path, default=DEFAULT_SRC_PATH, help="Source model root directory.")
    ap.add_argument(
        "--dst_path", type=Path, default=DEFAULT_DST_PATH, help="Destination directory for converted model."
    )
    # Backward-compatible aliases.
    ap.add_argument("--model_dir", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--output_dir", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help="Optional repository ID to push the converted assets to the Hugging Face Hub.",
    )
    args = ap.parse_args()

    src_root = (args.model_dir or args.src_path).expanduser().resolve()
    dst_root = (args.output_dir or args.dst_path).expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    # Load tokenizer early — needed for config token IDs.
    tokenizer_dir = src_root / "llm" if (src_root / "llm" / "tokenizer_config.json").exists() else src_root
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

    config = _build_config(src_root, tokenizer)
    processor = write_processor(src_root, dst_root, config)
    model = write_model(src_root, dst_root, config, tokenizer)

    if args.push_to_hub:
        logger.info("Pushing processor to the Hub ...")
        processor.push_to_hub(args.push_to_hub)
        logger.info("Pushing model to the Hub ...")
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
