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
Convert the official Ovis2.5-2B or Ovis2.5-9B checkpoint to native Transformers format.

The conversion never imports the checkpoint's Python files. It builds a native
``Ovis2_5Config``, uses Transformers' registered ``ovis2_5`` checkpoint conversion
mapping while loading the weights, and saves native weight names.

Examples:

```bash
python src/transformers/models/ovis2_5/convert_ovis2_5_weights_to_hf.py \
    --model-id AIDC-AI/Ovis2.5-2B \
    --dst-dir Ovis2.5-2B-hf
```

```bash
python src/transformers/models/ovis2_5/convert_ovis2_5_weights_to_hf.py \
    --src-dir /path/to/Ovis2.5-9B \
    --dst-dir Ovis2.5-9B-hf \
    --variant 9b \
    --max-shard-size 5GB
```
"""

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from transformers import AddedToken, GenerationConfig, Qwen2Tokenizer, Qwen3Config
from transformers.models.ovis2_5.configuration_ovis2_5 import Ovis2_5Config


logger = logging.getLogger(__name__)


VISUAL_SPECIAL_TOKENS = (
    ("<ovis_visual_atom>", 151669),
    ("<ovis_image_start>", 151670),
    ("<ovis_image_end>", 151671),
    ("<ovis_video_start>", 151672),
    ("<ovis_video_end>", 151673),
)
EXISTING_CHAT_SPECIAL_TOKENS = (
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
)
RAW_VISUAL_PLACEHOLDERS = ("<image>", "<video>")
MIN_PIXELS = 448 * 448

EXPECTED_TEXT_LAYOUT = {
    "vocab_size": 151936,
    "hidden_act": "silu",
    "max_position_embeddings": 40960,
    "rope_theta": 1_000_000,
    "rms_norm_eps": 1e-6,
    "head_dim": 128,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": None,
}

# These prefixes are the exact six renames registered for `ovis2_5` in
# `transformers.conversion_mapping`. They are also used here for a cheap,
# tensor-free validation of a source checkpoint before the model is allocated.
SOURCE_TO_NATIVE_PREFIXES = (
    ("llm.model.", "model.language_model."),
    ("llm.lm_head.", "lm_head."),
    ("visual_tokenizer.vit.vision_model.", "model.vision_tower.transformer."),
    ("visual_tokenizer.head.0.", "model.vision_tower.head_linear."),
    ("visual_tokenizer.head.1.", "model.vision_tower.head_norm."),
    ("vte.", "model.visual_embeddings_table."),
)

EXPECTED_VISION_LAYOUT = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_patches": -1,
    "image_size": 512,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "hidden_stride": 2,
    "window_size": 112,
    "fullatt_block_indexes": None,
    "temporal_patch_size": 1,
    "preserve_original_pe": True,
    "use_rope": True,
}

HUB_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.*",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
)


@dataclass(frozen=True)
class CheckpointSpec:
    variant: str
    text_signature: tuple[int, int, int, int, int]
    max_pixels: int


CHECKPOINT_SPECS = {
    "2b": CheckpointSpec(
        variant="2b",
        text_signature=(2048, 28, 16, 8, 6144),
        max_pixels=1344 * 1792,
    ),
    "9b": CheckpointSpec(
        variant="9b",
        text_signature=(4096, 36, 32, 8, 12288),
        max_pixels=1792 * 1792,
    ),
}
SPEC_BY_TEXT_SIGNATURE = {spec.text_signature: spec for spec in CHECKPOINT_SPECS.values()}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_source(
    model_id: str | None,
    src_dir: str | None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Resolve a local source without ever downloading or executing custom Python code."""
    if model_id is not None:
        logger.info("Downloading checkpoint metadata and safetensors from %s", model_id)
        source = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=list(HUB_ALLOW_PATTERNS),
        )
        return Path(source).resolve()

    if src_dir is None:
        raise ValueError("Exactly one of `model_id` or `src_dir` is required.")
    source = Path(src_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    return source


def detect_checkpoint_spec(config_dict: dict[str, Any], requested_variant: str = "auto") -> CheckpointSpec:
    """Identify the released 2B/9B layout from structural config fields."""
    if config_dict.get("model_type") != "ovis2_5":
        raise ValueError(
            f"Expected an Ovis2.5 source config with `model_type='ovis2_5'`, got {config_dict.get('model_type')!r}."
        )

    text_config = config_dict.get("llm_config")
    vision_config = config_dict.get("vit_config")
    if not isinstance(text_config, dict) or not isinstance(vision_config, dict):
        raise ValueError("The source config must contain dictionary-valued `llm_config` and `vit_config` fields.")
    if text_config.get("model_type") != "qwen3":
        raise ValueError(f"Expected a Qwen3 text tower, got {text_config.get('model_type')!r}.")

    signature = tuple(
        text_config.get(field)
        for field in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        )
    )
    spec = SPEC_BY_TEXT_SIGNATURE.get(signature)
    if spec is None:
        supported = ", ".join(f"{item.variant}: {item.text_signature}" for item in CHECKPOINT_SPECS.values())
        raise ValueError(f"Unsupported Ovis2.5 text layout {signature}. Supported released layouts are {supported}.")
    if requested_variant != "auto" and requested_variant != spec.variant:
        raise ValueError(
            f"`--variant {requested_variant}` does not match the detected {spec.variant} checkpoint layout {signature}."
        )

    text_mismatches = {
        field: (text_config.get(field), expected)
        for field, expected in EXPECTED_TEXT_LAYOUT.items()
        if text_config.get(field) != expected
    }
    expected_tied_embeddings = spec.variant == "2b"
    if bool(text_config.get("tie_word_embeddings")) != expected_tied_embeddings:
        text_mismatches["tie_word_embeddings"] = (
            text_config.get("tie_word_embeddings"),
            expected_tied_embeddings,
        )
    if text_mismatches:
        raise ValueError(f"Unsupported Ovis2.5 text layout; mismatched fields: {text_mismatches}")
    if config_dict.get("visual_vocab_size") != 65536:
        raise ValueError(
            f"Expected the released visual vocabulary size 65536, got {config_dict.get('visual_vocab_size')!r}."
        )

    vision_mismatches = {
        field: (vision_config.get(field), expected)
        for field, expected in EXPECTED_VISION_LAYOUT.items()
        if vision_config.get(field) != expected
    }
    if vision_mismatches:
        raise ValueError(f"Unsupported Ovis2.5 vision layout; mismatched fields: {vision_mismatches}")

    logger.info(
        "Detected Ovis2.5-%s layout (min_pixels=%d, max_pixels=%d)",
        spec.variant.upper(),
        MIN_PIXELS,
        spec.max_pixels,
    )
    return spec


def _clean_subconfig(config_dict: dict[str, Any], *, keep_model_type: bool) -> dict[str, Any]:
    """Drop source-only provenance and legacy serialization fields."""
    cleaned = dict(config_dict)
    for key in (
        "_attn_implementation_autoset",
        "_name_or_path",
        "architectures",
        "auto_map",
        "torch_dtype",
        "transformers_version",
        "use_bfloat16",
    ):
        cleaned.pop(key, None)
    if not keep_model_type:
        cleaned.pop("model_type", None)
    return cleaned


def build_native_config(config_dict: dict[str, Any]) -> Ovis2_5Config:
    """Translate remote-code config aliases into the native composite config."""
    text_config = _clean_subconfig(config_dict["llm_config"], keep_model_type=True)
    vision_config = _clean_subconfig(config_dict["vit_config"], keep_model_type=False)
    dtype = config_dict.get("torch_dtype", "bfloat16")
    if dtype != "bfloat16":
        raise ValueError(f"The released Ovis2.5 checkpoints are bfloat16, but the config declares {dtype!r}.")

    token_ids = dict(VISUAL_SPECIAL_TOKENS)
    config = Ovis2_5Config(
        text_config=text_config,
        vision_config=vision_config,
        visual_vocab_size=config_dict["visual_vocab_size"],
        visual_atom_token_id=token_ids["<ovis_visual_atom>"],
        image_start_token_id=token_ids["<ovis_image_start>"],
        image_end_token_id=token_ids["<ovis_image_end>"],
        video_start_token_id=token_ids["<ovis_video_start>"],
        video_end_token_id=token_ids["<ovis_video_end>"],
        image_token_id=token_ids["<ovis_visual_atom>"],
        video_token_id=token_ids["<ovis_visual_atom>"],
        dtype=dtype,
    )
    config.architectures = ["Ovis2_5ForConditionalGeneration"]

    text_model_config = config.text_config
    if not isinstance(text_model_config, Qwen3Config):
        raise TypeError(f"Expected a normalized Qwen3Config, got {type(text_model_config).__name__}.")
    if text_model_config.vocab_size <= VISUAL_SPECIAL_TOKENS[-1][1]:
        raise ValueError(
            "The model vocabulary does not contain the required positive visual token IDs "
            f"{VISUAL_SPECIAL_TOKENS[0][1]}..{VISUAL_SPECIAL_TOKENS[-1][1]}."
        )
    if "auto_map" in config.to_dict():
        raise ValueError("Native Ovis2.5 output config unexpectedly contains `auto_map`.")
    return config


def build_tokenizer(source: Path) -> Qwen2Tokenizer:
    """Register the five positive visual tokens at their released embedding rows."""
    tokenizer = Qwen2Tokenizer.from_pretrained(str(source), return_token_type_ids=False)
    existing_ids = [tokenizer.convert_tokens_to_ids(token) for token, _ in VISUAL_SPECIAL_TOKENS]
    expected_ids = [token_id for _, token_id in VISUAL_SPECIAL_TOKENS]

    if all(token_id is None for token_id in existing_ids):
        first_token_id = VISUAL_SPECIAL_TOKENS[0][1]
        if len(tokenizer) != first_token_id:
            raise ValueError(
                f"Expected the five Ovis2.5 tokens to start at ID {first_token_id}, "
                f"but the source tokenizer length is {len(tokenizer)}."
            )
        added = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(token, special=True, normalized=False) for token, _ in VISUAL_SPECIAL_TOKENS
                ]
            },
            replace_extra_special_tokens=False,
        )
        if added != len(VISUAL_SPECIAL_TOKENS):
            raise ValueError(f"Expected to add five Ovis2.5 visual tokens, but the tokenizer added {added}.")
    elif existing_ids != expected_ids:
        raise ValueError(
            "Ovis2.5 visual tokens are partially registered or have unexpected IDs: "
            f"got {existing_ids}, expected {expected_ids}."
        )

    for token, expected_id in VISUAL_SPECIAL_TOKENS:
        actual_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer(token, add_special_tokens=False).input_ids
        if actual_id != expected_id or encoded != [expected_id]:
            raise ValueError(
                f"Token {token!r} must be one positive token with ID {expected_id}, "
                f"but got token_id={actual_id} and encoded={encoded}."
            )

    for token, expected_id in EXISTING_CHAT_SPECIAL_TOKENS:
        actual_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer(token, add_special_tokens=False).input_ids
        if actual_id != expected_id or encoded != [expected_id]:
            raise ValueError(
                f"Adding Ovis2.5 visual tokens changed existing chat token {token!r}: "
                f"got token_id={actual_id} and encoded={encoded}, expected [{expected_id}]."
            )
    for placeholder in RAW_VISUAL_PLACEHOLDERS:
        placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
        encoded = tokenizer(placeholder, add_special_tokens=False).input_ids
        if placeholder_id is not None or len(encoded) <= 1:
            raise ValueError(
                f"Ovis2.5 requires {placeholder!r} to remain an unregistered raw processor placeholder, "
                f"but got token_id={placeholder_id} and encoded={encoded}."
            )
    if not tokenizer.chat_template:
        raise ValueError("The source tokenizer does not contain the official Ovis2.5 chat template.")
    return tokenizer


def build_processor(tokenizer: Qwen2Tokenizer, spec: CheckpointSpec):
    """Build native image and video processor metadata for one checkpoint size."""
    try:
        from transformers.models.ovis2_5.image_processing_ovis2_5 import Ovis2_5ImageProcessor
        from transformers.models.ovis2_5.processing_ovis2_5 import Ovis2_5Processor
        from transformers.models.ovis2_5.video_processing_ovis2_5 import Ovis2_5VideoProcessor
    except ImportError as error:
        raise ImportError(
            "Creating the native Ovis2.5 image/video processor requires the Transformers vision dependencies. "
            "Install them (including torchvision) and rerun the converter."
        ) from error

    image_processor = Ovis2_5ImageProcessor(min_pixels=MIN_PIXELS, max_pixels=spec.max_pixels)
    video_processor = Ovis2_5VideoProcessor(min_pixels=MIN_PIXELS, max_pixels=spec.max_pixels)
    processor = Ovis2_5Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )

    for name, component in (("image", image_processor), ("video", video_processor)):
        if component.size["shortest_edge"] != MIN_PIXELS or component.size["longest_edge"] != spec.max_pixels:
            raise ValueError(
                f"Native {name} processor has unexpected pixel bounds {component.size}; "
                f"expected ({MIN_PIXELS}, {spec.max_pixels})."
            )
    return processor


def build_generation_config(source: Path, model_config: Ovis2_5Config) -> GenerationConfig:
    generation_path = source / "generation_config.json"
    if not generation_path.is_file():
        return GenerationConfig.from_model_config(model_config)
    generation_dict = read_json(generation_path)
    generation_dict.pop("transformers_version", None)
    return GenerationConfig.from_dict(generation_dict)


def checkpoint_keys(directory: Path) -> set[str]:
    """Read checkpoint key names from the index/header without loading tensor data."""
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty safetensors index: {index_path}")
        return set(weight_map)

    safetensor_paths = sorted(directory.glob("model*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No model safetensors found in {directory}")

    from safetensors import safe_open

    keys = set()
    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _to_native_key(source_key: str) -> str:
    for source_prefix, native_prefix in SOURCE_TO_NATIVE_PREFIXES:
        if source_key.startswith(source_prefix):
            return native_prefix + source_key.removeprefix(source_prefix)
    raise ValueError(f"Source checkpoint key is not covered by the Ovis2.5 conversion mapping: {source_key}")


def validate_source_checkpoint(source: Path, spec: CheckpointSpec) -> None:
    source_keys = checkpoint_keys(source)
    native_keys = [_to_native_key(key) for key in source_keys]
    if len(native_keys) != len(set(native_keys)):
        raise ValueError("Ovis2.5 source-to-native weight renaming produces a key collision.")

    required_native_keys = {
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.transformer.embeddings.patch_embedding.weight",
        "model.vision_tower.head_linear.weight",
        "model.vision_tower.head_norm.weight",
        "model.visual_embeddings_table.weight",
    }
    if spec.variant == "9b":
        required_native_keys.add("lm_head.weight")
    missing = required_native_keys.difference(native_keys)
    if missing:
        raise ValueError(f"Source checkpoint is missing required Ovis2.5 weights after mapping: {sorted(missing)}")
    if spec.variant == "2b" and any(key.startswith("llm.lm_head.") for key in source_keys):
        raise ValueError("The released tied-embedding Ovis2.5-2B layout must not contain a separate `llm.lm_head`.")

    logger.info("Validated %d source keys; every key has one collision-free native rename", len(source_keys))


def _validate_loading_info(loading_info: dict[str, Any], config: Ovis2_5Config) -> None:
    text_model_config = config.text_config
    if not isinstance(text_model_config, Qwen3Config):
        raise TypeError(f"Expected a normalized Qwen3Config, got {type(text_model_config).__name__}.")
    allowed_missing = {"lm_head.weight"} if text_model_config.tie_word_embeddings else set()
    missing = set(loading_info.get("missing_keys", ())).difference(allowed_missing)
    unexpected = set(loading_info.get("unexpected_keys", ()))
    mismatched = loading_info.get("mismatched_keys", ())
    errors = loading_info.get("error_msgs", ())
    if missing or unexpected or mismatched or errors:
        raise ValueError(
            "Native Ovis2.5 weight loading was not exact: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"mismatched={mismatched}, errors={errors}"
        )


def convert_model(
    source: Path,
    destination: Path,
    config: Ovis2_5Config,
    generation_config: GenerationConfig,
    max_shard_size: str,
):
    """Load through the central mapping and save native (not reverse-mapped) keys."""
    from transformers.models.ovis2_5.modeling_ovis2_5 import Ovis2_5ForConditionalGeneration

    model, loading_info = Ovis2_5ForConditionalGeneration.from_pretrained(
        str(source),
        config=config,
        dtype="auto",
        local_files_only=True,
        output_loading_info=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    _validate_loading_info(loading_info, config)
    model.generation_config = generation_config
    model.save_pretrained(
        str(destination),
        max_shard_size=max_shard_size,
        save_original_format=False,
    )
    return model


def _contains_auto_map(value: Any) -> bool:
    if isinstance(value, dict):
        return "auto_map" in value or any(_contains_auto_map(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_auto_map(item) for item in value)
    return False


def validate_native_output(destination: Path, spec: CheckpointSpec) -> None:
    """Verify native names, token IDs, processor bounds, and no remote-code metadata."""
    config_dict = read_json(destination / "config.json")
    if config_dict.get("model_type") != "ovis2_5":
        raise ValueError(f"Converted config has unexpected model type: {config_dict.get('model_type')!r}")
    if config_dict.get("architectures") != ["Ovis2_5ForConditionalGeneration"]:
        raise ValueError(f"Converted config has unexpected architectures: {config_dict.get('architectures')!r}")
    if config_dict.get("vision_config", {}).get("model_type") != "ovis2_5_vision":
        raise ValueError("Converted config does not reference the native Ovis2.5 vision config.")

    processor_dict = read_json(destination / "processor_config.json")
    if processor_dict.get("processor_class") != "Ovis2_5Processor":
        raise ValueError(f"Converted processor has unexpected class: {processor_dict.get('processor_class')!r}")
    component_types = (
        ("image_processor", "image_processor_type", "Ovis2_5ImageProcessor"),
        ("video_processor", "video_processor_type", "Ovis2_5VideoProcessor"),
    )
    for component_name, type_key, expected_type in component_types:
        component = processor_dict.get(component_name, {})
        if component.get(type_key) != expected_type:
            raise ValueError(f"Converted {component_name} has unexpected type metadata: {component.get(type_key)!r}")
        expected_size = {"shortest_edge": MIN_PIXELS, "longest_edge": spec.max_pixels}
        if component.get("size") != expected_size:
            raise ValueError(
                f"Converted {component_name} has size metadata {component.get('size')!r}, expected {expected_size}."
            )

    for filename in ("config.json", "processor_config.json", "tokenizer_config.json"):
        data = read_json(destination / filename)
        if _contains_auto_map(data):
            raise ValueError(f"Converted native metadata must not contain `auto_map`, but found it in {filename}.")

    tokenizer = Qwen2Tokenizer.from_pretrained(str(destination))
    for token, expected_id in VISUAL_SPECIAL_TOKENS:
        actual_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer(token, add_special_tokens=False).input_ids
        if actual_id != expected_id or encoded != [expected_id]:
            raise ValueError(
                f"Saved token {token!r} has token_id={actual_id} and encoded={encoded}; expected [{expected_id}]."
            )

    native_keys = checkpoint_keys(destination)
    source_prefixes = tuple(source for source, _ in SOURCE_TO_NATIVE_PREFIXES)
    non_native_keys = sorted(key for key in native_keys if key.startswith(source_prefixes))
    if non_native_keys:
        raise ValueError(f"Converted checkpoint still contains source-format weight names: {non_native_keys[:10]}")
    required = {
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.transformer.embeddings.patch_embedding.weight",
        "model.vision_tower.head_linear.weight",
        "model.vision_tower.head_norm.weight",
        "model.visual_embeddings_table.weight",
    }
    if spec.variant == "9b":
        required.add("lm_head.weight")
    missing = required.difference(native_keys)
    if missing:
        raise ValueError(f"Converted checkpoint is missing required native weights: {sorted(missing)}")


def prepare_destination(source: Path, destination: Path, overwrite: bool) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination directories must be distinct and must not contain one another.")
    if destination.exists():
        if not overwrite:
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Destination is not empty: {destination}. Choose another path or pass `--overwrite`."
                )
        else:
            logger.info("Removing existing destination because --overwrite was supplied: %s", destination)
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def convert_checkpoint(
    source: Path,
    destination: Path,
    requested_variant: str = "auto",
    max_shard_size: str = "5GB",
    overwrite: bool = False,
):
    source_config = read_json(source / "config.json")
    spec = detect_checkpoint_spec(source_config, requested_variant)
    validate_source_checkpoint(source, spec)
    native_config = build_native_config(source_config)
    tokenizer = build_tokenizer(source)
    text_model_config = native_config.text_config
    if not isinstance(text_model_config, Qwen3Config):
        raise TypeError(f"Expected a normalized Qwen3Config, got {type(text_model_config).__name__}.")
    if len(tokenizer) > text_model_config.vocab_size:
        raise ValueError(
            f"Tokenizer length {len(tokenizer)} exceeds model vocabulary size {text_model_config.vocab_size}."
        )
    processor = build_processor(tokenizer, spec)
    generation_config = build_generation_config(source, native_config)

    prepare_destination(source, destination, overwrite)
    logger.info("Loading and converting Ovis2.5-%s weights", spec.variant.upper())
    model = convert_model(source, destination, native_config, generation_config, max_shard_size)
    processor.save_pretrained(str(destination))
    validate_native_output(destination, spec)
    logger.info("Native Ovis2.5-%s checkpoint saved and validated at %s", spec.variant.upper(), destination)
    return model, processor, spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert official Ovis2.5-2B/9B safetensors to native Transformers format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model-id AIDC-AI/Ovis2.5-2B --dst-dir Ovis2.5-2B-hf\n"
            "  %(prog)s --src-dir ./Ovis2.5-9B --dst-dir ./Ovis2.5-9B-hf --variant 9b\n\n"
            "The converter never loads the source repository's Python files and writes no `auto_map` metadata."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-id",
        "--model_id",
        dest="model_id",
        help="Hugging Face Hub model ID for the original checkpoint.",
    )
    source_group.add_argument(
        "--src-dir",
        "--src_dir",
        dest="src_dir",
        help="Local directory containing the original checkpoint.",
    )
    parser.add_argument(
        "--dst-dir",
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="Destination directory for the converted native checkpoint.",
    )
    parser.add_argument(
        "--variant",
        choices=("auto", "2b", "9b"),
        default="auto",
        help="Expected checkpoint layout. Defaults to structural auto-detection.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub revision used with --model-id.",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        dest="cache_dir",
        default=None,
        help="Optional Hugging Face Hub cache directory used with --model-id.",
    )
    parser.add_argument(
        "--max-shard-size",
        "--max_shard_size",
        dest="max_shard_size",
        default="5GB",
        help="Maximum size of each saved safetensors shard (default: 5GB).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a non-empty destination directory before conversion.",
    )
    parser.add_argument(
        "--push-to-hub",
        "--push_to_hub",
        dest="push_to_hub",
        default=None,
        metavar="REPO_ID",
        help="Optionally upload the validated destination folder to this Hub repository.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if args.src_dir is not None and (args.revision is not None or args.cache_dir is not None):
        parser.error("--revision and --cache-dir can only be used with --model-id.")

    source = resolve_source(args.model_id, args.src_dir, args.revision, args.cache_dir)
    destination = Path(args.dst_dir).expanduser().resolve()
    convert_checkpoint(
        source=source,
        destination=destination,
        requested_variant=args.variant,
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )

    if args.push_to_hub:
        logger.info("Uploading validated native checkpoint to %s", args.push_to_hub)
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.push_to_hub, repo_type="model", folder_path=str(destination))
        logger.info("Upload complete: https://huggingface.co/%s", args.push_to_hub)


if __name__ == "__main__":
    main()
