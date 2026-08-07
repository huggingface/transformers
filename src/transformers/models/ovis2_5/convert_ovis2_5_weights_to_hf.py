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

r"""Convert official Ovis2.5 checkpoints to the native Transformers format.

The checkpoint Python files are not downloaded or imported. Transformers v5 applies the registered
``ovis2_5`` ``WeightRenaming`` rules while streaming tensors through ``from_pretrained``. Saving with
``save_original_format=False`` then serializes the native names instead of reversing those renames.

Example:

```bash
python src/transformers/models/ovis2_5/convert_ovis2_5_weights_to_hf.py \
    --input_model_id AIDC-AI/Ovis2.5-2B \
    --output_dir Ovis2.5-2B-hf
```

``--input_model_id`` may also point to a local checkpoint directory.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from transformers import (
    AddedToken,
    AutoTokenizer,
    GenerationConfig,
    Ovis2_5Config,
    Ovis2_5ForConditionalGeneration,
    Ovis2_5ImageProcessor,
    Ovis2_5Processor,
    Ovis2_5VideoProcessor,
)


logger = logging.getLogger(__name__)

MIN_PIXELS = 448 * 448
MAX_PIXELS_BY_TEXT_HIDDEN_SIZE = {
    2048: 1344 * 1792,  # Ovis2.5-2B
    4096: 1792 * 1792,  # Ovis2.5-9B
}
VISUAL_SPECIAL_TOKENS = (
    ("<ovis_visual_atom>", "visual_atom_token_id"),
    ("<ovis_image_start>", "image_start_token_id"),
    ("<ovis_image_end>", "image_end_token_id"),
    ("<ovis_video_start>", "video_start_token_id"),
    ("<ovis_video_end>", "video_end_token_id"),
)
HUB_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.safetensors",
    "merges.txt",
    "tokenizer.model",
    "vocab.json",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input(input_model_id: str, revision: str | None = None) -> Path:
    input_path = Path(input_model_id).expanduser()
    if input_path.exists():
        if not input_path.is_dir():
            raise ValueError(f"The input path must be a checkpoint directory, got {input_path}.")
        return input_path.resolve()

    logger.info("Downloading %s", input_model_id)
    return Path(
        snapshot_download(
            repo_id=input_model_id,
            revision=revision,
            allow_patterns=list(HUB_ALLOW_PATTERNS),
        )
    ).resolve()


def _clean_subconfig(config_dict: dict[str, Any], remove_model_type: bool = False) -> dict[str, Any]:
    config_dict = dict(config_dict)
    for key in (
        "_attn_implementation_autoset",
        "_name_or_path",
        "architectures",
        "auto_map",
        "torch_dtype",
        "transformers_version",
        "use_bfloat16",
    ):
        config_dict.pop(key, None)
    if remove_model_type:
        config_dict.pop("model_type", None)
    return config_dict


def convert_config(source: Path) -> tuple[Ovis2_5Config, int]:
    original_config = read_json(source / "config.json")
    if "llm_config" not in original_config or "vit_config" not in original_config:
        raise ValueError("The source config must contain `llm_config` and `vit_config`.")

    config = Ovis2_5Config(
        text_config=_clean_subconfig(original_config["llm_config"]),
        vision_config=_clean_subconfig(original_config["vit_config"], remove_model_type=True),
        visual_vocab_size=original_config.get("visual_vocab_size", 65536),
        dtype=original_config.get("torch_dtype", "bfloat16"),
    )
    config.architectures = ["Ovis2_5ForConditionalGeneration"]

    hidden_size = config.text_config.hidden_size
    if hidden_size not in MAX_PIXELS_BY_TEXT_HIDDEN_SIZE:
        raise ValueError(
            "The converter supports the released Ovis2.5-2B and Ovis2.5-9B checkpoints, "
            f"but the text hidden size is {hidden_size}."
        )
    return config, MAX_PIXELS_BY_TEXT_HIDDEN_SIZE[hidden_size]


def convert_tokenizer(source: Path, config: Ovis2_5Config):
    tokenizer = AutoTokenizer.from_pretrained(
        str(source),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer is None:
        raise RuntimeError(f"Could not load the tokenizer from {source}.")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(token, special=True, normalized=False) for token, _ in VISUAL_SPECIAL_TOKENS
            ]
        },
        replace_extra_special_tokens=False,
    )

    expected_ids = [getattr(config, config_attribute) for _, config_attribute in VISUAL_SPECIAL_TOKENS]
    actual_ids = [tokenizer.convert_tokens_to_ids(token) for token, _ in VISUAL_SPECIAL_TOKENS]
    if actual_ids != expected_ids:
        raise ValueError(f"Unexpected Ovis2.5 visual token IDs: got {actual_ids}, expected {expected_ids}.")
    if len(tokenizer) > config.text_config.vocab_size:
        raise ValueError(
            f"Tokenizer length {len(tokenizer)} exceeds the model vocabulary size {config.text_config.vocab_size}."
        )
    return tokenizer


def convert_processor(source: Path, config: Ovis2_5Config, max_pixels: int) -> Ovis2_5Processor:
    tokenizer = convert_tokenizer(source, config)
    if tokenizer.chat_template is None:
        raise ValueError("The source checkpoint does not contain an Ovis2.5 chat template.")
    size = {"shortest_edge": MIN_PIXELS, "longest_edge": max_pixels}
    image_processor = Ovis2_5ImageProcessor(size=size)
    video_processor = Ovis2_5VideoProcessor(size=size)
    return Ovis2_5Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )


def convert_generation_config(source: Path, model_config: Ovis2_5Config) -> GenerationConfig:
    generation_config_path = source / "generation_config.json"
    if not generation_config_path.is_file():
        return GenerationConfig.from_model_config(model_config)

    generation_config_dict = read_json(generation_config_path)
    generation_config_dict.pop("transformers_version", None)
    return GenerationConfig.from_dict(generation_config_dict)


def _validate_loading_info(loading_info: dict[str, Any], config: Ovis2_5Config) -> None:
    allowed_missing = {"lm_head.weight"} if config.text_config.tie_word_embeddings else set()
    missing_keys = set(loading_info.get("missing_keys", ())).difference(allowed_missing)
    unexpected_keys = set(loading_info.get("unexpected_keys", ()))
    mismatched_keys = loading_info.get("mismatched_keys", ())
    error_messages = loading_info.get("error_msgs", ())
    if missing_keys or unexpected_keys or mismatched_keys or error_messages:
        raise RuntimeError(
            "Ovis2.5 weights did not load exactly: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}, "
            f"mismatched={mismatched_keys}, errors={error_messages}."
        )


def convert_checkpoint(
    input_model_id: str,
    output_dir: str,
    revision: str | None = None,
    max_shard_size: str = "5GB",
) -> Path:
    source = resolve_input(input_model_id, revision)
    destination = Path(output_dir).expanduser().resolve()
    if source == destination:
        raise ValueError("The input and output directories must be different.")
    if destination.exists():
        raise FileExistsError(f"The output directory already exists: {destination}")

    config, max_pixels = convert_config(source)
    processor = convert_processor(source, config, max_pixels)
    generation_config = convert_generation_config(source, config)

    logger.info("Loading and converting weights with the registered Ovis2.5 mapping")
    model, loading_info = Ovis2_5ForConditionalGeneration.from_pretrained(
        str(source),
        config=config,
        dtype="auto",
        local_files_only=True,
        output_loading_info=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    _validate_loading_info(loading_info, model.config)
    model.generation_config = generation_config

    destination.mkdir(parents=True)
    # The v5 loader records the applied renames and reverses them on a default save. Keep the converted native names.
    model.save_pretrained(
        destination,
        max_shard_size=max_shard_size,
        save_original_format=False,
    )
    processor.save_pretrained(destination)
    del model
    gc.collect()

    logger.info("Reloading the native checkpoint")
    model, loading_info = Ovis2_5ForConditionalGeneration.from_pretrained(
        destination,
        dtype="auto",
        local_files_only=True,
        output_loading_info=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    _validate_loading_info(loading_info, model.config)
    Ovis2_5Processor.from_pretrained(destination, local_files_only=True)
    del model
    gc.collect()
    logger.info("Converted checkpoint saved to %s", destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input_model_id",
        required=True,
        help="Official Hub model ID or local checkpoint directory.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for the native checkpoint.")
    parser.add_argument("--revision", default=None, help="Optional Hub revision.")
    parser.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Maximum size of each saved safetensors shard.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        metavar="REPO_ID",
        help="Optionally upload the validated output directory to this Hub repository.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_path = convert_checkpoint(
        input_model_id=args.input_model_id,
        output_dir=args.output_dir,
        revision=args.revision,
        max_shard_size=args.max_shard_size,
    )

    if args.push_to_hub:
        logger.info("Uploading the converted checkpoint to %s", args.push_to_hub)
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.push_to_hub, repo_type="model", folder_path=output_path)


if __name__ == "__main__":
    main()
