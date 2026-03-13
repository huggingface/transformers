"""
Reproducible Usage
==================

1) Convert directly from a Hugging Face model ID and push to the Hub:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --model_id Qwen/Qwen3-ASR-0.6B \
  --dst_dir qwen3-asr-hf \
  --push_to_hub <username-or-org>/Qwen3-ASR-0.6B
```

2) Convert from a local directory:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --src_dir /path/to/local/model \
  --dst_dir qwen3-asr-hf
```
"""
import argparse
import json
import logging
import re
import shutil
import tempfile
import torch
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# fmt: off
STATE_DICT_MAPPING = {
    # Remove thinker. prefix from all keys since we flattened the model structure
    r"^thinker\.": r"",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    """Map checkpoint keys to transformers model keys."""
    new_key = old_key

    # Apply all regex patterns
    for pattern, replacement in STATE_DICT_MAPPING.items():
        # Check if replacement needs index shifting
        if isinstance(replacement, tuple):
            replacement_pattern, index_shift = replacement

            # Use callback to handle index shifting
            def shift_index(match):
                result = replacement_pattern
                for i, group in enumerate(match.groups(), 1):
                    if group and group.isdigit():
                        shifted_idx = int(group) + index_shift
                        result = result.replace(f"\\{i}", str(shifted_idx))
                    else:
                        result = result.replace(f"\\{i}", group)
                return result

            new_key, n = re.subn(pattern, shift_index, new_key)
        else:
            new_key, n = re.subn(pattern, replacement, new_key)

    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert checkpoint state dict to transformers format."""
    new_state_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    return new_state_dict

def write_processor(src_root: Path, dst_root: Path):
    # Load tokenizer from source model
    tokenizer = AutoTokenizer.from_pretrained(src_root)

    # Load chat template from separate file if it exists
    chat_template_file = src_root / "chat_template.json"
    chat_template = None
    if chat_template_file.exists():
        logger.info("Loading chat template from %s", chat_template_file)
        with open(chat_template_file, "r", encoding="utf-8") as f:
            chat_template_data = json.load(f)
            chat_template = chat_template_data.get("chat_template")

    processor = Qwen3ASRProcessor(
        feature_extractor=WhisperFeatureExtractor(feature_size=128),
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.save_pretrained(str(dst_root))

    logger.info("processor saved to %s", dst_root)
    return processor

def write_model(src_root: Path, dst_root: Path):
    # Load and clean up config
    config_path = src_root / "config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Clean up config for transformers compatibility
    config_dict = model_config.copy()
    
    # Add any config field mappings here if needed
    # Example: if "old_name" in config_dict:
    #     config_dict["new_name"] = config_dict.pop("old_name")
    
    # fmt: off
    # Remove unused/constant parameters at top level
    unused_keys = ["support_languages"]
    for key in unused_keys:
        config_dict.pop(key, None)

    # Flatten thinker_config structure (move to top level)
    if "thinker_config" in config_dict:
        thinker_config = config_dict.pop("thinker_config")
        
        # Move thinker_config fields to top level
        if "audio_config" in thinker_config:
            config_dict["audio_config"] = thinker_config["audio_config"]
        if "text_config" in thinker_config:
            config_dict["text_config"] = thinker_config["text_config"]
        if "audio_token_id" in thinker_config:
            config_dict["audio_token_id"] = thinker_config["audio_token_id"]
        if "initializer_range" in thinker_config:
            config_dict["initializer_range"] = thinker_config["initializer_range"]
    
    # Remove non-standard fields and auto-populated defaults from audio_config
    if "audio_config" in config_dict:
        audio_config_unused = [
            "_name_or_path", "architectures", "dtype", "use_bfloat16", "add_cross_attention",
            "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
            "finetuning_task", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
            "output_attentions", "output_hidden_states", "pad_token_id", "bos_token_id", "eos_token_id",
            "prefix", "problem_type", "pruned_heads", "return_dict", "sep_token_id", "task_specific_params",
            "tf_legacy_loss", "tie_encoder_decoder", "tie_word_embeddings", "tokenizer_class", "torchscript",
        ]
        for key in audio_config_unused:
            config_dict["audio_config"].pop(key, None)
    
    # Remove non-standard fields and auto-populated defaults from text_config
    if "text_config" in config_dict:
        text_config_unused = [
            "_name_or_path", "architectures", "dtype", "use_bfloat16", "add_cross_attention",
            "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id",
            "finetuning_task", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
            "output_attentions", "output_hidden_states", "prefix", "problem_type", "pruned_heads",
            "return_dict", "sep_token_id", "task_specific_params", "tf_legacy_loss", "tie_encoder_decoder",
            "tokenizer_class", "torchscript",
            # Note: pad_token_id, bos_token_id, eos_token_id are actual Qwen3ASRTextConfig params, keep them
        ]
        for key in text_config_unused:
            config_dict["text_config"].pop(key, None)
    # fmt: on

    config = Qwen3ASRConfig(**config_dict)
    model = Qwen3ASRForConditionalGeneration(config).to(torch.bfloat16)
    state = {}

    # Support single model.safetensors or sharded model-00001-of-NNNNN.safetensors
    shard_files = sorted(src_root.glob("model-*.safetensors"))
    single_file = src_root / "model.safetensors"

    if shard_files:
        logger.info("Found %d sharded safetensor files", len(shard_files))
        safetensor_paths = shard_files
    elif single_file.exists():
        safetensor_paths = [single_file]
    else:
        raise FileNotFoundError(f"No safetensor files found in {src_root}")

    for path in safetensor_paths:
        logger.info("Loading %s", path.name)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)

    # Convert state dict to transformers format
    logger.info("Converting state dict")
    state = convert_state_dict(state)

    load_res = model.load_state_dict(state, strict=True)
    if load_res.missing_keys:
        raise ValueError(f"Missing keys: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        raise ValueError(f"Unexpected keys: {load_res.unexpected_keys}")
    model.to(torch.bfloat16)  # Ensure model is in correct dtype before saving
    
    # Set generation config on model before saving
    model.generation_config = GenerationConfig(
        eos_token_id=[151643, 151645],
        pad_token_id=151645,
        do_sample=False,
    )
    
    model.save_pretrained(str(dst_root))

    logger.info("Model saved to %s", dst_root)
    return model

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Qwen3ASR to Hugging Face format.")
    ap.add_argument("--model_id", default=None, type=str, help="Hugging Face model ID (e.g., Qwen/Qwen3-ASR-0.6B)")
    ap.add_argument("--src_dir", default=None, help="Source model root directory (alternative to --model_id)")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for converted model")
    ap.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help=("Whether or not to push the converted model to the Hugging Face hub."),
    )
    args = ap.parse_args()

    # Determine source directory
    if args.model_id:
        logger.info("Downloading model from Hugging Face Hub: %s", args.model_id)
        src_root = Path(tempfile.mkdtemp())
        src_root = Path(snapshot_download(args.model_id, cache_dir=str(src_root)))
        logger.info("Model downloaded to: %s", src_root)
    elif args.src_dir:
        src_root = Path(args.src_dir).resolve()
    else:
        raise ValueError("Either --model_id or --src_dir must be provided")

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_root = Path(args.dst_dir).resolve()
    if dst_root.exists():
        logger.info("Removing existing destination directory: %s", dst_root)
        shutil.rmtree(dst_root)

    processor = write_processor(src_root, dst_root)
    model = write_model(src_root, dst_root)

    # Optionally push converted assets using native push_to_hub only
    if args.push_to_hub:
        logger.info("Pushing processor to the Hub ...")
        processor.push_to_hub(args.push_to_hub)
        logger.info("Pushing model to the Hub ...")
        model.push_to_hub(args.push_to_hub)

        # try loading from hub to verify
        logger.info("Verifying upload by loading from Hub: %s", args.push_to_hub)
        _ = Qwen3ASRProcessor.from_pretrained(args.push_to_hub)
        _ = Qwen3ASRForConditionalGeneration.from_pretrained(args.push_to_hub)
        logger.info("Verification successful!")


if __name__ == "__main__":
    main()
