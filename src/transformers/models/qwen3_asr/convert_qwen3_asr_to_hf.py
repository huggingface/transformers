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
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

from transformers import (
    AutoTokenizer,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    config = Qwen3ASRConfig.from_pretrained(src_root)

    model = Qwen3ASRForConditionalGeneration(config)

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

    load_res = model.load_state_dict(state, strict=True)

    if load_res.missing_keys:
        raise ValueError(f"Missing keys: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        raise ValueError(f"Unexpected keys: {load_res.unexpected_keys}")

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
