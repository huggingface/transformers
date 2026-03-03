"""
Reproducible Usage
==================

1) Convert directly from a Hugging Face model ID and push to the Hub:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --model_id Qwen/Qwen3-ASR-0.6B \
  --dst_dir qwen3-asr-hf \
  --push_to_hub <username-or-org>/qwen3-asr
```

2) Convert from a local directory:

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --src_dir /path/to/local/model \
  --dst_dir qwen3-asr-hf
```

The script will automatically download the model from Hugging Face Hub if a model_id is provided.
This command uploads both the processor (tokenizer + feature extractor) and the converted
model (sharded safetensors + configs) to the specified Hub repository.
"""
import argparse
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
    # fmt: off
    chat_template = (
        "{% set ns = namespace(system_text='') %}"
        "{% for m in messages %}"
            "{% if m.role == 'system' %}"
                "{% if m.content is string %}"
                    "{% set ns.system_text = ns.system_text + m.content %}"
                "{% else %}"
                    "{% for c in m.content %}"
                        "{% if c.type == 'text' and (c.text is defined) %}"
                            "{% set ns.system_text = ns.system_text + c.text %}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endif %}"
        "{% endfor %}"

        "{% set ns2 = namespace(audio_tokens='') %}"
        "{% for m in messages %}"
            "{% if m.content is not string %}"
                "{% for c in m.content %}"
                    "{% if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) %}"
                        "{% set ns2.audio_tokens = ns2.audio_tokens + '<|audio_start|><|audio_pad|><|audio_end|>' %}"
                    "{% endif %}"
                "{% endfor %}"
            "{% endif %}"
        "{% endfor %}"

        "{{ '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' }}"
        "{{ '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' }}"
        "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )
    # fmt: on

    processor = Qwen3ASRProcessor(
        feature_extractor=WhisperFeatureExtractor(),
        tokenizer=AutoTokenizer.from_pretrained(src_root),  # check this
        chat_template=chat_template,
    )
    processor.save_pretrained(str(dst_root))

    logger.info("processor saved to %s", dst_root)
    return processor

def write_model(src_root: Path, dst_root: Path):
    config = Qwen3ASRConfig.from_pretrained(src_root)

    model = Qwen3ASRForConditionalGeneration(config)

    state = {}

    model_path = src_root / "model.safetensors"
    with safe_open(model_path, framework="pt", device="cpu") as f:
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


if __name__ == "__main__":
    main()
