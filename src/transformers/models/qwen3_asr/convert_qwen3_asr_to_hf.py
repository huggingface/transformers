"""
Reproducible Usage
==================

1) Download the original Qwen3-ASR weights (requires Git LFS):

```
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-ASR-0.6B
```

2) Convert to the Hugging Face Transformers format (locally):

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py --src_dir qwen3-asr-0.6b --dst_dir qwen3-asr-hf
```

3) Convert and push directly to the Hub (requires `huggingface-cli login` or `HF_TOKEN`):

```
python src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py \
  --src_dir qwen3-asr-0.6b \
  --dst_dir qwen3-asr-hf \
  --push_to_hub <username-or-org>/qwen3-asr
```

This command uploads both the processor (tokenizer + feature extractor) and the converted
model (sharded safetensors + configs) to the specified Hub repository.
"""
import argparse
import logging
from pathlib import Path

from safetensors.torch import safe_open

from transformers import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
    WhisperFeatureExtractor,
    AutoTokenizer,
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
        feature_extractor=WhisperFeatureExtractor.from_pretrained(src_root),
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
    ap.add_argument("--src_dir", required=True, help="Source model root directory")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for converted model")
    ap.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help=("Whether or not to push the converted model to the Hugging Face hub."),
    )
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_root = Path(args.dst_dir).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

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