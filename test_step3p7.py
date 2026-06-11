# ruff: noqa: E402
"""Smoke test for the in-tree Step 3.7 Flash Transformers implementation.

This script intentionally loads the model with ``trust_remote_code=False`` so it
exercises the code registered in this repository instead of the Python files in
the checkpoint directory.

Example:
    python3 test_step3p7.py --model-path /data/Step-3.7-Flash --max-new-tokens 50

For a cheap import/AutoClass-only check that does not instantiate the large
checkpoint:
    python3 test_step3p7.py --skip-model-load
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import sys
from pathlib import Path


def _prefer_repo_transformers() -> None:
    """Make the script use this checkout's ``src/transformers`` first."""

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


def _patch_local_dependency_mismatch() -> None:
    """Allow this checkout to import in local envs with an older hf-hub.

    The repo currently requires huggingface-hub>=1.5.0. Some local dev images
    still have an older hub installed. This patch is only for running this smoke
    script in that environment; it does not change library behavior.
    """

    original_version = importlib_metadata.version

    def patched_version(package_name: str) -> str:
        if package_name == "huggingface-hub":
            try:
                current = original_version(package_name)
            except importlib_metadata.PackageNotFoundError:
                return original_version(package_name)
            major_minor = tuple(int(part) for part in current.split(".")[:2] if part.isdigit())
            if major_minor and major_minor < (1, 5):
                return "1.5.0"
        return original_version(package_name)

    importlib_metadata.version = patched_version


_prefer_repo_transformers()
_patch_local_dependency_mismatch()

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: E402


try:  # noqa: E402
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
except Exception:  # pragma: no cover - compatibility with older installs
    TokenizersBackend = AutoTokenizer


DEFAULT_MODEL_PATH = Path("/mnt/chensiyu-jfs/multi-hardware/models/step3p7-flash/step3p7_s0514_mtp3_bf16")
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Step 3.7 Flash smoke/inference test.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Checkpoint directory.")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="Image URL used in the demo prompt.")
    parser.add_argument("--prompt", default="What is in this picture?", help="Text prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum generated tokens.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to from_pretrained.")
    parser.add_argument("--dtype", default="auto", help="dtype passed to from_pretrained.")
    parser.add_argument(
        "--pin-vision-device",
        default="cuda:0",
        help="Move vision_model and vit_large_projector to this device after loading. Use 'none' to disable.",
    )
    parser.add_argument(
        "--expect-cat",
        action="store_true",
        default=True,
        help="Fail if the decoded response does not mention cat/cats.",
    )
    parser.add_argument(
        "--no-expect-cat",
        action="store_false",
        dest="expect_cat",
        help="Do not check whether the decoded response mentions cats.",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Only verify local AutoConfig/AutoProcessor/AutoTokenizer mappings; do not load large weights.",
    )
    return parser.parse_args()


def build_messages(image_url: str, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def summarize_inputs(inputs, image_token_id: int) -> None:
    print(f"[inputs] keys={list(inputs.keys())}")
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"[inputs] {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"[inputs] {key}: {value!r}")
    if "input_ids" in inputs:
        image_token_count = (inputs["input_ids"] == image_token_id).sum().item()
        print(f"[inputs] image_token_count={image_token_count}")
    if "pixel_values" not in inputs:
        raise ValueError("Processor output has no pixel_values; the image was not passed to the model.")


def maybe_pin_vision_modules(model, pin_vision_device: str):
    if pin_vision_device.lower() in {"", "none", "false", "off"}:
        print("[vision-pin] disabled")
        return
    if not torch.cuda.is_available() and pin_vision_device.startswith("cuda"):
        print(f"[vision-pin] cuda unavailable, skip pinning to {pin_vision_device}")
        return

    device = torch.device(pin_vision_device)
    vision_model = getattr(getattr(model, "model", None), "vision_model", None)
    projector = getattr(getattr(model, "model", None), "vit_large_projector", None)
    if vision_model is None or projector is None:
        print("[vision-pin] model does not expose model.vision_model/vit_large_projector; skip")
        return

    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(vision_model, recurse=True)
        remove_hook_from_module(projector, recurse=True)
        print("[vision-pin] removed accelerate hooks from vision path")
    except Exception as exc:
        print(f"[vision-pin] could not remove accelerate hooks: {exc!r}")

    vision_model.to(device)
    projector.to(device)

    if getattr(model, "hf_device_map", None):
        for key in list(model.hf_device_map):
            if key.startswith("model.vision_model") or key.startswith("model.vit_large_projector"):
                model.hf_device_map[key] = 0 if device.type == "cuda" and device.index is not None else str(device)

    bad = [(name, str(param.device)) for name, param in vision_model.named_parameters() if param.device != device]
    bad += [(name, str(buf.device)) for name, buf in vision_model.named_buffers() if buf.device != device]
    bad += [(name, str(param.device)) for name, param in projector.named_parameters() if param.device != device]
    print(f"[vision-pin] pinned vision path to {device}; off-device tensors={len(bad)}")
    for name, tensor_device in bad[:10]:
        print(f"[vision-pin]   {name} -> {tensor_device}")


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    print(f"[config] {type(config).__name__}, model_type={config.model_type}, image_token_id={config.image_token_id}")

    mapped_cls = AutoModelForCausalLM._model_mapping[type(config)]
    print(f"[auto-model] {mapped_cls.__name__}")
    if mapped_cls.__name__ != "Step3p7ForConditionalGeneration":
        raise TypeError(f"Expected Step3p7ForConditionalGeneration, got {mapped_cls!r}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    print(f"[processor] {type(processor).__name__}")
    if processor.__class__.__name__ != "Step3VLProcessor":
        raise TypeError(f"Expected Step3VLProcessor, got {type(processor)!r}")

    decoder_tokenizer = TokenizersBackend.from_pretrained(
        model_path,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    print(f"[tokenizer] {type(decoder_tokenizer).__name__}, eos={decoder_tokenizer.eos_token_id}")
    processor.tokenizer = decoder_tokenizer

    messages = build_messages(args.image_url, args.prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    summarize_inputs(inputs, processor.image_token_id)

    if args.skip_model_load:
        print("[ok] skip-model-load requested; AutoClass/processor/tokenizer smoke test passed.")
        return

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=False,
    ).eval()
    print(f"[model] loaded on device={model.device}")
    maybe_pin_vision_modules(model, args.pin_vision_device)

    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            eos_token_id=decoder_tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    decoded = decoder_tokenizer.decode(
        generate_ids[0, prompt_length:],
        skip_special_tokens=True,
    )
    print("[decoded]")
    print(decoded)

    if args.expect_cat and "cat" not in decoded.lower():
        raise AssertionError(f"Expected the response to mention cats, got: {decoded!r}")


if __name__ == "__main__":
    main()
