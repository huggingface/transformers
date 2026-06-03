"""Minimal MiniMax M3 VL generation demo.

Round-trips an image + text prompt through the AutoProcessor and runs
``.generate()`` on the model. Uses the tiny model under
``./tiny-minimax-m3-vl`` by default (so it runs in a few seconds on CPU),
but works the same on a real bf16 checkpoint produced by
``scripts/minimax_m3_vl/dequantize_mxfp8.py``.

Usage:
    PYTHONPATH=src python minimax.py                              # default fake image
    PYTHONPATH=src python minimax.py --image /path/to/cat.jpg
    PYTHONPATH=src python minimax.py --model /raid/.../M3-bf16
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from PIL import Image

warnings.filterwarnings("ignore")


def _fake_image(size: int = 224) -> Image.Image:
    """A deterministic 224×224 RGB image so the demo runs without a real file."""
    import numpy as np

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("./tiny-minimax-m3-vl"))
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to an image file. If omitted, a deterministic random image is used.")
    parser.add_argument("--prompt", type=str,
                        default="What do you see in the image?")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"loading processor + model from {args.model.resolve()}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model).to(args.device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}, {n / 1e6:.1f}M params")

    image = Image.open(args.image).convert("RGB") if args.image else _fake_image()
    print(f"image: {image.size}")

    # The MiniMax processor uses literal markers (`]<]image[>[`) that the
    # tokenizer expands into ``vision_start + N * placeholder + vision_end``
    # after seeing how many merged patches the image produces.
    text = processor.IMAGE_TOKEN + "\n" + args.prompt
    inputs = processor(images=[image], text=[text], return_tensors="pt").to(args.device)
    print("input_ids:", tuple(inputs["input_ids"].shape),
          "pixel_values:", tuple(inputs["pixel_values"].shape),
          "image_grid_thw:", inputs["image_grid_thw"].tolist())

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    n_prompt = inputs["input_ids"].shape[1]
    completion = processor.tokenizer.decode(out_ids[0, n_prompt:], skip_special_tokens=True)
    print("\n=== generation ===")
    print(completion)


if __name__ == "__main__":
    main()
