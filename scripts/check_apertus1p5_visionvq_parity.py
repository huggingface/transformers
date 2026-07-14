# Copyright 2026 The Emu team, BAAI, The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
End-to-end parity check between the transformers `Apertus1p5VQVAE` (encode-only IBQ vision tokenizer) and the
ORIGINAL EMU3.5 Vision Tokenizer implementation (`BAAI/Emu3.5-VisionTokenizer` remote code + weights).

What it verifies (per test image: gradients, checkerboards, noise, solid, mixed; several sizes):
  1. convert   - the original state dict loads into the port with `strict=True`
                 (drop `decoder.*` / `post_quant_conv.*`, keep the remaining 247 tensors 1:1)
  2. encode    - our code grids are BIT-EXACT equal to the original `encode()[2][2]` codes
  3. grid      - code count equals (H // 16) * (W // 16) for every size, incl. non-multiples of 16
  4. batch     - equal-size batched encode is bit-exact vs per-image encode (ours)
  5. reload    - save_pretrained/from_pretrained round trip is bit-exact
  6. dtype     - full {fp32, bf16} input x {fp32, bf16-loaded, bf16-forced} model matrix:
                 * input fp32 -> model fp32: the main suite (bit-exact vs the original)
                 * input bf16 -> model fp32: agreement vs baseline (input quantization loss only; INFO)
                 * input fp32 -> model from_pretrained(dtype=bf16): `_keep_in_fp32_modules_strict` keeps the
                   tokenizer fp32 -> codes must be BIT-EXACT (PASS/FAIL — validates the guard on real weights)
                 * input bf16 -> model from_pretrained(dtype=bf16): agreement (INFO)
                 * model force-cast `.to(bf16)` (guard bypassed): true bf16 compute, input dtype irrelevant
                   (cast to model dtype internally); agreement (INFO)

How the ORIGINAL inference works, dtype-wise: strictly fp32. The checkpoint ships `torch_dtype: float32`, the
BAAI remote code does no dtype handling (it runs at whatever dtype the weights are, fp32 by default), and the
vLLM integration hardcodes `vision_dtype = torch.float32` for both the tokenizer weights and the input tensor
(`apertus.py:221,301`). There is no half-precision reference — fp32/fp32 is the only cell with a ground truth;
all other cells are measured against our fp32/fp32 baseline.

Inputs are preprocessed exactly like the vLLM/Apertus pipeline feeds the encoder: RGB, float32, `x/127.5 - 1`.
(The resize policy — smart_resize to multiples of 16 within [256^2, 1400^2] — is processor scope and not
exercised here; sizes are chosen directly.)

Requirements: downloads the original weights (~1.8 GB, float32) and executes the original repo's remote code
(`trust_remote_code=True`) — the same code this port was reviewed against.

Example:
    python scripts/check_apertus1p5_visionvq_parity.py \
        [--original BAAI/Emu3.5-VisionTokenizer]   # or a local snapshot dir \
        [--save_converted /path/to/apertus1p5-visionvq-hf] \
        [--device cpu]
"""

import argparse
import sys

import numpy as np
import torch


DOWNSAMPLE = 16

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


def make_test_images() -> dict[str, np.ndarray]:
    """Deterministic RGB test images (H, W, 3) uint8, various sizes incl. a non-multiple of 16."""
    rng = np.random.default_rng(seed=0)

    def gradient(h, w):
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
        y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
        return np.broadcast_to((x + y) / 2, (h, w, 3)).astype(np.uint8)

    def checkerboard(h, w, cell=32):
        yy, xx = np.mgrid[0:h, 0:w]
        board = (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
        return np.stack([board, 255 - board, board], axis=-1)

    def noise(h, w):
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def solid(h, w, value=127):
        return np.full((h, w, 3), value, dtype=np.uint8)

    def mixed(h, w):
        img = gradient(h, w).astype(np.int32)
        img[: h // 2, : w // 2] = checkerboard(h // 2, w // 2, cell=16)[..., :3]
        img[h // 2 :, w // 2 :] = rng.integers(0, 256, size=(h - h // 2, w - w // 2, 3))
        return np.clip(img, 0, 255).astype(np.uint8)

    return {
        "gradient_256x256": gradient(256, 256),
        "checkerboard_256x384": checkerboard(256, 384),
        "noise_384x256": noise(384, 256),
        "solid_256x256": solid(256, 256),
        "mixed_512x320": mixed(512, 320),
        "noise_250x230_nonmult16": noise(250, 230),
        "noise_192x192": noise(192, 192),  # small image reused by the dtype matrix (bf16 conv is slow on CPU)
    }


def preprocess(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """vLLM/Apertus pixel preprocessing: RGB float32, x/127.5 - 1, (1, 3, H, W)."""
    tensor = torch.tensor(image.astype(np.float32) / 127.5 - 1.0, device=device)
    return tensor.permute(2, 0, 1)[None]


def original_encode(original, pixel_values: torch.Tensor) -> torch.Tensor:
    """Original codes for a (1, 3, H, W) input -> flat int64 (the original returns ind.flatten())."""
    with torch.no_grad():
        encode_out = original.encode(pixel_values)
    return encode_out[2][2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--original",
        default="BAAI/Emu3.5-VisionTokenizer",
        help="Hub repo id or local snapshot dir of the original tokenizer (remote code + weights)",
    )
    parser.add_argument("--save_converted", default=None, help="Optionally save the converted port here")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps (fp32 either way)")
    args = parser.parse_args()

    device = torch.device(args.device)
    results: list[tuple[str, str, str]] = []

    def record(status: str, name: str, detail: str = ""):
        results.append((status, name, detail))
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    from transformers import AutoModel
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VQVAE, Apertus1p5VQVAEConfig

    print(f"Loading ORIGINAL model from {args.original} (trust_remote_code, ~1.8 GB fp32) ...")
    original = AutoModel.from_pretrained(args.original, trust_remote_code=True).to(device).eval()

    # ---- 1. convert: original state dict -> port, strict --------------------------------------------------
    port = Apertus1p5VQVAE(Apertus1p5VQVAEConfig())
    converted = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith(("decoder.", "post_quant_conv."))
    }
    try:
        port.load_state_dict(converted, strict=True)
        record(PASS, "convert: strict load of original weights", f"{len(converted)} tensors")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "convert", f"{type(exc).__name__}: {exc}")
        _summarize(results)
        sys.exit(1)
    port = port.to(device).eval()

    images = make_test_images()
    our_codes: dict[str, torch.Tensor] = {}

    # ---- 2 + 3. encode parity (bit-exact) and grid geometry -----------------------------------------------
    for name, image in images.items():
        pixel_values = preprocess(image, device)
        height, width = image.shape[:2]
        with torch.no_grad():
            ours = port.encode(pixel_values)  # (1, H//16, W//16)
        theirs = original_encode(original, pixel_values)  # flat
        our_codes[name] = ours

        expected_grid = (height // DOWNSAMPLE, width // DOWNSAMPLE)
        if ours.shape[1:] != torch.Size(expected_grid):
            record(FAIL, f"grid[{name}]", f"got {tuple(ours.shape[1:])}, expected {expected_grid}")
            continue
        if ours.numel() != theirs.numel():
            record(FAIL, f"encode[{name}]", f"count ours {ours.numel()} vs original {theirs.numel()}")
        elif not torch.equal(ours.flatten().long().cpu(), theirs.flatten().long().cpu()):
            n_diff = (ours.flatten().long().cpu() != theirs.flatten().long().cpu()).sum().item()
            record(FAIL, f"encode[{name}]", f"{n_diff}/{ours.numel()} codes differ")
        else:
            record(PASS, f"encode[{name}]", f"{expected_grid[0]}x{expected_grid[1]} codes bit-exact")

    # ---- 4. batched vs per-image (ours; same size, real weights) ------------------------------------------
    batch = torch.cat(
        [preprocess(images["gradient_256x256"], device), preprocess(images["solid_256x256"], device)], dim=0
    )
    with torch.no_grad():
        batched = port.encode(batch)
        singles = torch.cat([port.encode(batch[i : i + 1]) for i in range(2)], dim=0)
    record(PASS if torch.equal(batched, singles) else FAIL, "batch: same-size batched == per-image encode")

    # ---- 5. save/load round trip ----------------------------------------------------------------------------
    import shutil
    import tempfile

    target = args.save_converted or tempfile.mkdtemp()
    port.save_pretrained(target)
    reloaded = Apertus1p5VQVAE.from_pretrained(target).to(device).eval()
    with torch.no_grad():
        ok = torch.equal(reloaded.encode(preprocess(images["mixed_512x320"], device)), our_codes["mixed_512x320"])
    record(PASS if ok else FAIL, "reload: save_pretrained/from_pretrained round trip")
    if args.save_converted:
        print(f"Converted model saved to {args.save_converted}")

    # ---- 6. dtype matrix: {fp32, bf16} inputs x {fp32, bf16-loaded, bf16-forced} model -----------------------
    # The ORIGINAL stack is fp32-only (checkpoint torch_dtype=float32; the BAAI remote code has no dtype
    # handling; vLLM hardcodes vision_dtype=torch.float32 for both weights and inputs), so fp32/fp32 is the only
    # cell with an original reference (checked bit-exact above). The remaining cells are measured against our
    # fp32/fp32 baseline. Note: with a bf16 WEIGHT model, fp32 and bf16 inputs are equivalent (encode() casts
    # the input to the model dtype), so forced-bf16 needs a single measurement.
    matrix_name = "noise_192x192"
    baseline = our_codes[matrix_name]
    pixels_fp32 = preprocess(images[matrix_name], device)
    pixels_bf16 = pixels_fp32.to(torch.bfloat16)

    # (a) input bf16 -> model fp32: encode() upcasts the (lossily bf16-rounded) input to fp32
    with torch.no_grad():
        codes = port.encode(pixels_bf16)
    agreement = (codes == baseline).float().mean().item()
    record(INFO, "dtype[input bf16 -> model fp32]", f"{agreement:.1%} agreement (input quantization loss only)")

    # (b) input fp32 -> model loaded with dtype=bf16: _keep_in_fp32_modules_strict must keep the tokenizer
    #     weights in fp32, making codes BIT-EXACT vs the fp32 baseline (the guard, verified with real weights)
    guarded = Apertus1p5VQVAE.from_pretrained(target, dtype=torch.bfloat16).to(device).eval()
    weights_fp32 = guarded.encoder.conv_in.weight.dtype == torch.float32
    with torch.no_grad():
        codes = guarded.encode(pixels_fp32)
    ok = weights_fp32 and torch.equal(codes, baseline)
    record(
        PASS if ok else FAIL,
        "dtype[input fp32 -> model from_pretrained(dtype=bf16)]",
        "weights kept fp32 by _keep_in_fp32_modules_strict, codes bit-exact"
        if ok
        else f"weights fp32: {weights_fp32}, codes equal: {torch.equal(codes, baseline)}",
    )

    # (c) input bf16 -> model loaded with dtype=bf16: guard keeps weights fp32; only the input rounding remains
    with torch.no_grad():
        codes = guarded.encode(pixels_bf16)
    agreement = (codes == baseline).float().mean().item()
    record(INFO, "dtype[input bf16 -> model from_pretrained(dtype=bf16)]", f"{agreement:.1%} agreement")
    del guarded

    # (d) model force-cast to bf16 (`.to(torch.bfloat16)` bypasses the guard — torch semantics): true bf16
    #     compute; input dtype is irrelevant (cast to model dtype internally)
    forced = Apertus1p5VQVAE.from_pretrained(target).to(device).to(torch.bfloat16).eval()
    with torch.no_grad():
        codes = forced.encode(pixels_bf16)
    agreement = (codes == baseline).float().mean().item()
    record(
        INFO,
        "dtype[any input -> model force-cast .to(bf16)]",
        f"{agreement:.1%} agreement — run the tokenizer in fp32",
    )
    del forced

    if not args.save_converted:
        shutil.rmtree(target, ignore_errors=True)

    _summarize(results)
    sys.exit(1 if any(status == FAIL for status, _, _ in results) else 0)


def _summarize(results):
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 80)
    print(f"PARITY SUMMARY: {n_pass} passed, {n_fail} failed, {len(results) - n_pass - n_fail} informational")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name} — {detail}")
    print("=" * 80)


if __name__ == "__main__":
    main()
