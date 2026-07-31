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
End-to-end parity check between the transformers `Apertus1p5VisionTokenizerModel` (encode-only IBQ vision tokenizer) and the
ORIGINAL EMU3.5 Vision Tokenizer implementation (`BAAI/Emu3.5-VisionTokenizer` remote code + weights).

This is the one check that cannot ship, because it needs the original repo's remote code as a reference.
Everything verifiable without that reference lives in the shipped converter's `--verify`
(`src/transformers/models/apertus1p5/convert_apertus1p5_vision_tokenizer_to_hf.py`): stored precision, the
derived configuration, code-grid geometry, batched encoding, the save/reload round trip, and the float32 keep
under a bfloat16 load. This script reuses that converter's conversion functions so the two cannot drift apart.

What it verifies (per test image: gradients, checkerboards, noise, solid, mixed; several sizes):
  1. convert   - the original state dict loads into the port with `strict=True`, using the shipped converter's
                 config mapping and key filter (drop `decoder.*` / `post_quant_conv.*`, keep 247 tensors 1:1);
                 a precondition of the comparison below, and here run against the live remote-code module
  2. encode    - our code grids are BIT-EXACT equal to the original `encode()[2][2]` codes, for every image
  3. dtype     - two informational half-precision measurements against our fp32 baseline. They stay here
                 rather than in the shipped converter because they are agreement percentages, not pass/fail
                 checks, and they are the source of the "~10% of codes flip in bf16" figure quoted in
                 `modeling_apertus1p5.py` and in the model documentation:
                 * input bf16 -> model fp32: input quantization loss only
                 * model force-cast `.to(bf16)` (the `_keep_in_fp32_modules_strict` guard bypassed, torch
                   semantics): true bf16 compute; input dtype is irrelevant (cast internally)

How the ORIGINAL inference works, dtype-wise: strictly fp32. The checkpoint ships `torch_dtype: float32`, the
BAAI remote code does no dtype handling (it runs at whatever dtype the weights are, fp32 by default), and the
vLLM integration keeps the tokenizer fp32 in both generations: the original implementation hardcoded
`vision_dtype = torch.float32` for weights and inputs (`apertus.py:221,301` at `apertus_integration`
commit 40a5516b7), and the current design-v2 loads it under `set_default_torch_dtype(torch.float32)`
(`apertus_mm.py:482`). There is no half-precision reference: fp32/fp32 is the only cell with a ground truth;
all other cells are measured against our fp32/fp32 baseline.

Inputs are preprocessed exactly like the vLLM/Apertus pipeline feeds the encoder: RGB, float32, `x/127.5 - 1`.
(The resize policy, smart_resize to multiples of 16 within [256^2, 1400^2], is processor scope and not
exercised here; sizes are chosen directly.)

Requirements: downloads the original weights (~1.8 GB, float32) and executes the original repo's remote code
(`trust_remote_code=True`), the same code this port was reviewed against.

This script does not write a checkpoint. To produce one, use the shipped converter:
`python src/transformers/models/apertus1p5/convert_apertus1p5_vision_tokenizer_to_hf.py --checkpoint_path
BAAI/Emu3.5-VisionTokenizer --output_dir <dir> --verify`.

Example:
    python scripts/check_apertus1p5_vision_tokenizer_parity.py \
        [--original BAAI/Emu3.5-VisionTokenizer]   # or a local snapshot dir \
        [--device cpu]
"""

import argparse
import sys

import numpy as np
import torch


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
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps (fp32 either way)")
    args = parser.parse_args()

    device = torch.device(args.device)
    results: list[tuple[str, str, str]] = []

    def record(status: str, name: str, detail: str = ""):
        results.append((status, name, detail))
        print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))

    from transformers import AutoModel
    from transformers.models.apertus1p5.convert_apertus1p5_vision_tokenizer_to_hf import (
        convert_config,
        convert_state_dict,
    )
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VisionTokenizerModel

    print(f"Loading ORIGINAL model from {args.original} (trust_remote_code, ~1.8 GB fp32) ...")
    original = AutoModel.from_pretrained(args.original, trust_remote_code=True).to(device).eval()

    # ---- 1. convert: original state dict -> port, strict --------------------------------------------------
    # the mapping is fed the config object the ORIGINAL remote code built, so a semantic drift between their
    # config class and the shipped converter's field table surfaces here
    port = Apertus1p5VisionTokenizerModel(convert_config(original.config.to_dict()))
    converted = convert_state_dict(original.state_dict())
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

    # ---- 2. encode parity against the original (bit-exact) ------------------------------------------------
    # the code COUNT is compared against the original's own output rather than a (H // 16) * (W // 16) formula,
    # which makes this a check against ground truth instead of against our own arithmetic
    for name, image in images.items():
        pixel_values = preprocess(image, device)
        with torch.no_grad():
            ours = port.encode(pixel_values)  # (1, H//16, W//16)
        theirs = original_encode(original, pixel_values)  # flat
        our_codes[name] = ours

        if ours.numel() != theirs.numel():
            record(FAIL, f"encode[{name}]", f"count ours {ours.numel()} vs original {theirs.numel()}")
        elif not torch.equal(ours.flatten().long().cpu(), theirs.flatten().long().cpu()):
            n_diff = (ours.flatten().long().cpu() != theirs.flatten().long().cpu()).sum().item()
            record(FAIL, f"encode[{name}]", f"{n_diff}/{ours.numel()} codes differ")
        else:
            record(PASS, f"encode[{name}]", f"{ours.shape[1]}x{ours.shape[2]} codes bit-exact")

    # ---- 3. dtype measurements (INFO only; see the module docstring) ---------------------------------------
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

    # (b) model force-cast to bf16 (`.to(torch.bfloat16)` bypasses `_keep_in_fp32_modules_strict`, torch
    #     semantics): true bf16 compute; input dtype is irrelevant (cast to model dtype internally). The cast
    #     is in place and lossy, so it runs last: `port` is no longer an fp32 reference afterwards. The
    #     complementary guard check, that `from_pretrained(dtype=bf16)` KEEPS fp32, is pass/fail and lives in
    #     the shipped converter's `--verify`.
    port = port.to(torch.bfloat16)
    with torch.no_grad():
        codes = port.encode(pixels_bf16)
    agreement = (codes == baseline).float().mean().item()
    record(
        INFO,
        "dtype[any input -> model force-cast .to(bf16)]",
        f"{agreement:.1%} agreement; run the tokenizer in fp32",
    )

    _summarize(results)
    sys.exit(1 if any(status == FAIL for status, _, _ in results) else 0)


def _summarize(results):
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 80)
    print(f"PARITY SUMMARY: {n_pass} passed, {n_fail} failed, {len(results) - n_pass - n_fail} informational")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name} - {detail}")
    print("=" * 80)


if __name__ == "__main__":
    main()
