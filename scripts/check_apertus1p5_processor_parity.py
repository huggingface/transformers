# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
Checks Apertus 1.5 resize geometry against the training pipeline and token layout against reference vLLM.
Resize reference: swiss-ai/benchmark-image-tokenzier at f32cc20e5aeeb22460383ced277b534a89c69050,
`vision_tokenization/utils/image_geometry.py`. Token layout reference: vllm_swissai,
`vllm/model_executor/models/apertus.py` at `apertus_integration` commit 40a5516b7.

Note on the reference: newer vllm_swissai branches (`apertus-1-5`, design v2) deleted this independent
implementation and instead consume the transformers processor directly via `get_hf_processor()` in
`apertus_mm.py`. The layout helpers below preserve that independent token-layout reference.

Checks (PASS/FAIL), in run order:
  1. smart_resize parity: target sizes must match the training pipeline for fixed and random sizes in
     the shared supported range. The reused Qwen helper additionally rejects aspect ratios above 200
     and floors shrinking dimensions at one spatial factor; these are not training-parity cases.
  2. Structural token-stream parity: the processor's placeholder expansion must equal the reference
     `build_apertus_image_prompt` / `encode_audios` layout (reimplemented here verbatim as the ground truth,
     matching the golden test in vllm_swissai/tests/models/multimodal/processing/test_apertus.py, which also
     exists only at the pinned commit).
  3. Spliced-stream parity (needs `--vision_tokenizer_checkpoint`): processor output + real VQ codes, spliced
     into the placeholder positions, must reproduce the reference prompt string byte-for-byte for the same codes.

Informational (INFO, non-failing):
  4. torchvision-vs-PIL resize drift (needs `--vision_tokenizer_checkpoint`): the image processor resizes
     with torchvision BICUBIC while the reference uses PIL BICUBIC; target sizes and token counts are
     identical, but resampled pixel values differ slightly, which can flip a small fraction of VQ codes.
     Measured and reported here.

Without `--vision_tokenizer_checkpoint`, the weight-based checks 3 and 4 are skipped. Any FAIL makes the
script exit non-zero. Pass the original EMU3.5 source (local directory or Hub `repo_id[@revision]`);
the shared converter loads and validates its vision encoder without writing a standalone checkpoint.

Example:
    python scripts/check_apertus1p5_processor_parity.py \
        --checkpoint /path/to/Apertus-1.5-8B-composite-hf \
        --vision_tokenizer_checkpoint BAAI/Emu3.5-VisionTokenizer
"""

import argparse
import math

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor
from transformers.models.apertus1p5.image_processing_apertus1p5 import smart_resize


RESULTS = []


def report(name, passed, detail="", informational=False):
    """Print one [PASS]/[FAIL]/[INFO] line and record the tag for the final summary."""
    tag = "INFO" if informational else ("PASS" if passed else "FAIL")
    RESULTS.append((name, tag))
    print(f"[{tag}] {name}: {detail}")


# --- independent reference implementations ---


def reference_smart_resize(height, width, ds_factor=16, min_pixels=128 * 128, max_pixels=1400 * 1400):
    """Training pipeline's `smart_resize_dims`, with the configured Apertus area limits."""
    if height < ds_factor or width < ds_factor:
        raise ValueError("height/width must be at least the spatial factor")
    resized_height = round(height / ds_factor) * ds_factor
    resized_width = round(width / ds_factor) * ds_factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / scale / ds_factor) * ds_factor
        resized_width = math.floor(width / scale / ds_factor) * ds_factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / ds_factor) * ds_factor
        resized_width = math.ceil(width * scale / ds_factor) * ds_factor
    return resized_height, resized_width


def reference_image_prompt(image_tokens: torch.Tensor) -> str:
    """`ApertusImageTokenizer.build_apertus_image_prompt` (apertus.py:248-271)."""
    height, width = image_tokens.shape
    rows = ["".join(f"<|visual token {int(token_id)}|>" for token_id in row) for row in image_tokens.tolist()]
    imgstr = "<|img_end_of_row|>".join(rows)
    return f"<|img_start|>{height}*{width}<|img_token_start|>{imgstr}<|img_end|>"


def reference_audio_prompt(num_codes: int) -> str:
    """`ApertusAudioTokenizer.encode_audios` layout (apertus.py:420-466), with placeholder-code names."""
    return "<|audio_start|>" + "<placeholder>" * num_codes + "<|audio_end|>"


def reference_preprocess(pil_image: Image.Image, *, min_pixels: int, max_pixels: int) -> torch.Tensor:
    """Training preprocessing: RGB, PIL BICUBIC resize, then float32 normalization to [-1, 1]."""
    pil_image = pil_image.convert("RGB")
    new_height, new_width = reference_smart_resize(
        pil_image.height, pil_image.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    resized = pil_image.resize((new_width, new_height), Image.BICUBIC)
    pixels = torch.tensor(np.array(resized), dtype=torch.float32).permute(2, 0, 1)
    return pixels.div_(127.5).sub_(1.0)[None]


# --- checks -------------------------------------------------------------------------------------------------


def check_smart_resize_parity():
    """Check training resize parity across alignment ties, budget boundaries, and 200 random sizes."""
    rng = np.random.default_rng(0)
    sizes = [(128, 128), (264, 264), (250, 230), (2000, 2000), (720, 1280), (33, 47), (1400, 1400), (100, 1600)]
    sizes += [tuple(rng.integers(17, 2500, 2)) for _ in range(200)]
    mismatches = []
    for height, width in sizes:
        reference = reference_smart_resize(int(height), int(width))
        ours = smart_resize(int(height), int(width), factor=16, min_pixels=128 * 128, max_pixels=1400 * 1400)
        if ours != reference:
            mismatches.append((height, width))
    report(
        "smart_resize parity",
        not mismatches,
        f"{len(sizes)} sizes checked, {len(mismatches)} mismatches {mismatches[:3]}",
    )


def check_structural_parity(processor):
    """The processor's placeholder expansions must equal the reference prompt builders.

    Purely structural, no weights involved: compares `replace_image_token` / `replace_audio_token` output
    for several grid shapes and code counts against the reference builders, with every code token replaced
    by the generic placeholder.
    """
    # image: the vLLM golden layout with code tokens replaced by the placeholder token
    for grid_height, grid_width in [(2, 2), (16, 16), (2, 4), (7, 3)]:
        codes = torch.zeros(grid_height, grid_width, dtype=torch.long)
        expected = reference_image_prompt(codes).replace("<|visual token 0|>", processor.image_token)
        factor = processor.image_processor.spatial_factor
        actual = processor.replace_image_token(
            {"image_sizes": [[grid_height * factor, grid_width * factor]]}, image_idx=0
        )
        if actual != expected:
            report("image structural parity", False, f"grid {grid_height}x{grid_width}:\n {actual}\n {expected}")
            return
    report("image structural parity", True, "grids 2x2, 16x16, 2x4, 7x3 equal the reference builder")

    for num_codes in [1, 2, 40]:
        expected = reference_audio_prompt(num_codes).replace("<placeholder>", processor.audio_token)
        actual = processor.replace_audio_token({"num_audio_codes": [num_codes]}, audio_idx=0)
        if actual != expected:
            report("audio structural parity", False, f"{num_codes} codes:\n {actual}\n {expected}")
            return
    report("audio structural parity", True, "1/2/40 codes equal the reference layout")


def check_spliced_stream_parity(processor, vision_tokenizer, device):
    """Processor placeholders + real VQ codes spliced in must reproduce the reference prompt byte-for-byte."""
    tokenizer = processor.tokenizer
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, (300, 200, 3), dtype=np.uint8))

    # reference: PIL preprocessing -> VQ codes -> reference prompt string
    reference_pixels = reference_preprocess(image, **dict(processor.image_processor.size)).to(device)
    with torch.no_grad():
        reference_codes = vision_tokenizer.encode(reference_pixels).image_tokens[0].cpu()
    reference = reference_image_prompt(reference_codes)

    # transformers: processor expansion + the SAME codes spliced into the placeholder positions
    inputs = processor(text=processor.image_token, images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    image_token_id = processor.image_token_id
    offset_ids = reference_codes.flatten() + int(processor_image_offset(processor))
    spliced = input_ids.clone()
    spliced[input_ids == image_token_id] = offset_ids
    # strip everything the reference does not include (bos)
    spliced = spliced[1:] if spliced[0].item() == tokenizer.bos_token_id else spliced
    actual = tokenizer.decode(spliced)

    report(
        "spliced image stream parity",
        actual == reference,
        f"{len(offset_ids)} codes, byte-equal: {actual == reference}",
    )


def processor_image_offset(processor):
    """Vocabulary id of `<|visual token 0|>`: the start of the contiguous visual-token block."""
    return processor.tokenizer.convert_tokens_to_ids("<|visual token 0|>")


def check_backend_drift(processor, vision_tokenizer, device):
    """Measure VQ-code agreement between PIL (reference) and torchvision (ours) preprocessing.

    Informational by design: the resize backends differ in the last bits of the resampled pixels, which
    flips a small fraction of near-tied codes. Target sizes and token counts must still be identical
    (a shape mismatch is the only FAIL); the per-image agreement ratio is reported, not asserted.
    """
    rng = np.random.default_rng(2)
    agreements = []
    for height, width in [(300, 200), (513, 289), (1024, 768)]:
        image = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        reference_pixels = reference_preprocess(image, **dict(processor.image_processor.size)).to(device)
        inputs = processor(text=processor.image_token, images=[image], return_tensors="pt")
        torchvision_pixels = inputs["pixel_values"].to(device)
        if reference_pixels.shape != torchvision_pixels.shape:
            report(
                "torchvision-vs-PIL drift",
                False,
                f"shape mismatch {reference_pixels.shape} vs {torchvision_pixels.shape}",
            )
            return
        with torch.no_grad():
            reference_codes = vision_tokenizer.encode(reference_pixels).image_tokens
            torchvision_codes = vision_tokenizer.encode(torchvision_pixels).image_tokens
        agreements.append((reference_codes == torchvision_codes).float().mean().item())
    report(
        "torchvision-vs-PIL resize code drift",
        True,
        f"code agreement per image: {[f'{a:.1%}' for a in agreements]} "
        "(sizes/counts identical; use PIL preprocessing for byte-exact reference codes)",
        informational=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Composite checkpoint dir (processor + config)")
    parser.add_argument(
        "--vision_tokenizer_checkpoint",
        default=None,
        help="Original EMU3.5 checkpoint directory or Hub repo (enables the weight-based checks)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    check_smart_resize_parity()
    check_structural_parity(processor)

    if args.vision_tokenizer_checkpoint:
        from transformers.models.apertus1p5.convert_apertus1p5_weights_to_hf import (
            load_vision_tokenizer,
            resolve_checkpoint_dir,
        )
        from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME

        source = resolve_checkpoint_dir(
            args.vision_tokenizer_checkpoint, allow_patterns=[CONFIG_NAME, SAFE_WEIGHTS_NAME]
        )
        vision_tokenizer = load_vision_tokenizer(source).to(args.device)
        check_spliced_stream_parity(processor, vision_tokenizer, args.device)
        check_backend_drift(processor, vision_tokenizer, args.device)
    else:
        print("[SKIP] weight-based checks (pass --vision_tokenizer_checkpoint to enable)")

    passed = sum(tag == "PASS" for _, tag in RESULTS)
    failed = sum(tag == "FAIL" for _, tag in RESULTS)
    info = sum(tag == "INFO" for _, tag in RESULTS)
    print(f"\nPARITY SUMMARY: {passed} passed, {failed} failed, {info} informational")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
