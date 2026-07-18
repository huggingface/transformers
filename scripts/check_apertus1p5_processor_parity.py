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
End-to-end parity of the Apertus 1.5 processor stack against the reference vLLM implementation
(vllm_swissai, branch `apertus_integration`, `vllm/model_executor/models/apertus.py`).

Checks (PASS/FAIL):
  1. Structural token-stream parity: the processor's placeholder expansion must equal the reference
     `build_apertus_image_prompt` / `encode_audios` layout (reimplemented here verbatim as the ground truth,
     matching the golden test in vllm_swissai/tests/models/multimodal/processing/test_apertus.py).
  2. Spliced-stream parity (needs vision tokenizer weights): processor output + real VQ codes, spliced into the placeholder
     positions, must reproduce the reference prompt string byte-for-byte for the same codes.
  3. smart_resize parity: target sizes must match the reference math exactly for a sweep of input sizes.

Informational (INFO, non-failing):
  4. torchvision-vs-PIL resize drift: the image processor resizes with torchvision BICUBIC while the reference
     uses PIL BICUBIC; target sizes and token counts are identical, but resampled pixel values differ slightly,
     which can flip a small fraction of VQ codes. Measured and reported here.

Example:
    python scripts/check_apertus1p5_processor_parity.py \
        --checkpoint /path/to/Apertus-1.5-8B-composite-hf \
        --vision_tokenizer_checkpoint /path/to/apertus1p5-visionvq-hf
"""

import argparse

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor
from transformers.models.apertus1p5.image_processing_apertus1p5 import smart_resize


RESULTS = []


def report(name, passed, detail="", informational=False):
    tag = "INFO" if informational else ("PASS" if passed else "FAIL")
    RESULTS.append((name, tag))
    print(f"[{tag}] {name}: {detail}")


# --- reference implementations, kept verbatim from vllm_swissai apertus.py ---------------------------------


def reference_smart_resize(height, width, ds_factor=16, min_pixels=256 * 256, max_pixels=1400 * 1400):
    target_area = max(min(max_pixels, height * width), min_pixels)
    aspect_ratio = width / height
    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
    return new_height, new_width


def reference_image_prompt(image_tokens: torch.Tensor) -> str:
    """`ApertusImageTokenizer.build_apertus_image_prompt` (apertus.py:248-271)."""
    height, width = image_tokens.shape
    rows = ["".join(f"<|visual token {int(token_id)}|>" for token_id in row) for row in image_tokens.tolist()]
    imgstr = "<|img_end_of_row|>".join(rows)
    return f"<|img_start|>{height}*{width}<|img_token_start|>{imgstr}<|img_end|>"


def reference_audio_prompt(num_codes: int) -> str:
    """`ApertusAudioTokenizer.encode_audios` layout (apertus.py:420-466), with placeholder-code names."""
    return "<|audio_start|>" + "<placeholder>" * num_codes + "<|audio_end|>"


def reference_preprocess(pil_image: Image.Image) -> torch.Tensor:
    """The reference image preprocessing (apertus.py:288-302): PIL BICUBIC resize, /127.5 - 1, fp32 CHW."""
    pil_image = pil_image.convert("RGB")
    new_height, new_width = reference_smart_resize(pil_image.height, pil_image.width)
    resized = pil_image.resize((new_width, new_height), Image.BICUBIC)
    array = np.asarray(resized).astype(np.float64) / 127.5 - 1.0
    return torch.tensor(array, dtype=torch.float32).permute(2, 0, 1)[None]


# --- checks -------------------------------------------------------------------------------------------------


def check_smart_resize_parity():
    rng = np.random.default_rng(0)
    sizes = [(256, 256), (250, 230), (2000, 2000), (720, 1280), (33, 47), (1400, 1400), (100, 1600)]
    sizes += [tuple(rng.integers(17, 2500, 2)) for _ in range(200)]
    # extreme aspect ratios where the reference rounds one side to 0 (and would crash PIL);
    # our documented deviation floors each side at `factor` instead
    sizes += [(1, 10000), (10000, 1), (2, 4000)]
    mismatches, floor_cases = [], 0
    for height, width in sizes:
        reference = reference_smart_resize(int(height), int(width))
        ours = smart_resize(int(height), int(width))
        if min(reference) <= 0:
            floor_cases += 1
            # the floored side must be exactly `factor`; the other side must still match the reference
            expected = tuple(max(side, 16) for side in reference)
            if ours != expected:
                mismatches.append((height, width))
        elif ours != reference:
            mismatches.append((height, width))
    report(
        "smart_resize parity",
        not mismatches,
        f"{len(sizes)} sizes checked ({floor_cases} degenerate, floor engaged), {len(mismatches)} mismatches "
        f"{mismatches[:3]}",
    )


def check_structural_parity(processor):
    # image: the vLLM golden layout with code tokens replaced by the placeholder token
    for grid_height, grid_width in [(2, 2), (16, 16), (2, 4), (7, 3)]:
        codes = torch.zeros(grid_height, grid_width, dtype=torch.long)
        expected = reference_image_prompt(codes).replace("<|visual token 0|>", processor.image_token)
        actual = processor.replace_image_token({"image_grids": [[grid_height, grid_width]]}, image_idx=0)
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
    reference_pixels = reference_preprocess(image).to(device)
    with torch.no_grad():
        reference_codes = vision_tokenizer.encode(reference_pixels)[0].cpu()
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
    # the visual-token block is contiguous; derive the offset from the vocabulary itself
    return processor.tokenizer.convert_tokens_to_ids("<|visual token 0|>")


def check_backend_drift(processor, vision_tokenizer, device):
    rng = np.random.default_rng(2)
    agreements = []
    for height, width in [(300, 200), (513, 289), (1024, 768)]:
        image = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        reference_pixels = reference_preprocess(image).to(device)
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
            reference_codes = vision_tokenizer.encode(reference_pixels)
            torchvision_codes = vision_tokenizer.encode(torchvision_pixels)
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
        help="Converted Apertus1p5VisionTokenizerModel dir (enables the weight-based checks)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    check_smart_resize_parity()
    check_structural_parity(processor)

    if args.vision_tokenizer_checkpoint:
        from transformers import Apertus1p5VisionTokenizerModel

        vision_tokenizer = (
            Apertus1p5VisionTokenizerModel.from_pretrained(args.vision_tokenizer_checkpoint).to(args.device).eval()
        )
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
