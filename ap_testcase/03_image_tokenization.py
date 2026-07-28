"""Image tokenization: processor-driven vs. fully manual.

Loads the full composite model (CPU, bf16 backbone with the fp32-guarded vision tokenizer) and turns
one image into discrete codes two ways:
  1) from PROCESSOR OUTPUT: `processor(...)` -> `model.model.get_image_tokens(pixel_values, image_sizes)`
     (vocabulary ids, offset into the `<|visual token i|>` range), verifying the count matches the
     number of `<|image|>` placeholders the processor emitted;
  2) fully MANUAL: PIL resize to a multiple of 16 + `/127.5 - 1` normalization ->
     `model.model.vision_tokenizer.encode(...)` (raw codebook indices) -> manual offset into vocabulary ids.
Both paths must yield the same grid geometry; code values may differ slightly because the manual path
uses PIL's bicubic kernel while the processor uses torchvision's (agreement is reported).
"""

import os

import numpy as np
import torch
from PIL import Image

from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor


# local dir, or hub repo id (optionally `repo_id@revision`); default: the published composite
CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
if not os.path.isdir(CHECKPOINT):
    from huggingface_hub import snapshot_download

    repo_id, _, revision = CHECKPOINT.partition("@")
    CHECKPOINT = snapshot_download(repo_id, revision=revision or None)

processor = AutoProcessor.from_pretrained(CHECKPOINT)
tokenizer = processor.tokenizer
print("loading model (bf16, CPU) ...")
model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).eval()
config = model.config

pil_image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8))

# --- 1) processor output -> model tokenizer --------------------------------------------------------
inputs = processor(text="<|image|>", images=[pil_image], return_tensors="pt")
with torch.no_grad():
    vocab_ids = model.model.get_image_tokens(inputs["pixel_values"], inputs["image_sizes"])

num_placeholders = tokenizer.decode(inputs["input_ids"][0]).count("<|image|>")
assert vocab_ids.numel() == num_placeholders, "one code per placeholder"
assert int(vocab_ids.min()) >= config.image_token_offset
assert int(vocab_ids.max()) < config.image_token_offset + config.vision_tokenizer_config.codebook_size
first = int(vocab_ids[0])
assert tokenizer.convert_ids_to_tokens(first) == f"<|visual token {first - config.image_token_offset}|>"
print(
    f"[OK] processor path: {vocab_ids.numel()} codes, first id {first} -> {tokenizer.convert_ids_to_tokens(first)!r}"
)

# --- 2) fully manual preprocessing -> vision tokenizer ---------------------------------------------
# resize to the same target the processor chose (multiples of 16), PIL BICUBIC like the reference pipeline
target_h, target_w = (int(side) for side in inputs["image_sizes"][0])
resized = pil_image.convert("RGB").resize((target_w, target_h), Image.BICUBIC)
pixels = torch.tensor(np.asarray(resized) / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)[None]

with torch.no_grad():
    code_grid = model.model.vision_tokenizer.encode(pixels)[0]  # (H/16, W/16) raw codebook indices

assert code_grid.shape == (target_h // 16, target_w // 16)
manual_vocab_ids = code_grid.flatten() + config.image_token_offset
agreement = (manual_vocab_ids == vocab_ids).float().mean().item()
print(
    f"[OK] manual path: grid {tuple(code_grid.shape)}, same geometry; "
    f"code agreement vs processor path {agreement:.1%} (PIL vs torchvision resize kernels)"
)
assert agreement > 0.9, "the two resize kernels should only flip a small fraction of codes"

print("\nALL IMAGE TOKENIZATION CHECKS PASSED")
