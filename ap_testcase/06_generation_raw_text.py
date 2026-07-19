"""Full conditional generation from raw text (the base-model / full-control path).

Loads the full composite model and generates without any chat template:
  1) a PURE-TEXT completion from a raw prompt (base-model style continuation);
  2) a MULTIMODAL raw prompt: text containing `<|image|>` / `<|audio|>` placeholders, prepared by a
     direct `processor(...)` call;
  3) a BATCH of two raw prompts with different media (one image vs. one audio clip), left-padded, and
     cross-checked against the same prompts generated one at a time.
"""

import os

import numpy as np
import torch
from PIL import Image

from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor

# local dir, or hub repo id (optionally `repo_id@revision`); default: the published composite
CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "apertus-ai/Apertus-v1.5-8B-integration@refs/pr/2")
if not os.path.isdir(CHECKPOINT):
    from huggingface_hub import snapshot_download

    repo_id, _, revision = CHECKPOINT.partition("@")
    CHECKPOINT = snapshot_download(repo_id, revision=revision or None)

processor = AutoProcessor.from_pretrained(CHECKPOINT)
tokenizer = processor.tokenizer
print("loading model (bf16, CPU) ...")
model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).eval()


def decode_new(output, inputs):
    return [
        tokenizer.decode(sequence[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for sequence in output
    ]


image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)

# --- 1) pure text, no template ----------------------------------------------------------------------
inputs = processor(text="The capital of Switzerland is", return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
completion = decode_new(output, inputs)[0]
assert "Bern" in completion, f"expected 'Bern' in the continuation, got: {completion!r}"
print(f"[OK] raw text continuation: {completion!r}")

# --- 2) multimodal raw prompt -----------------------------------------------------------------------
# note: outside the chat format an instruct checkpoint may answer tersely or stop early; the check here
# is that generation runs on the fused image+text stream, not the wording of the continuation
inputs = processor(text="<|image|> The picture shows", images=[image], return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
assert output.shape[1] > inputs["input_ids"].shape[1], "generation must produce new tokens"
completion = decode_new(output, inputs)[0]
print(f"[OK] raw multimodal prompt: {completion!r}")

# --- 3) batched raw prompts with different media, vs. one-at-a-time ---------------------------------
texts = ["<|image|> The image shows", "<|audio|> The audio contains"]
batched = processor(text=texts, images=[[image], []], audio=[[], [audio]], padding=True, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**batched, max_new_tokens=10, do_sample=False)
batched_completions = decode_new(output, batched)

single_completions = []
for text, media in zip(texts, [{"images": [image]}, {"audio": [audio]}]):
    inputs = processor(text=text, **media, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    single_completions.append(decode_new(output, inputs)[0])

assert batched_completions == single_completions, (
    f"batched and per-sample generations must match:\n {batched_completions}\n {single_completions}"
)
print(f"[OK] batched == per-sample: {batched_completions}")

print("\nALL RAW-TEXT GENERATION CHECKS PASSED")
