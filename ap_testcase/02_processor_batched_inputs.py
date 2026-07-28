"""Batched processing with arbitrary per-sample media counts.

Runs the processor on a batch of three prompts with different numbers of images and audio clips per
sample, given (a) as nested lists (one sub-list per sample, empty sub-lists allowed) and (b) as flat
lists consumed left-to-right by placeholder order. Also shows that empty media collections mean "no
media" and that count mismatches raise clear errors. Fast: does not load the model.
"""

import os

import numpy as np

from transformers import AutoProcessor


# local dir, or hub repo id (optionally `repo_id@revision`); default: the published composite.
# This script only needs the processor stack, so the weight shards are not downloaded.
CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
if not os.path.isdir(CHECKPOINT):
    from huggingface_hub import snapshot_download

    repo_id, _, revision = CHECKPOINT.partition("@")
    CHECKPOINT = snapshot_download(repo_id, revision=revision or None, ignore_patterns=["*.safetensors*"])

processor = AutoProcessor.from_pretrained(CHECKPOINT)
tokenizer = processor.tokenizer


def image(height, width):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def clip(seconds):
    return np.random.randn(int(24000 * seconds)).astype(np.float32)


texts = [
    "Transcribe this recording: <|audio|>",
    "<|image|> Describe the image.",
    "Compare <|image|> and <|image|> while listening to <|audio|> and <|audio|>.",
]

# --- (a) nested lists: explicit per-sample ownership ------------------------------------------------
images = [[], [image(256, 256)], [image(256, 256), image(320, 256)]]
audio = [[clip(1.0)], [], [clip(0.5), clip(2.0)]]
out = processor(text=texts, images=images, audio=audio, padding=True, return_tensors="pt")

assert out["pixel_values"].shape[0] == 3, "three images total, flattened over the batch"
assert out["input_features"].shape[0] == 3, "three clips total, flattened over the batch"
assert out["input_ids"].shape[0] == 3 and out["attention_mask"].shape[0] == 3
sample0 = tokenizer.decode(out["input_ids"][0], skip_special_tokens=False)
assert "<|img_start|>" not in sample0 and "<|audio_start|>" in sample0, "sample 0 has audio but no image"
counts = [tokenizer.decode(ids).count("<|audio|>") for ids in out["input_ids"]]
assert counts == [40, 0, 20 + 80], f"per-sample audio placeholders follow clip lengths, got {counts}"
print(
    f"[OK] nested batch: pixel_values {tuple(out['pixel_values'].shape)} (padded to batch max), "
    f"per-sample audio placeholder counts {counts}"
)

# --- (b) flat lists: consumed left-to-right across the batch ---------------------------------------
out_flat = processor(
    text=texts,
    images=[image(256, 256), image(256, 256), image(320, 256)],
    audio=[clip(1.0), clip(0.5), clip(2.0)],
    padding=True,
    return_tensors="pt",
)
assert out_flat["image_sizes"].tolist() == out["image_sizes"].tolist(), "same ownership as the nested form"
print(f"[OK] flat batch: image_sizes {out_flat['image_sizes'].tolist()} distributed by placeholder order")

# --- empty media collections mean 'no media' --------------------------------------------------------
out_empty = processor(text=["plain text", "more plain text"], images=[[], []], audio=[], padding=True)
assert "pixel_values" not in out_empty and "input_features" not in out_empty
print("[OK] all-empty media collections are treated as text-only")

# --- strict validation raises on count mismatches ---------------------------------------------------
for kwargs, label in [
    ({"text": "<|image|>", "images": [image(256, 256), image(256, 256)]}, "more images than placeholders"),
    ({"text": "<|image|><|image|>", "images": [image(256, 256)]}, "more placeholders than images"),
    ({"text": ["a", "<|audio|>b"], "audio": [[clip(1.0)], []]}, "nested audio on the wrong sample"),
]:
    try:
        processor(**kwargs)
        raise AssertionError(f"expected a ValueError for: {label}")
    except ValueError as error:
        print(f"[OK] rejected ({label}): {str(error)[:80]}...")

print("\nALL PROCESSOR BATCHED-INPUT CHECKS PASSED")
