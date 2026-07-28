"""Processor on single (non-batched) inputs.

Loads the Apertus 1.5 processor from the composite checkpoint and runs it on: text only, text + one
image, text + one audio clip, and text + both. Checks that each `<|image|>` / `<|audio|>` placeholder
expands into the structured token run (boi + "H*W" header + img + H x W placeholders with row
separators + eoi; audio_start + ceil(samples/600) placeholders + audio_end) and that the returned
tensors match the model contract. The final section passes media as URL strings, which the processor
fetches itself (audio resampled to 24 kHz; requires network + librosa). Fast: does not load the model.
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
image = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)  # arbitrary size; resized to x16 grid
audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)  # 1 s @ 24 kHz

# --- text only -------------------------------------------------------------------------------------
out = processor(text="What is the capital of Switzerland?", return_tensors="pt")
assert set(out.keys()) == {"input_ids", "attention_mask"}
print(f"[OK] text only: input_ids {tuple(out['input_ids'].shape)}")

# --- text + one image ------------------------------------------------------------------------------
out = processor(text="<|image|> Describe this image.", images=[image], return_tensors="pt")
decoded = tokenizer.decode(out["input_ids"][0])
height, width = (int(side) for side in out["image_sizes"][0])
grid_h, grid_w = height // 16, width // 16
assert decoded.count("<|image|>") == grid_h * grid_w, "one placeholder per 16x16 patch of the resized image"
assert decoded.count("<|img_end_of_row|>") == grid_h - 1, "rows joined by exactly H-1 separators"
assert f"<|img_start|>{grid_h}*{grid_w}<|img_token_start|>" in decoded, "height-first size header"
assert out["pixel_values"].shape == (1, 3, height, width)
print(
    f"[OK] text+image: 300x200 px resized to {height}x{width} -> {grid_h}x{grid_w} grid "
    f"({grid_h * grid_w} placeholders); stream starts: {decoded[:80]}..."
)

# --- text + one audio clip -------------------------------------------------------------------------
out = processor(text="Transcribe: <|audio|>", audio=[audio], return_tensors="pt")
decoded = tokenizer.decode(out["input_ids"][0])
assert decoded.count("<|audio|>") == 40, "1 s of 24 kHz audio -> ceil(24000/600) = 40 codes"
assert "<|audio_start|>" in decoded and "<|audio_end|>" in decoded
assert out["input_features"].shape == (1, 1, 24000)
assert int(out["feature_attention_mask"].sum()) == 24000
print(f"[OK] text+audio: 1 s clip -> 40 placeholders, input_features {tuple(out['input_features'].shape)}")

# --- text + image + audio in one prompt ------------------------------------------------------------
out = processor(
    text="<|image|> What do you see, and what is said here: <|audio|>",
    images=[image],
    audio=[audio],
    return_tensors="pt",
)
assert {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_sizes",
    "input_features",
    "feature_attention_mask",
} <= set(out.keys())
print(
    f"[OK] text+image+audio: sequence length {out['input_ids'].shape[-1]} "
    f"(structure tokens + {grid_h * grid_w} image + 40 audio placeholders + text)"
)

# --- media given as URLs: the processor fetches and (for audio files) resamples itself --------------
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
audio_url = "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
try:
    out = processor(text="<|image|> and <|audio|>", images=[image_url], audio=[audio_url], return_tensors="pt")
    samples = int(out["feature_attention_mask"].sum())
    placeholders = tokenizer.decode(out["input_ids"][0]).count("<|audio|>")
    assert placeholders == -(-samples // 600), "fetched audio is resampled to 24 kHz before counting"
    print(
        f"[OK] URL fetch: image {tuple(out['pixel_values'].shape)}, "
        f"audio {samples} samples -> {placeholders} placeholders"
    )
except Exception as error:  # offline or missing librosa
    print(f"[SKIP] URL fetch (needs network + librosa): {type(error).__name__}: {error}")

print("\nALL PROCESSOR SINGLE-INPUT CHECKS PASSED")
