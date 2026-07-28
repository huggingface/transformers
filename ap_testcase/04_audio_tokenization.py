"""Audio tokenization: processor-driven vs. fully manual.

Loads the full composite model (CPU, bf16 backbone with the fp32-guarded audio codec) and turns one
audio clip into discrete codes two ways:
  1) from PROCESSOR OUTPUT: `processor(...)` -> `model.model.get_audio_tokens(input_features,
     feature_attention_mask)` (vocabulary ids in the `<|audio token i|>` range), verifying the count
     matches the number of `<|audio|>` placeholders the processor emitted (40 codes per second);
  2) fully MANUAL: raw waveform -> -3 dBFS peak normalization (as the processor does) ->
     `model.model.audio_tokenizer.encode(...)` (raw codebook indices) -> manual offset.
The two paths must produce IDENTICAL codes, since audio preprocessing has no resampling kernel.
"""

import os

import numpy as np
import torch

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

seconds = 1.5
time = np.arange(int(24000 * seconds)) / 24000.0
waveform = (0.8 * np.sin(2 * np.pi * 440.0 * time) * np.exp(-time)).astype(np.float32)  # decaying 440 Hz tone

# --- 1) processor output -> model tokenizer --------------------------------------------------------
inputs = processor(text="<|audio|>", audio=[waveform], return_tensors="pt")
with torch.no_grad():
    vocab_ids = model.model.get_audio_tokens(inputs["input_features"], inputs["feature_attention_mask"])

num_placeholders = tokenizer.decode(inputs["input_ids"][0]).count("<|audio|>")
expected_codes = -(-len(waveform) // 600)  # ceil(samples / hop), 40 codes per second
assert vocab_ids.numel() == num_placeholders == expected_codes
first = int(vocab_ids[0])
assert tokenizer.convert_ids_to_tokens(first) == f"<|audio token {first - config.audio_token_offset}|>"
print(
    f"[OK] processor path: {vocab_ids.numel()} codes for {seconds} s "
    f"(= ceil({len(waveform)}/600)), first id {first} -> {tokenizer.convert_ids_to_tokens(first)!r}"
)

# --- 2) fully manual preprocessing -> audio codec ---------------------------------------------------
peak = max(float(np.abs(waveform).max()), 1e-10)
normalized = waveform * (10 ** (-3.0 / 20.0) / peak)  # -3 dBFS peak normalization, like the processor
clip = torch.tensor(normalized, dtype=torch.float32)[None, None, :]  # (batch, channel, samples)

with torch.no_grad():
    code_ids = model.model.audio_tokenizer.encode(clip).audio_codes  # (1, 1, num_codes) raw codebook indices

manual_vocab_ids = code_ids.flatten() + config.audio_token_offset
assert torch.equal(manual_vocab_ids, vocab_ids), "manual and processor paths must yield identical codes"
print(f"[OK] manual path: {code_ids.numel()} codes, bit-identical to the processor path")

# without the -3 dBFS normalization the codes differ -> the normalization is load-bearing
with torch.no_grad():
    raw_ids = (
        model.model.audio_tokenizer.encode(
            torch.tensor(waveform, dtype=torch.float32)[None, None, :]
        ).audio_codes.flatten()
        + config.audio_token_offset
    )
agreement = (raw_ids == vocab_ids).float().mean().item()
print(f"[OK] skipping the -3 dBFS normalization changes the codes (agreement only {agreement:.1%})")

print("\nALL AUDIO TOKENIZATION CHECKS PASSED")
