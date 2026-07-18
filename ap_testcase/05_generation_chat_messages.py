"""Full conditional generation from chat messages (the instruct quick-start path).

Loads the full composite model and generates:
  1) a TEXT-ONLY chat turn rendered by the checkpoint's chat template (expects "Bern");
  2) a MULTIMODAL chat turn: the template is rendered first (string content with placeholders), then
     the processor prepares image + audio tensors and `generate` runs on the fused stream;
  3) the fully AUTOMATIC path: standard list-of-content-blocks messages with a local image file:
     `apply_chat_template(tokenize=True, return_dict=True)` loads the media itself.
"""

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor

CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "/Users/rkre/swissai_repos/material/Apertus-1.5-8B-composite-hf")

processor = AutoProcessor.from_pretrained(CHECKPOINT)
print("loading model (bf16, CPU) ...")
model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).eval()


def generate(inputs, max_new_tokens=16):
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --- 1) text-only chat ------------------------------------------------------------------------------
messages = [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
completion = generate(inputs, max_new_tokens=8)
assert "Bern" in completion, f"expected 'Bern', got: {completion!r}"
print(f"[OK] text-only chat: {completion!r}")

# --- 2) multimodal chat: render template, then process media explicitly -----------------------------
image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)
messages = [{"role": "user", "content": "<|image|> What do you see, and what do you hear? <|audio|>"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], audio=[audio], return_tensors="pt")
completion = generate(inputs, max_new_tokens=24)
assert len(completion.strip()) > 0
print(f"[OK] multimodal chat (explicit media): {completion!r}")

# --- 3) fully automatic: content blocks with a local file, media auto-loaded ------------------------
with tempfile.TemporaryDirectory() as tmp:
    image_path = os.path.join(tmp, "test_image.png")
    image.save(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    assert "pixel_values" in inputs, "the chat template path must load and process the image itself"
    completion = generate(inputs, max_new_tokens=24)
assert len(completion.strip()) > 0
print(f"[OK] multimodal chat (auto-loaded from file): {completion!r}")

print("\nALL CHAT-MESSAGE GENERATION CHECKS PASSED")
