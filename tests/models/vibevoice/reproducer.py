"""
1. Setup
```
git clone git@github.com:pengzhiliang/transformers.git vibevoice-original
cd vibevoice-original
git checkout 6e6e60fb95ca908feb0b039483adcc009809f579
pip install -e .
pip install diffusers
```
2. Place this script inside `tests/models/vibevoice/vibevoice`
3. Run: python tests/models/vibevoice/reproducer.py

"""

import os
import re
import time
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor
from transformers.audio_utils import load_audio_librosa
from transformers.trainer_utils import set_seed


# set seed for deterministic
set_seed(42)

model_path = "microsoft/VibeVoice-1.5B"
sampling_rate = 24000
cfg_scale = 1.3
max_new_tokens = 32
bfloat16 = False
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Results fixture path - relative to tests/models/vibevoice
NUM_SAVED_VALS = 200
RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_results_single.json"
results = {}

# example files: https://huggingface.co/datasets/bezzam/vibevoice_samples/tree/main
example_files_repo = "bezzam/vibevoice_samples"
script_fn = "text_examples/test.txt"
audio_fn = [
    "voices/en-Alice_woman.wav", 
    "voices/en-Frank_man.wav"
]

# Download example files
repo_dir = snapshot_download(
    repo_id=example_files_repo,
    repo_type="dataset",
)
txt_path = f"{repo_dir}/{script_fn}"
audio_paths = [f"{repo_dir}/{fn}" for fn in audio_fn]
voice_samples = []
for voice_path in audio_paths:
    voice_samples.append(load_audio_librosa(voice_path, sampling_rate=sampling_rate))


# Script preparation functions utility
def parse_txt_script(txt_path: str) -> tuple[list[str], list[str]]:
    """
    Open and parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []

    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    full_script = '\n'.join(scripts)

    # Cleaning
    full_script = full_script.replace("’", "'")

    return full_script, speaker_numbers


print(f"Reading script from: {txt_path}")
full_script, speaker_numbers = parse_txt_script(txt_path)

# load model
processor = VibeVoiceProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_path,
    device_map=torch_device,
).to(torch_device, dtype=torch.bfloat16 if bfloat16 else torch.float32).eval()
model.set_ddpm_inference_steps(num_steps=10)
model_dtype = next(model.parameters()).dtype


"""
INTEGRATION TEST WITH AUDIO SAMPLES FOR VOICE CLONING
"""
print("\n\nRunning integration test WITH audio samples...\n")

# Prepare inputs for the model
inputs = processor(
    text=[full_script], # Wrap in list for batch processing
    voice_samples=[voice_samples], # Wrap in list for batch processing
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
).to(model_dtype)

# Generate audio
start_time = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    cfg_scale=cfg_scale,
    tokenizer=processor.tokenizer,
    generation_config={'do_sample': False},
    verbose=True,
)
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save results to fixtures
results["speech_outputs"] = outputs.speech_outputs[0].cpu().float().numpy()[..., :NUM_SAVED_VALS].tolist()

# Create fixtures directory if it doesn't exist
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save to JSON
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)

print(f"Results saved to: {RESULTS_PATH}")

# Save audio files for debugging (processor handles device internally)
txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
output_dir = "./debug_audio_outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"{txt_filename}_voicetext_ORIGINAL.wav")
processor.save_audio(
    outputs.speech_outputs[0],
    output_path,
)
print(f"Saved output to {output_path}")


# """
# INTEGRATION TEST WITHOUT AUDIO SAMPLES
# - not supported as generation assumes speech tensors: https://github.com/vibevoice-community/VibeVoice/blob/187cf6203600a45cac0b9527977cb0e933b30af4/vibevoice/modular/modeling_vibevoice_inference.py#L470
# """
# print("\n\nRunning integration test WITHOUT audio samples...\n\n")

# # Prepare inputs for the model
# inputs = processor(
#     text=[full_script], # Wrap in list for batch processing
#     padding=True,
#     return_tensors="pt",
#     return_attention_mask=True,
# ).to(model_dtype)

# del inputs["speech_tensors"] 
# del inputs["speech_masks"]
# del inputs["speech_input_mask"]

# # Generate audio
# start_time = time.time()
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=max_new_tokens,
#     cfg_scale=cfg_scale,
#     tokenizer=processor.tokenizer,
#     generation_config={'do_sample': False},
#     verbose=True,
# )
# generation_time = time.time() - start_time
# print(f"Generation time: {generation_time:.2f} seconds")

# # Save results to fixtures
# results["speech_outputs"] = outputs.speech_outputs[0].cpu().float().numpy()[..., :NUM_SAVED_VALS].tolist()

# # Create fixtures directory if it doesn't exist
# RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# # Save to JSON
# with open(RESULTS_PATH, "w") as f:
#     json.dump(results, f)

# print(f"Results saved to: {RESULTS_PATH}")

# # Save audio files for debugging (processor handles device internally)
# txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
# output_dir = "./debug_audio_outputs"
# os.makedirs(output_dir, exist_ok=True)

# output_path = os.path.join(output_dir, f"{txt_filename}_textonly_ORIGINAL.wav")
# processor.save_audio(
#     outputs.speech_outputs[0],
#     output_path,
# )
# print(f"Saved output to {output_path}")
