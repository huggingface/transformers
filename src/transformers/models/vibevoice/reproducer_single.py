"""
Setup
```
git clone git@github.com:pengzhiliang/transformers.git vibevoice-original
cd vibevoice-original
git checkout 6e6e60fb95ca908feb0b039483adcc009809f579
pip install -e .
pip install diffusers
```
"""


import os
import re
import time
import numpy as np

import torch
from huggingface_hub import snapshot_download
from transformers import VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor
from transformers.audio_utils import load_audio_librosa

# set seed for deterministic
torch.manual_seed(42)
np.random.seed(42)

model_path = "microsoft/VibeVoice-1.5B"
sampling_rate = 24000
cfg_scale = 1.3
max_new_tokens = 32
output_dir = "./vibevoice_output"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# example files: https://huggingface.co/datasets/bezzam/vibevoice_samples/tree/main
example_files_repo = "bezzam/vibevoice_samples"
script_fn = "text_examples/2p_goat.txt"
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


# 1) Prepare text
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
).to(torch_device).eval()
model.set_ddpm_inference_steps(num_steps=10)

# Prepare inputs for the model
inputs = processor(
    text=[full_script], # Wrap in list for batch processing
    voice_samples=[voice_samples], # Wrap in list for batch processing
    # # NOTE: batch of different lengths fails when shorter one complete
    # text=[full_script[:319], full_script],
    # voice_samples=[voice_samples, voice_samples],
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)
# # save inputs
# torch.save(inputs, "vibevoice_inputs.pt")

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

# Calculate audio duration and additional metrics
if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
    # Assuming 24kHz sample rate (common for speech synthesis)
    sample_rate = sampling_rate
    audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
    audio_duration = audio_samples / sample_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

    print(f"Generated audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
else:
    print("No audio output generated")

# Calculate token metrics
input_tokens = inputs['input_ids'].shape[1]  # Number of input tokens
output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
generated_tokens = output_tokens - input_tokens

print(f"Prefilling tokens: {input_tokens}")
print(f"Generated tokens: {generated_tokens}")
print(f"Total tokens: {output_tokens}")

# Save output (processor handles device internally)
txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
os.makedirs(output_dir, exist_ok=True)

for i, speech in enumerate(outputs.speech_outputs):
    output_path = os.path.join(output_dir, f"{txt_filename}_generated_{i}.wav")
    processor.save_audio(
        speech,
        output_path,
    )
    print(f"Saved output to {output_path}")
print("\nGenerated speech output shape:", outputs.speech_outputs[0].shape if outputs.speech_outputs else "No speech output")
