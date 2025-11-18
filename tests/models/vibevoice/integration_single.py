import os
import time
import json
from pathlib import Path
import diffusers

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
from transformers.audio_utils import load_audio_librosa
from transformers.trainer_utils import set_seed

# set seed for deterministic
set_seed(42)

model_path = "bezzam/VibeVoice-1.5B"
sampling_rate = 24000
cfg_scale = 1.3
max_new_tokens = 32
bfloat16 = False
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Results fixture path - relative to tests/models/vibevoice
RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/vibevoice/expected_results_single.json"
results = {}

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
audio_paths = [f"{repo_dir}/{fn}" for fn in audio_fn]


conversation = [
    {"role": "0", "content": [
        {"type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."},
        {"type": "audio", "path": load_audio_librosa(audio_paths[0], sampling_rate=sampling_rate)}
    ]},
    {"role": "1", "content": [
        {"type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings."},
        {"type": "audio", "path": load_audio_librosa(audio_paths[1], sampling_rate=sampling_rate)}
    ]},
]

# load model
processor = AutoProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    device_map=torch_device,
).to(torch_device, dtype=torch.bfloat16 if bfloat16 else torch.float32).eval()
model_dtype = next(model.parameters()).dtype

# Prepare inputs for the model
inputs = processor.apply_chat_template(
    conversation, 
    tokenize=True,
    return_dict=True
).to(torch_device, dtype=model_dtype)
print("\ninput_ids shape : ", inputs["input_ids"].shape)

# Generate audio
start_time = time.time()
noise_scheduler = getattr(diffusers, model.generation_config.noise_scheduler)(
    **model.generation_config.noise_scheduler_config
)
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    cfg_scale=cfg_scale,
    do_sample=False,
    noise_scheduler=noise_scheduler,
    return_dict_in_generate=True,
)
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save audio files for debugging
output_dir = "./debug_audio_outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"generated_voicetext.wav")
processor.save_audio(outputs.speech_outputs[0], output_path)
print(f"Saved output to {output_path}")

# Load expected results and compare
with open(RESULTS_PATH, "r") as f:
    expected_results = json.load(f)
expected_speech = torch.tensor(expected_results["speech_outputs"])

# Save results to fixtures
generated_speech = outputs.speech_outputs[0].cpu().float()[..., :expected_speech.shape[-1]]
if bfloat16:
    torch.testing.assert_close(generated_speech, expected_speech, rtol=1e-4, atol=1e-4)
else:
    torch.testing.assert_close(generated_speech, expected_speech, rtol=1e-6, atol=1e-6)
print(f"✓ Speech outputs match expected values within tolerance")
