"""
1. Setup
```
git clone git@github.com:pengzhiliang/transformers.git transformers-vibevoice
cd transformers-vibevoice
pip install -e .
pip install diffusers
```
2. Place this script inside `src/transformers/models/vibevoice`
3. Run: python src/transformers/models/vibevoice/example.py

Example outputs:

batch_inference=False (like `original_example.py`)
```
Fetching 21 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 46726.99it/s]
Fetching 2 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 5178.15it/s]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.33s/it]
Model loaded on cuda with dtype torch.float32

input_ids shape :  torch.Size([1, 1116])
Generating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [00:30<00:00,  8.50it/s]
Generation time: 30.13 seconds
Generated audio duration: 34.13 seconds
RTF (Real Time Factor): 0.88x
Prefilling tokens: 1116
Generated tokens: 256
Total tokens: 1372
Saved output to ./vibevoice_output/2p_goat_generated_0.wav
```

batch_inference=True (not supported by original)
```
Fetching 2 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 43690.67it/s]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.24s/it]
Model loaded on cuda with dtype torch.float32
Inputs IDs shape :  torch.Size([2, 1116])
Generating:  69%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                   | 176/256 [00:23<00:09,  8.70it/s]Sample 0 completed at step 176
Generating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [00:33<00:00,  7.56it/s]
Generation time: 33.87 seconds
Generated audio duration: 23.20 seconds
RTF (Real Time Factor): 1.46x
Prefilling tokens: 1116
Generated tokens: 256
Total tokens: 1372
Saved output to ./vibevoice_output/2p_goat_generated_0.wav
Saved output to ./vibevoice_output/2p_goat_generated_1.wav
```
"""

import os
import re
import time

from huggingface_hub import snapshot_download
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
from transformers.audio_utils import load_audio_librosa



model_path = "bezzam/VibeVoice-1.5B"
# model_path = "bezzam/VibeVoice-7B"   # comment/uncomment to switch model

seed = 42
sampling_rate = 24000
cfg_scale = 1.3
n_diffusion_steps = 10
max_new_tokens = None    # None for full generation or max of model generation_config
output_dir = "./vibevoice_output"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
batch_inference = False     # with conversations of diff length (not supported by original)
verbose = False
bfloat16 = False

# set seed for deterministic
torch.manual_seed(seed)
np.random.seed(seed)

# example files: https://huggingface.co/datasets/bezzam/vibevoice_samples/tree/main
example_files_repo = "bezzam/vibevoice_samples"
script_fn = "text_examples/2p_goat.txt"
audio_fn = [
    "voices/en-Alice_woman.wav", 
    "voices/en-Frank_man.wav"
]


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


def convert_script_to_conversation(script_text: str, audio_files: list = None, sampling_rate: int = 24000, verbose: bool = True) -> list:
    """
    Convert original VibeVoice script format to conversation format for chat templates.
    Input: "Speaker 1: text\nSpeaker 2: text\n..."
    Output: [{"role": "0", "content": [{"type": "text", "text": "..."}, {"type": "audio", "audio": numpy_array}]}, ...]
    
    Args:
        script_text: Script in "Speaker X: text" format
        audio_files: Optional list of audio file paths to assign to speakers.
                    Order should match speaker appearance order (first unique speaker gets first audio, etc.)
        sampling_rate: Sampling rate for loading audio files
        verbose: Whether to print detailed information about the conversion process
    
    Note: Speaker IDs are normalized to start from 0 (e.g., Speaker 1 -> role "0", Speaker 2 -> role "1")
    """
    lines = script_text.strip().split('\n')
    conversation = []
    
    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    
    # Track unique speakers in order of first appearance
    unique_speakers = []
    speaker_to_audio = {}
    
    # First pass: collect unique speakers in order of appearance
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            speaker_id = match.group(1).strip()
            if speaker_id not in unique_speakers:
                unique_speakers.append(speaker_id)
    
    # Find the minimum speaker ID for normalization
    min_speaker_id = min(int(spk) for spk in unique_speakers)
    if verbose:
        print(f"Original speaker IDs: {unique_speakers}")
        print(f"Normalizing speakers: min_id={min_speaker_id}, will subtract {min_speaker_id} from all IDs")
    
    # Create mapping from original ID to normalized ID (starting from 0)
    speaker_id_mapping = {}
    for i, speaker_id in enumerate(unique_speakers):
        normalized_id = str(int(speaker_id) - min_speaker_id)
        speaker_id_mapping[speaker_id] = normalized_id
    
    if verbose:
        print(f"Speaker ID mapping: {speaker_id_mapping}")
    
    # Map audio files to normalized speakers if provided
    if audio_files is not None:
        if len(audio_files) != len(unique_speakers):
            raise ValueError(
                f"Number of audio files ({len(audio_files)}) must match "
                f"number of unique speakers ({len(unique_speakers)}). "
                f"Unique speakers found: {unique_speakers}"
            )
        
        # Map each unique speaker (normalized) to their corresponding audio content
        for i, original_speaker_id in enumerate(unique_speakers):
            normalized_speaker_id = speaker_id_mapping[original_speaker_id]
            # Load audio content instead of storing path
            audio_content = load_audio_librosa(audio_files[i], sampling_rate=sampling_rate)
            speaker_to_audio[normalized_speaker_id] = audio_content
        
        if verbose:
            print(f"Audio mapping (normalized IDs):")
            for normalized_id, audio_content in speaker_to_audio.items():
                original_id = unique_speakers[int(normalized_id)]
                print(f"  Speaker {original_id} -> {normalized_id} -> loaded audio ({len(audio_content)} samples)")
    
    # Second pass: build conversation with normalized IDs and audio assignments
    speakers_with_audio_assigned = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            original_speaker_id = match.group(1).strip()
            normalized_speaker_id = speaker_id_mapping[original_speaker_id]
            text = match.group(2).strip()
            
            if text:
                # Build content array
                content = [{"type": "text", "text": text}]
                
                # Add audio if this speaker has one assigned and hasn't been used yet
                if (audio_files is not None and 
                    normalized_speaker_id in speaker_to_audio and 
                    normalized_speaker_id not in speakers_with_audio_assigned):
                    content.append({
                        "type": "audio", 
                        "audio": speaker_to_audio[normalized_speaker_id]
                    })
                    speakers_with_audio_assigned.add(normalized_speaker_id)
                
                conversation.append({
                    "role": normalized_speaker_id,
                    "content": content
                })
    
    return conversation


# Download example files
repo_dir = snapshot_download(
    repo_id=example_files_repo,
    repo_type="dataset",
)
if verbose:
    print(f"Dataset snapshot downloaded to: {repo_dir}")
txt_path = f"{repo_dir}/{script_fn}"
audio_paths = [f"{repo_dir}/{fn}" for fn in audio_fn]

# Convert script to conversation format with audio
full_script, speaker_numbers = parse_txt_script(txt_path)
conversation = convert_script_to_conversation(full_script, audio_paths, sampling_rate=sampling_rate, verbose=verbose)
if batch_inference:
    conversation_batch = [conversation[:2], conversation]
else:
    conversation_batch = [conversation]

# load model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16 if bfloat16 else None,
    device_map=torch_device,
).eval()

# determine model dtype
model_dtype = next(model.parameters()).dtype
print(f"Model loaded on {torch_device} with dtype {model_dtype}")

if verbose:
    # Verify prepared text sequence
    inputs = processor.apply_chat_template(
        conversation_batch, 
        tokenize=False,
    )
    print("\nPrepared text sequence:")
    for i, script in enumerate(inputs):
        print(f"--- Script {i+1} ---")
        print(script)

# Prepare inputs for the model
inputs = processor.apply_chat_template(
    conversation_batch, 
    tokenize=True,
    return_dict=True
).to(torch_device, dtype=model_dtype)
print("\ninput_ids shape : ", inputs["input_ids"].shape)

# Generate audio
start_time = time.time()

# Define a callback to monitor the progress of the generation
completed_samples = set()
with tqdm(desc="Generating") as pbar:
    def monitor_progress(p_batch):
        # p_batch format: [current_step, max_step, completion_step] for each sample
        finished_samples = (p_batch[:, 0] == p_batch[:, 1]).nonzero(as_tuple=False).squeeze(1)
        if finished_samples.numel() > 0:
            for sample_idx in finished_samples.tolist():
                if sample_idx not in completed_samples:
                    completed_samples.add(sample_idx)
                    completion_step = int(p_batch[sample_idx, 2])
                    print(f"Sample {sample_idx} completed at step {completion_step}", flush=True)

        active_samples = p_batch[:, 0] < p_batch[:, 1]
        if active_samples.any():
            active_progress = p_batch[active_samples]
            max_active_idx = torch.argmax(active_progress[:, 0])
            p = active_progress[max_active_idx].detach().cpu()
        else:
            p = p_batch[0].detach().cpu()

        pbar.total = int(p[1])
        pbar.n = int(p[0])
        pbar.update()

    # Generate!
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        n_diffusion_steps=n_diffusion_steps,
        monitor_progress=monitor_progress,
        return_dict_in_generate=True,
    )
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Calculate audio duration and additional metrics
if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
    audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
    audio_duration = audio_samples / sampling_rate
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