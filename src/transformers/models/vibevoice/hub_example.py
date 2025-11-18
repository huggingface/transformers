import time
import diffusers
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
from transformers.audio_utils import load_audio_librosa


# model_path = "bezzam/VibeVoice-7B"
model_path = "bezzam/VibeVoice-1.5B"
sampling_rate = 24000
max_new_tokens = 512
cfg_scale = 1.3     # classifier-free guidance for diffusion process

# set seed for deterministic
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# create conversation with an audio for the first time a speaker appears to clone that particular voice
conversation = [
    {"role": "0", "content": [
        {"type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."},
        {"type": "audio", "path": load_audio_librosa("https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav", sampling_rate=sampling_rate)}
    ]},
    {"role": "1", "content": [
        {"type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings."},
        {"type": "audio", "path": load_audio_librosa("https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Frank_man.wav", sampling_rate=sampling_rate)}
    ]},
    {"role": "0", "content": [
        {"type": "text", "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible."},
    ]},
    {"role": "1", "content": [
        {"type": "text", "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it"},
    ]},
]

# load model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    device_map=torch_device,
).to(torch_device).eval()

# prepare inputs
inputs = processor.apply_chat_template(
    conversation, 
    tokenize=True,
    return_dict=True
).to(torch_device)

# Generate audio with a callback to track progress
noise_scheduler = getattr(diffusers, model.generation_config.noise_scheduler)(
    **model.generation_config.noise_scheduler_config
)
start_time = time.time()
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
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        noise_scheduler=noise_scheduler,
        monitor_progress=monitor_progress,
        return_dict_in_generate=True,
    )
generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")

# Save audio
output_fp = "vibevoice_output.wav"
processor.save_audio(outputs.speech_outputs[0], output_fp)
print(f"Saved output to {output_fp}")
