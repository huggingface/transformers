from huggingface_hub import snapshot_download
import torch
from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
import diffusers
from transformers.audio_utils import load_audio_librosa


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "bezzam/VibeVoice-1.5B"
example_files_repo = "bezzam/vibevoice_samples"
audio_fn = [
    "voices/en-Alice_woman.wav", 
    "voices/en-Frank_man.wav"
]
sampling_rate = 24000
max_new_tokens = 32

# prepare input
repo_dir = snapshot_download(
    repo_id=example_files_repo,
    repo_type="dataset",
)
audio_paths = [f"{repo_dir}/{fn}" for fn in audio_fn]

# # 1) With audio
# conversation = [
#     {"role": "0", "content": [
#         {"type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."},
#         {"type": "audio", "path": load_audio_librosa(audio_paths[0], sampling_rate=sampling_rate)}
#     ]},
#     {"role": "1", "content": [
#         {"type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings."},
#         {"type": "audio", "path": load_audio_librosa(audio_paths[1], sampling_rate=sampling_rate)}
#     ]},
#     {"role": "0", "content": [{"type": "text", "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible."}]},
#     {"role": "1", "content": [{"type": "text", "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just… sinks it. I remember jumping off my couch and yelling, 'Oh man, is that true? That's Unbelievable!'"}]},
# ]

# 2) Without audio
conversation = [
    {"role": "0", "content": [
        {"type": "text", "text": "Hello everyone, and welcome to the VibeVoice podcast. I'm your host, Linda, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."},
    ]},
    {"role": "1", "content": [
        {"type": "text", "text": "Thanks so much for having me, Linda. You're absolutely right—this question always brings out some seriously strong feelings."},
    ]},
    {"role": "0", "content": [{"type": "text", "text": "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the Finals, six championships. That kind of perfection is just incredible."}]},
    {"role": "1", "content": [{"type": "text", "text": "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just… sinks it. I remember jumping off my couch and yelling, 'Oh man, is that true? That's Unbelievable!'"}]},
]


# load model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path, device_map=torch_device,
).eval()
model_dtype = next(model.parameters()).dtype

# apply
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(torch_device, dtype=model_dtype)
print("\ninput_ids shape : ", inputs["input_ids"].shape)

# create noise scheduler
noise_scheduler_class = getattr(diffusers, model.generation_config.noise_scheduler)
noise_scheduler = noise_scheduler_class(
    **model.generation_config.noise_scheduler_config
)

# generate
outputs = model.generate(
    **inputs,
    noise_scheduler=noise_scheduler,
    return_dict_in_generate=True,
    do_sample=False,
    max_new_tokens=max_new_tokens,
    # return_dict_in_generate=True,
)

speech_output = outputs.speech_outputs[0]
print("\nGenerated speech output shape:", speech_output.shape)

# save
output_path = "integration_test_output.wav"
processor.save_audio(speech_output, output_path, sampling_rate=sampling_rate)
print(f"Audio saved to {output_path}")
