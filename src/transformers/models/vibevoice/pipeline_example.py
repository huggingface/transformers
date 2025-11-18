from transformers import pipeline
import diffusers
import soundfile as sf
import time

model_id = "bezzam/VibeVoice-1.5B"
txt = "Hello everyone, and welcome to the VibeVoice podcast. I'm your host Vibey, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Thomas here to talk about it with me."
audio_fp = "src/transformers/models/vibevoice/voices/en-Alice_woman.wav"
max_new_tokens = 32

# load pipeline
pipe = pipeline("text-to-speech", model=model_id)

# prepare input
conversation = [
    {"role": "0", "content": [
        {"type": "text", "text": txt},
    ]},
]
if audio_fp is not None:
    conversation[0]["content"].append({"type": "audio", "path": audio_fp})
input_data = pipe.tokenizer.apply_chat_template(
    conversation, tokenize=False,
)

# generate
noise_scheduler = diffusers.DPMSolverMultistepScheduler(
    beta_schedule="squaredcos_cap_v2",
    num_train_timesteps=1000,
    prediction_type="v_prediction"
)
generate_kwargs = {
    # default generation args can be added here: https://huggingface.co/bezzam/VibeVoice-1.5B/blob/main/generation_config.json
    # "cfg_scale": 1.3,
    # "ddpm_inference_steps": 10,
    "noise_scheduler": noise_scheduler,
    "max_new_tokens": max_new_tokens,
}
start_time = time.time()
output = pipe(input_data, generate_kwargs=generate_kwargs)
end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds.")

# save the audio to a file
audio = output["audio"][0].squeeze()
fn = "vibevoice_pipeline_output.wav"
sf.write(fn, audio, 24000)
print(f"Audio saved to {fn}")
