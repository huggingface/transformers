import gc
import os
import wave


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from transformers import (
    AutoModelForAudioTokenization,
    AutoModelForTextToWaveform,
    AutoTokenizer,
    MossTTSDelayProcessor,
)


model_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
# audio_tokenizer_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer" # Ted: Need to convert this to HF transformers style!
audio_tokenizer_id = "/tmp/moss-audio-tokenizer-hf/"
output_path = "/tmp/ted.wav"

text = "Hello world!"
requested_audio_tokens = None
max_new_tokens = 1024
text_temperature = 1.5
text_top_p = 1.0
text_top_k = 50
audio_temperature = 1.7
audio_top_p = 0.8
audio_top_k = 25
audio_repetition_penalty = 1.0
audio_tokenizer_dtype = torch.float32


def describe_waveform(label: str, waveform: torch.Tensor) -> tuple[float, float]:
    waveform = waveform.detach().cpu().to(torch.float32)
    peak = waveform.abs().max().item() if waveform.numel() > 0 else 0.0
    rms = waveform.square().mean().sqrt().item() if waveform.numel() > 0 else 0.0
    print(f"{label}: samples={waveform.numel()}, peak={peak:.8f}, rms={rms:.8f}")
    return peak, rms


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int):
    waveform = waveform.detach().cpu().to(torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    peak, rms = describe_waveform("Waveform before normalization", waveform)
    if peak == 0 or rms == 0:
        raise RuntimeError("Decoded waveform is all zeros.")

    # Generated audio can be low-RMS even when nonzero. Normalize by RMS first,
    # then protect against clipping with a peak limiter.
    target_rms = 0.08
    gain = min(target_rms / rms, 200.0)
    waveform = waveform * gain
    peak = waveform.abs().max().item()
    if peak > 0.95:
        waveform = waveform * (0.95 / peak)
    waveform = waveform.clamp(-1.0, 1.0)
    describe_waveform("Waveform after normalization", waveform)
    pcm = (waveform * 32767.0).to(torch.int16).numpy()

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_generated_outputs(generated):
    normalized = []
    for start_length, generation_ids in generated:
        if isinstance(start_length, torch.Tensor):
            start_length = int(start_length.detach().cpu().item())
        else:
            start_length = int(start_length)
        normalized.append((start_length, generation_ids.detach().cpu()))
    return normalized


tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForTextToWaveform.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

processor = MossTTSDelayProcessor(
    tokenizer=tokenizer,
    audio_tokenizer=None,
    model_config=model.config,
)
sampling_rate = int(processor.model_config.sampling_rate)

message = processor.build_user_message(
    text=text,
    language="English",
    tokens=requested_audio_tokens,
)

inputs = processor([message], mode="generation").to(model.device)

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        text_temperature=text_temperature,
        text_top_p=text_top_p,
        text_top_k=text_top_k,
        audio_temperature=audio_temperature,
        audio_top_p=audio_top_p,
        audio_top_k=audio_top_k,
        audio_repetition_penalty=audio_repetition_penalty,
    )

generated = normalize_generated_outputs(generated)
for item_idx, (start_length, generation_ids) in enumerate(generated):
    audio_codes = processor.apply_de_delay_pattern(generation_ids[:, 1:])
    pad_rows = (audio_codes == processor.model_config.audio_pad_code).all(dim=1)
    nonpad_rows = int((~pad_rows).sum().item())
    print(
        f"Generated item {item_idx}: start_length={int(start_length)}, "
        f"tokens={generation_ids.shape[0]}, channels={generation_ids.shape[1]}, "
        f"dedelayed_audio_rows={audio_codes.shape[0]}, nonpad_audio_rows={nonpad_rows}"
    )

del inputs
del model
cleanup_cuda()

audio_tokenizer = AutoModelForAudioTokenization.from_pretrained(
    audio_tokenizer_id,
    dtype=audio_tokenizer_dtype,
)
audio_tokenizer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_tokenizer = audio_tokenizer.to(audio_tokenizer_device).eval()
processor.audio_tokenizer = audio_tokenizer

with torch.inference_mode():
    decoded_messages = processor.decode(generated)

decoded_message = decoded_messages[0]
if decoded_message is None:
    raise RuntimeError("No audio was generated.")

print("Decoded content:", decoded_message.content)
print("Audio segments:", len(decoded_message.audio_codes_list))

waveforms = [segment for segment in decoded_message.audio_codes_list if isinstance(segment, torch.Tensor)]
if len(waveforms) == 0:
    raise RuntimeError("No audio waveform segments were decoded.")

for segment_idx, segment in enumerate(waveforms):
    describe_waveform(f"Segment {segment_idx}", segment)
waveform = torch.cat(waveforms, dim=-1)

save_wav(output_path, waveform, sample_rate=sampling_rate)

print(f"Saved {output_path}")
