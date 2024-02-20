from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import numpy as np

DEVICE = "cuda:0"
model_id = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(DEVICE)

audio, _ = librosa.load("../audio.mp3", sr=16_000)
audio2, _ = librosa.load("../audio_2.mp3", sr=16_000)
audio = np.hstack([audio, audio, audio, audio, audio, audio])
audio2 = np.hstack([audio2, audio2, audio2, audio2, audio2])

#audio = np.vstack([audio, audio])
inputs = processor(audio2,
                sampling_rate=16_000,
                return_tensors="pt",
                truncation=False, # False so the audio isn't truncated and whole audio is sent to the model
                return_attention_mask=True,
                padding=True
                )

input_features = inputs.to(DEVICE)
print(inputs.input_features.shape, inputs.keys())

gen_kwargs = {
        "return_token_timestamps": True,
        "return_segments": True,
        "condition_on_prev_tokens": False,
        "return_timestamps": True,
        "no_speech_threshold": 0.6,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        }
outputs = model.generate(**input_features, **gen_kwargs)
print(outputs["sequences"])


# decode token ids to text
transcription = processor.batch_decode(outputs["sequences"], skip_special_tokens=False)

print(transcription[0])
#print(outputs['token_timestamps'])

transcription = processor.tokenizer.decode(
        token_ids=outputs["sequences"][0],
        output_offsets=True,
        decode_with_timestamps=True,
        )

print(outputs["segments"][0][0].keys())
print(processor.tokenizer.all_special_ids[-1] + 1)

word_timestamps_length = [segment["token_timestamps"].shape for segment in outputs["segments"][0]]
print(word_timestamps_length, outputs['sequences'][0].shape)
print([(segment["start"], segment["end"]) for segment in outputs["segments"][0]])

per_segment_word_timestamps = [segment["token_timestamps"] for segment in outputs["segments"][0]]
print("Word level timestamps", per_segment_word_timestamps)


per_segment_word_timestamps_batch2 = [segment["token_timestamps"] for segment in outputs["segments"][1]]
print(f"Batch 2: {per_segment_word_timestamps_batch2}")

