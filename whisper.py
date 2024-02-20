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
audio = np.hstack([audio, audio, audio, audio, audio, audio])
inputs = processor(audio,
                sampling_rate=16_000,
                return_tensors="pt",
                truncation=False, # False so the audio isn't truncated and whole audio is sent to the model
                return_attention_mask=True,
                padding=True
                )

input_features = inputs.to(DEVICE)
#inputs["input_features"] = inputs.input_features.repeat(1, 1, 6)
print(inputs.input_features.shape)

outputs = model.generate(**input_features, return_segments=True, return_token_timestamps=True, return_timestamps=True)

# decode token ids to text
transcription = processor.batch_decode(outputs["sequences"], skip_special_tokens=False)

print(transcription[0])
print(outputs.keys())
print(outputs["sequences"])
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
all_word_timestamps = [x + y["start"] for x, y in zip(per_segment_word_timestamps, outputs["segments"][0])]
print("Word level timestamps", per_segment_word_timestamps)

