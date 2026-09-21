<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-17.*

# Nemotron 3 Diarization

## Overview

Nemotron 3 Diarization is a 100M-parameter streaming speaker diarization model from NVIDIA that answers "who spoke
when" for up to eight speakers. It is a [Streaming Sortformer](https://huggingface.co/papers/2507.18446): a
31-layer Transformer encoder with rotary position embeddings reads 10 ms log-mel frames stacked into 80 ms frames and
predicts, for every 10 ms frame, the activity probability of each speaker. Speakers are ordered by their first
arrival in the audio. Streaming works with an Arrival-Order Speaker Cache (AOSC) and a FIFO queue of recent frames that
every chunk attends to, so a single checkpoint runs at input buffer latencies from 320 ms to 30.4 s, and the audio
duration is unbounded.

The original implementation lives in [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/Speech).

## Usage tips

- The model always processes audio in chunks of `config.chunk_length` encoder frames (80 ms each) plus
  `config.chunk_right_context` look-ahead frames, even for a complete recording. The checkpoint ships with the
  offline profile below; pick another latency profile by overriding the four streaming fields when loading.
- [`Nemotron3DiarizationProcessor`] wraps the checkpoint's feature extractor ([`NemotronAsrStreamingFeatureExtractor`],
  the same NeMo mel front-end as Parakeet without per-feature normalization) and sizes the chunks of a streaming
  session.
- Outputs are logits at the spectrogram frame rate (one per 10 ms). `logits.sigmoid()` gives per-speaker activity
  probabilities; `attention_mask` marks the valid frames of a padded batch.
- In a padded batch, the last 80 ms of a shorter sample can differ slightly from a single-sample run: the upsampler's
  convolution sees a padded neighbour frame there, as in the original implementation.

| Profile | Input buffer latency | `fifo_length` | `chunk_length` | `chunk_right_context` | `speaker_cache_update_period` |
|---|---|---|---|---|---|
| offline (default) | 30.4 s | 40 | 340 | 40 | 300 |
| low latency | 1.04 s | 264 | 9 | 4 | 222 |
| very low latency | 0.64 s | 264 | 6 | 2 | 222 |
| ultra low latency | 0.32 s | 264 | 3 | 1 | 222 |

## Usage

### Diarizing a recording

```python
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization-preview"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto")
# config is "offline" by default, so equivalent to do:
# offline_profile = {"fifo_length": 40, "chunk_length": 340, "chunk_right_context": 40, "speaker_cache_update_period": 300}
# AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto", **offline_profile)

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)
inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)

probabilities = model(**inputs).logits.sigmoid()[0]  # (num_frames, 8), one frame every 10 ms
```

Turning the probabilities into speaker segments is a thresholding pass, at a frame duration of 10 ms:

```python
frame_duration = processor.feature_extractor.hop_length / sampling_rate
active = probabilities > 0.5  # (num_frames, num_speakers)
for speaker in range(active.shape[1]):
    changes = active[:, speaker].int().diff(prepend=active.new_zeros(1), append=active.new_zeros(1))
    starts, ends = (changes == 1).nonzero()[:, 0], (changes == -1).nonzero()[:, 0]
    for start, end in zip(starts.tolist(), ends.tolist()):
        print(f"speaker_{speaker}: {start * frame_duration:.2f}s - {end * frame_duration:.2f}s")
```

### Streaming

In a streaming session, audio arrives chunk by chunk and the spectrogram is extracted per chunk. The processor sizes
those chunks and switches the feature extractor between centered windows for the first chunk and uncentered windows for
the later ones, so that the per-chunk spectrogram reproduces, frame for frame, a single full-utterance pass.

Each model call takes one chunk together with its look-ahead frames, and returns the logits of the chunk alone.
`use_cache=True` says that more audio follows: the look-ahead frames are computed but neither returned nor kept, so
the next call has to open with those same frames again. The generator below does that by advancing its cursor
`processor.num_mel_frames_per_step` frames while each chunk carries `processor.num_mel_frames_per_audio_chunk` of
them. The call also returns the updated `speaker_cache`, to hand to the next one. The last call of a session uses
`use_cache=False`: no further audio will arrive to look ahead into, so every remaining frame is scored, including a
trailing partial chunk.

> [!IMPORTANT]
> A latency profile has to reach both the model and the processor. Whenever you load a model with a profile, pass the
> same chunk sizes to [`~Nemotron3DiarizationProcessor.set_streaming_profile`]: a processor left on another profile
> sizes its chunks for that one, and the model rejects them.

```python
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization-preview"
# the "low latency" row of the profile table above
profile = {"fifo_length": 264, "chunk_length": 9, "chunk_right_context": 4, "speaker_cache_update_period": 222}
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto", **profile)

# the model configuration is the source of truth for the chunk sizes the processor must produce
processor.set_streaming_profile(model.config.chunk_length, model.config.chunk_right_context)
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)


def input_features_generator():
    """Yields the spectrogram of each chunk, and whether it is the last one of the session."""
    inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
    )
    yield inputs.input_features, False

    mel_frame_idx = processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        inputs = processor(
            audio[start_idx:end_idx], sampling_rate=sampling_rate, is_streaming=True, is_first_audio_chunk=False
        )
        yield inputs.input_features, False

        mel_frame_idx += processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)

    # the audio ended: the frames left in the buffer are the last ones of the session
    inputs = processor(
        audio[start_idx:], sampling_rate=sampling_rate, is_streaming=True, is_first_audio_chunk=False
    )
    yield inputs.input_features, True


speaker_cache, probabilities = None, []
for input_features, is_last_chunk in input_features_generator():
    outputs = model(
        input_features.to(model.device, dtype=model.dtype),
        speaker_cache=speaker_cache,
        use_cache=not is_last_chunk,
    )
    probabilities.append(outputs.logits.sigmoid())
    speaker_cache = outputs.speaker_cache

probabilities = torch.cat(probabilities, dim=1)[0]  # (num_frames, 8), one frame every 10 ms
```

## Nemotron3DiarizationConfig

[[autodoc]] Nemotron3DiarizationConfig

## Nemotron3DiarizationEncoderConfig

[[autodoc]] Nemotron3DiarizationEncoderConfig

## Nemotron3DiarizationModel


[[autodoc]] Nemotron3DiarizationModel
    - forward

## Nemotron3DiarizationProcessor

[[autodoc]] Nemotron3DiarizationProcessor
    - __call__

## Nemotron3DiarizationSpeakerCache

[[autodoc]] Nemotron3DiarizationSpeakerCache

## Nemotron3DiarizationOutput

[[autodoc]] Nemotron3DiarizationOutput

## Nemotron3DiarizationForAudioFrameClassification

[[autodoc]] Nemotron3DiarizationForAudioFrameClassification
    - forward
