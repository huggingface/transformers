<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was contributed to Hugging Face Transformers on 2026-09-21.*

# Nemotron 3 Diarization



## Overview

Nemotron 3 Diarization is an open-weight streaming speaker diarization model designed to determine "who spoke when" in real-world audio. It supports both streaming and offline inference, handles up to eight speakers, and orders speaker outputs by each speaker's first arrival in the input audio.

The model uses the Arrival-Order Speaker Cache (AOSC) [1](https://huggingface.co/papers/2507.18446) and FIFO queue introduced for Streaming Sortformer [1](https://huggingface.co/papers/2507.18446), [2](https://huggingface.co/papers/2409.06656). A single checkpoint supports configurable latency profiles, from an 80 ms input buffer to a 30.4 s offline-style buffer, and configurable output frame resolution in multiples of 10 ms. With chunked inference, the maximum audio duration is not limited.

## Usage



### Offline

```python
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization-preview"
revision = "refs/pr/6"
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto", revision=revision)

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)
inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)

with torch.inference_mode():
    logits = model(**inputs).logits  # (1, num_frames, 8), one frame every 10 ms

segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
for segment in segments:
    print(f"speaker_{segment['Speaker']}: {segment['Start']:.2f}s - {segment['End']:.2f}s")
```



### Streaming

Audio arrives chunk by chunk, and each forward takes one chunk: the processor cuts it for its `streaming_mode` and
adds `num_lookahead_frames`, the number of trailing look-ahead frames the model attends to but does not score, since
they open the next chunk. The forward returns the `speaker_cache` to pass to the next call. The last chunk of a
session is extracted with `is_last_audio_chunk=True`: it has no look-ahead, so every remaining frame is scored.

| `streaming_mode`          | Latency¹ |
| ------------------------- | -------- |
| `"low_latency"` (default) | 1.04 s   |
| `"very_low_latency"`      | 0.64 s   |
| `"ultra_low_latency"`     | 0.32 s   |

¹ Audio to wait for before the model runs on a chunk: the chunk plus its look-ahead, excluding compute time.

```python
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization-preview"
revision = "refs/pr/6"
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto", revision=revision)
processor.set_streaming_mode("low_latency")  # the default, can also be "very_low_latency" and "ultra_low_latency"
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)


def inputs_generator():
    """Yields the processor outputs of each chunk."""
    yield processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
    )

    mel_frame_idx = processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
        )
        mel_frame_idx += processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)

    # the audio ended: the frames left in the buffer are the last ones of the session
    yield processor(
        audio[start_idx:],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=False,
        is_last_audio_chunk=True,
    )


speaker_cache, logits = None, []
with torch.inference_mode():
    for inputs in inputs_generator():
        inputs = inputs.to(model.device, dtype=model.dtype)
        # `inputs` carries `num_lookahead_frames` for every chunk but the last, `speaker_cache` links the chunks
        outputs = model(**inputs, speaker_cache=speaker_cache)
        logits.append(outputs.logits)  # the chunk's frames, without its look-ahead
        speaker_cache = outputs.speaker_cache

logits = torch.cat(logits, dim=1)  # (1, num_frames, 8), one frame every 10 ms
segments = processor.extract_speaker_dict(logits)[0]  # [{"Start": 0.0, "End": 15.43, "Speaker": 0}, ...]
```



### Making it go brrr

The encoder input of a streaming step is `[speaker cache | FIFO | chunk]`, whose length changes as the cache and the
FIFO fill and shrink: `torch.compile` would recompile about a hundred times per session. Padding every step to the
largest window of the mode fixes the shape. Positions restart at zero on every chunk, so right padding does not change
the valid frames:

```python
import torch.nn.functional as F

chunk_length, chunk_right_context = processor.streaming_modes[processor.streaming_mode]
max_window = (
    model.config.streaming_config.speaker_cache_length
    + model.config.streaming_config.fifo_length
    + chunk_length
    + chunk_right_context
)
encoder = model.model
compiled_forward = torch.compile(encoder.forward, mode="reduce-overhead", fullgraph=True, dynamic=False)


def padded_forward(inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
    batch_size, num_frames, _ = inputs_embeds.shape
    if attention_mask is None:
        attention_mask = inputs_embeds.new_ones(batch_size, num_frames, dtype=torch.bool)
    padding = max_window - num_frames
    hidden_states = compiled_forward(
        inputs_embeds=F.pad(inputs_embeds, (0, 0, 0, padding)),
        attention_mask=F.pad(attention_mask.bool(), (0, padding), value=False),
        position_ids=torch.arange(max_window, device=inputs_embeds.device)[None, :],
        **kwargs,
    )
    return hidden_states[:, :num_frames].clone()  # CUDA graphs reuse the output buffer


encoder.forward = padded_forward

# warm up before the session: compiles, then records the CUDA graph, so the first real chunk runs at full speed
with torch.inference_mode():
    for _ in range(3):
        hidden_size = model.config.audio_config.hidden_size
        padded_forward(torch.zeros(1, max_window, hidden_size, device=model.device, dtype=model.dtype))
```

The streaming loop above then compiles once. The offline forward chunks the same way, so the same wrapper applies with
`config.fifo_length`, `config.chunk_length` and `config.chunk_right_context` in `max_window`.


| Speedup vs eager (A100, batch size 1) | float32 | bfloat16 |
| ------------------------------------- | ------- | -------- |
| streaming, per step                   | 1.2x    | 4.4x     |
| offline, 488 s recording              | 1.3x    | 2.8x     |




## Nemotron3DiarizationConfig

[[autodoc]] Nemotron3DiarizationConfig

## Nemotron3DiarizationAudioConfig

[[autodoc]] Nemotron3DiarizationAudioConfig

## Nemotron3DiarizationHeadConfig

[[autodoc]] Nemotron3DiarizationHeadConfig

## Nemotron3DiarizationStreamingConfig

[[autodoc]] Nemotron3DiarizationStreamingConfig

## Nemotron3DiarizationAudioModel

[[autodoc]] Nemotron3DiarizationAudioModel
    - forward

## Nemotron3DiarizationProcessor

[[autodoc]] Nemotron3DiarizationProcessor
    - **call**

## Nemotron3DiarizationSpeakerCache

[[autodoc]] Nemotron3DiarizationSpeakerCache

## Nemotron3DiarizationOutput

[[autodoc]] Nemotron3DiarizationOutput

## Nemotron3DiarizationForAudioFrameClassification

[[autodoc]] Nemotron3DiarizationForAudioFrameClassification
    - forward