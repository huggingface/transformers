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
*This model was published in HF papers on 2023-05-08 and contributed to Hugging Face Transformers on 2026-06-02.*

# NemotronAsr

## Overview

TODO

## Usage

### Offline transcription

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="nvidia/nemotron-speech-streaming-en-0.6b")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id, device_map="auto")

audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    sampling_rate=processor.feature_extractor.sampling_rate,
)

inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True)
print(processor.decode(output.sequences, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

### Streaming transcription
> [!NOTE]
> This is an experimental feature and the API is subject to change.

For real-time transcription, audio is split into chunks following:

```python
from threading import Thread
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
from transformers.audio_utils import load_audio

model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id, device_map="auto")

processor.set_num_lookahead_tokens(6)
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
    sampling_rate=sampling_rate,
)

first_chunk_inputs = processor(
    audio[: processor.num_samples_first_audio_chunk],
    sampling_rate=sampling_rate,
    is_streaming=True,
    is_first_audio_chunk=True,
    return_tensors="pt",
)
first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)


def input_features_generator():
    yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft

    start_idx = mel_frame_idx * hop_length - n_fft // 2
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
        inputs = processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        yield inputs.input_features

        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2


streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
generate_kwargs = {
    **first_chunk_inputs,
    "input_features": input_features_generator(),
    "streamer": streamer,
}
thread = Thread(target=model.generate, kwargs=generate_kwargs)
thread.start()

# Iterate over the streamer to get text chunks as they are generated
print("Model output (streaming):", end=" ", flush=True)
for text_chunk in streamer:
    print(text_chunk, end="", flush=True)
thread.join()
```

#### Streaming latency

The latency is set by `num_lookahead_tokens`, the right attention context (lookahead, in subsampled encoder frames) each chunk waits for before it is emitted. A larger value lets each chunk see more future audio: better accuracy at the cost of higher latency. Inspect the supported trade-offs, select one, and read back the resulting latency:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")

# Each supported `num_lookahead_tokens` mapped to its streaming latency in milliseconds:
print(processor.supported_streaming_latencies_ms)
# {13: 1120, 6: 560, 1: 160, 0: 80}

# Select a right attention context (this also re-derives the streaming chunk sizes used above):
processor.set_num_lookahead_tokens(6)

# Latency of the current selection:
print(processor.streaming_latency_ms)
# 560
```

`set_num_lookahead_tokens` sizes the chunks the processor emits, and the matching `num_lookahead_tokens` must reach `generate` (in the snippet above it travels through `**inputs`/`**first_chunk_inputs`, which carries `num_lookahead_tokens`). Streaming `generate` raises if it is omitted.

## NemotronAsrConfig

[[autodoc]] NemotronAsrConfig

## NemotronAsrEncoderConfig

[[autodoc]] NemotronAsrEncoderConfig

## NemotronAsrFeatureExtractor

[[autodoc]] NemotronAsrFeatureExtractor

## NemotronAsrProcessor

[[autodoc]] NemotronAsrProcessor
    - __call__
    - decode

## NemotronAsrEncoderModelOutput

[[autodoc]] NemotronAsrEncoderModelOutput

## NemotronAsrRNNTOutput

[[autodoc]] NemotronAsrRNNTOutput

## NemotronAsrEncoder

[[autodoc]] NemotronAsrEncoder
    - forward

## NemotronAsrForRNNT

[[autodoc]] NemotronAsrForRNNT
    - forward
    - generate
