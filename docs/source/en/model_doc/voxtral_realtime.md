<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-02-15.*

# VoxtralRealtime

VoxtralRealtime is a streaming speech-to-text model from [Mistral AI](https://mistral.ai), designed for real-time automatic speech recognition (ASR). Unlike the offline [Voxtral](./voxtral) model which processes complete audio files, VoxtralRealtime is architected for low-latency, incremental transcription by processing audio in chunks as they arrive.

The model combines an audio encoder with a Mistral-based language model decoder, using time conditioning embeddings and causal convolutions with padding caches to enable efficient streaming inference.


## Usage

### Offline Transcription

For transcribing complete audio files, use the processor and model directly. The generation length is automatically determined from the audio length.

```python
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from datasets import load_dataset

repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]

inputs = processor(audio, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs)
decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

print(decoded_outputs[0])
```

### Batched Offline Transcription

Multiple audio samples can be transcribed in a single forward pass:

```python
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from datasets import load_dataset

repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = [ds[i]["audio"]["array"] for i in range(2)]

inputs = processor(audio, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs)
decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

for decoded_output in decoded_outputs:
    print(decoded_output)
```

### Audio encoder precomputation

By default, when the full audio is available (i.e. not streaming), the audio encoder and projector are run once before generation begins. The resulting embeddings are then simply sliced at each decoding step, which is much faster than running the encoder repeatedly.

This is the default behavior (`precompute_audio_embeds=True`). You can disable it if needed. Note that the default vLLM implementation runs the encoder at every step since it relies on a different optimization paradigm.

### Streaming Transcription
> [!NOTE]
> This is an experimental feature and the API is subject to change.

For real-time transcription, audio is split into chunks following:

```python
from transformers import (
    VoxtralRealtimeProcessor,
    VoxtralRealtimeForConditionalGeneration,
    TextIteratorStreamer,
)
from datasets import load_dataset
from threading import Thread
import numpy as np

model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
processor = VoxtralRealtimeProcessor.from_pretrained(model_id)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]
# Manually pad the audio to account for right padding tokens required by the model
xaudio = np.pad(audio, (0, processor.num_right_pad_tokens * processor.raw_audio_length_per_tok))

first_chunk_inputs = processor(
    audio[:processor.num_samples_first_audio_chunk],
    is_streaming=True,
    is_first_audio_chunk=True,
    return_tensors="pt"
)
first_chunk_inputs.to(model.device, dtype=model.dtype)

def input_features_generator():
    yield first_chunk_inputs.input_features

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    win_length = processor.feature_extractor.win_length
    
    start_idx = mel_frame_idx * hop_length - win_length // 2
    end_idx = start_idx + processor.num_samples_per_audio_chunk

    while (end_idx:=start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
        inputs = processor(
            audio[start_idx:end_idx],
            is_streaming=True,
            is_first_audio_chunk=False,
            return_tensors="pt"
        )
        inputs.to(model.device, dtype=model.dtype)
        yield inputs.input_features

        mel_frame_idx += processor.audio_length_per_tok
        start_idx = mel_frame_idx * hop_length - win_length // 2

streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
generate_kwargs = {
    "input_ids": first_chunk_inputs.input_ids,
    "input_features": input_features_generator(),
    "num_delay_tokens": first_chunk_inputs.num_delay_tokens,
    "streamer": streamer,
}
thread = Thread(target=model.generate, kwargs=generate_kwargs)
thread.start()

# Iterate over the streamer to get text chunks as they are generated
print("Model output (streaming):", end=" ", flush=True)
for text_chunk in streamer:
    print(text_chunk, end="", flush=True)
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).

## VoxtralRealtimeConfig

[[autodoc]] VoxtralRealtimeConfig

## VoxtralRealtimeEncoderConfig

[[autodoc]] VoxtralRealtimeEncoderConfig

## VoxtralRealtimeTextConfig

[[autodoc]] VoxtralRealtimeTextConfig

## VoxtralRealtimeFeatureExtractor

[[autodoc]] VoxtralRealtimeFeatureExtractor

## VoxtralRealtimeProcessor

[[autodoc]] VoxtralRealtimeProcessor
    - __call__

## VoxtralRealtimeEncoder

[[autodoc]] VoxtralRealtimeEncoder
    - forward

## VoxtralRealtimeForConditionalGeneration

[[autodoc]] VoxtralRealtimeForConditionalGeneration
    - forward
    - get_audio_features
