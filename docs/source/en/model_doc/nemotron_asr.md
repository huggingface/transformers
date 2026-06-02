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

NemotronAsr is a cache-aware, streaming automatic speech recognition (ASR) model from the NVIDIA **Nemotron Speech Streaming**
family. It pairs a [Fast Conformer](https://huggingface.co/papers/2305.05084) encoder ([`NemotronAsrEncoder`]) — adapted for
cache-aware streaming inference with limited attention context, causal downsampling, and convolution/attention caches — with an
RNN-T (Recurrent Neural Network Transducer) head ([`NemotronAsrForRNNT`]) composed of an LSTM prediction network and a joint
network.

Because the encoder is cache-aware, audio can be transcribed chunk-by-chunk with bounded latency while sharing left context
through the encoder caches. The [`NemotronAsrCacheAwareStreamingBuffer`] helper splits an audio stream into the chunk sizes the
encoder expects (handling the pre-encode cache, STFT lookahead, and mel-frame trimming), and [`NemotronAsrForRNNT.streaming_step`]
threads the encoder + decoder state across chunks.

This model can also be run offline (full utterance at once) with [`NemotronAsrForRNNT.generate`].

## Usage

### Offline transcription

```python
from transformers import AutoProcessor, NemotronAsrForRNNT
from datasets import load_dataset, Audio

model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = NemotronAsrForRNNT.from_pretrained(model_id).eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt", sampling_rate=16000)
generated = model.generate(**inputs)
print(processor.batch_decode(generated.sequences, skip_special_tokens=True))
```

### Streaming transcription

```python
import torch
import soundfile as sf
from transformers import AutoProcessor, NemotronAsrForRNNT, NemotronAsrCacheAwareStreamingBuffer

model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = NemotronAsrForRNNT.from_pretrained(model_id).eval()

audio, _ = sf.read("audio.wav", dtype="float32")  # 16 kHz mono

buffer = NemotronAsrCacheAwareStreamingBuffer(model, processor, att_context_size=[70, 6])
buffer.append_audio(audio)

state = model.get_initial_streaming_state(batch_size=1, device=model.device, dtype=model.dtype)
all_tokens = []
for inputs, drop in buffer:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        chunk_tokens = model.streaming_step(inputs, drop, state)
    all_tokens.extend(chunk_tokens[0])

print(processor.batch_decode(torch.tensor([all_tokens]), skip_special_tokens=True)[0])
```

## NemotronAsrConfig

[[autodoc]] NemotronAsrConfig

## NemotronAsrEncoderConfig

[[autodoc]] NemotronAsrEncoderConfig

## NemotronAsrFeatureExtractor

[[autodoc]] NemotronAsrFeatureExtractor

## NemotronAsrCacheAwareStreamingBuffer

[[autodoc]] NemotronAsrCacheAwareStreamingBuffer

## NemotronAsrEncoderModelOutput

[[autodoc]] NemotronAsrEncoderModelOutput

## NemotronAsrTDTOutput

[[autodoc]] NemotronAsrTDTOutput

## NemotronAsrEncoder

[[autodoc]] NemotronAsrEncoder
    - forward

## NemotronAsrForRNNT

[[autodoc]] NemotronAsrForRNNT
    - forward
    - generate
    - streaming_step
