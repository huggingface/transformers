<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-26.*

# CohereAsr

## Overview

Cohere ASR, [released](https://cohere.com/blog/transcribe) by Cohere on March 26th, 2026, is a 2B parameter Conformer-based encoder-decoder speech recognition model.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).

## Usage

### Short-form transcription

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

revision = "refs/pr/6"
processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", revision=revision)
model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto", revision=revision)

audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    sampling_rate=16000,
)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True)
print(text)
```

### Punctuation control

Pass `punctuation=False` to obtain lower-cased output without punctuation marks.

```python
inputs_pnc = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=True)
inputs_nopnc = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=False)
```

### Long-form transcription

For audio longer than the feature extractor's `max_audio_clip_s`, the feature extractor automatically splits the waveform into chunks.
The processor reassembles the per-chunk transcriptions using the returned `audio_chunk_index`.

```python
audio_long = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
    sampling_rate=16000,
)

inputs = processor(audio=audio_long, return_tensors="pt", language="en", sampling_rate=16000)
audio_chunk_index = inputs.get("audio_chunk_index")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en")
print(text)
```

### Batched inference

Multiple audio files can be processed in a single call. When the batch mixes short-form and long-form audio, the
processor handles chunking and reassembly.

```python
audio_short = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    sampling_rate=16000,
)
audio_long = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
    sampling_rate=16000,
)

inputs = processor([audio_short, audio_long], sampling_rate=16000, return_tensors="pt", language="en")
audio_chunk_index = inputs.get("audio_chunk_index")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(
    outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
)
print(text)
```

### Non-English transcription

Specify the language code to transcribe in any of the 14 supported languages.

```python
audio_es = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/fleur_es_sample.wav",
    sampling_rate=16000,
)

inputs = processor(audio_es, sampling_rate=16000, return_tensors="pt", language="es", punctuation=True)
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True)
print(text)
```

## CohereAsrConfig

[[autodoc]] CohereAsrConfig

## CohereAsrFeatureExtractor

[[autodoc]] CohereAsrFeatureExtractor
    - __call__

## CohereAsrProcessor

[[autodoc]] CohereAsrProcessor
    - __call__

## CohereAsrPreTrainedModel

[[autodoc]] CohereAsrPreTrainedModel
    - forward

## CohereAsrModel

[[autodoc]] CohereAsrModel
    - forward

## CohereAsrForConditionalGeneration

[[autodoc]] CohereAsrForConditionalGeneration
    - forward
