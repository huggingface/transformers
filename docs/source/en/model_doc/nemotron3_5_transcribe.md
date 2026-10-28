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
*This model was contributed to Hugging Face Transformers on 2026-10-28.*

# Nemotron 3.5 Transcribe

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Nemotron 3.5 Transcribe is an audio language model for multilingual transcription, speaker-attributed transcription,
speech translation and audio question answering. A speech recognition encoder and the
[Nemotron 3 Diarization](./nemotron3_diarization) model read the same spectrogram; their outputs are projected into a
[Nemotron-H](./nemotron_h) language model. A word timestamp head on the encoder outputs gives word-level times.

## Usage

### Transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(samples + "en-Alice_woman.wav").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

### Batched transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = [samples + "en-Alice_woman.wav", samples + "diarization_example.mp3"]
speaker_attributed_prompt = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation "
    "and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; "
    "assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged "
    "transcript."
)
prompt = [None, speaker_attributed_prompt]

inputs = processor.apply_transcription_request(audio, prompt=prompt).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
texts = processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
print([text.replace(processor.tokenizer.pad_token, "") for text in texts])
```

### Speaker-attributed transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

prompt = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation "
    "and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; "
    "assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged "
    "transcript."
)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": samples + "diarization_example.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    enable_thinking=False,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
text = processor.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
print(text)
```

[`~Nemotron3_5TranscribeProcessor.extract_speaker_dict`] splits the output into one dictionary per speaker turn:

```python
for turn in processor.extract_speaker_dict(text):
    print(turn)
```

```text
{'Speaker': 0, 'Content': "to celebrate the release of nemotron three diarization we invited ... from nvidia"}
{'Speaker': 1, 'Content': "hi everybody thanks for having me i'm jane a pocket voice from qti"}
{'Speaker': 2, 'Content': "nice to meet you chris and jane i'm serena the queen three voice"}
...
```

### Word timestamps

Requires `torchaudio` and `scipy`.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

prompt = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation "
    "and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; "
    "assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged "
    "transcript."
)

inputs = processor.apply_transcription_request(samples + "diarization_example.mp3", prompt=prompt).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, return_word_timestamp_logits=True)
texts = processor.decode(outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
timestamps = processor.align_word_timestamps(
    texts,
    outputs.word_timestamp_logits,
    inputs.input_features_mask,
    outputs.speaker_logits,
)
print(timestamps[0]["chunks"][:3])
```

To align a transcript that was not generated:

```python
input_features_mask = inputs.input_features_mask
audio_outputs = model.get_audio_features(inputs.input_features, input_features_mask)
word_timestamp_logits = model.get_word_timestamp_logits(audio_outputs.asr_last_hidden_state, input_features_mask)
timestamps = processor.align_word_timestamps(
    ["<spk:0> to celebrate the ..."],
    word_timestamp_logits,
    input_features_mask,
    audio_outputs.speaker_logits,
)
```

## Nemotron3_5TranscribeConfig

[[autodoc]] Nemotron3_5TranscribeConfig

## Nemotron3_5TranscribeAsrAudioConfig

[[autodoc]] Nemotron3_5TranscribeAsrAudioConfig

## Nemotron3_5TranscribeAudioConfig

[[autodoc]] Nemotron3_5TranscribeAudioConfig

## Nemotron3_5TranscribeWordTimestampConfig

[[autodoc]] Nemotron3_5TranscribeWordTimestampConfig

## Nemotron3_5TranscribeProcessor

[[autodoc]] Nemotron3_5TranscribeProcessor
    - __call__
    - apply_transcription_request
    - apply_chat_template
    - extract_speaker_dict
    - align_word_timestamps

## Nemotron3_5TranscribeAsrAudioModel

[[autodoc]] Nemotron3_5TranscribeAsrAudioModel
    - forward

## Nemotron3_5TranscribeAudioModel

[[autodoc]] Nemotron3_5TranscribeAudioModel
    - forward

## Nemotron3_5TranscribeWordTimestampModel

[[autodoc]] Nemotron3_5TranscribeWordTimestampModel
    - forward

## Nemotron3_5TranscribeModel

[[autodoc]] Nemotron3_5TranscribeModel
    - forward
    - get_audio_features
    - get_word_timestamp_hidden_states

## Nemotron3_5TranscribeForConditionalGeneration

[[autodoc]] Nemotron3_5TranscribeForConditionalGeneration
    - forward
    - get_audio_features
    - get_word_timestamp_logits
