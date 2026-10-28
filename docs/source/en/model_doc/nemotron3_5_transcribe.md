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
*This model was contributed to Hugging Face Transformers on 2026-10-22.*

# Nemotron 3.5 Transcribe

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Nemotron 3.5 Transcribe is an Audio Language Model for 
- multilingual transcription
- speaker-attributed transcription
- speech translation
- audio question answering.

The model has three parts:

1. An audio model ([`Nemotron3_5TranscribeAudioModel`]) that turns the log-mel spectrogram into one embedding every
   80 ms. Two networks read the same spectrogram:
   - a speech recognition encoder ([`Nemotron3_5TranscribeAsrAudioModel`]), which encodes what is said,
   - the [Nemotron 3 Diarization](./nemotron3_diarization) model, which predicts which of up to eight speakers are
     active.

   Each embedding is the encoder output plus a fixed embedding of the active speakers.
2. A linear projector into the language model's embedding space.
3. A [Nemotron-H](./nemotron_h) language model (hybrid Mamba-2, attention and mixture-of-experts layers), which
   generates the text.

A word timestamp head (47M parameters, NVIDIA's `ctc_timestamps` feature)
reads the speech recognition encoder outputs: two Transformer layers of the same type as the encoder's, followed by a
CTC classifier over its own vocabulary of 23,862 SentencePiece pieces. It runs only when word timestamps are requested
(see [Word timestamps](#word-timestamps)).

### Long audio

Audio longer than 40 s is encoded in 40 s chunks, each seeing up to 4 s of context on either side; only the chunk's
own frames are kept. The speech recognition encoder, the diarization model and the word timestamp head use the same
chunks, and the diarization speaker cache carries the speaker identities across chunks. The lengths are
`chunk_length`, `chunk_left_context` and `chunk_right_context` of `config.audio_config.diarization_config`.

## Usage

### Transcription

[`~Nemotron3_5TranscribeProcessor.apply_transcription_request`] builds the prompt from the processor's default
transcription request (verbatim transcript with punctuation and capitalization). It takes a path, a URL or an audio
array sampled at 16 kHz.

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

Pass a list of audio, all paths or URLs, or all arrays. A list of prompts of the same length can be passed as `prompt`
(`None` entries use the default prompt). Here the second recording, a conversation, is
transcribed with the speaker-tagged prompt of the [next section](#speaker-attributed-transcription): decode with
`skip_special_tokens=False` to keep the speaker tags, and remove the padding of the shorter output.

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

Ask for speaker tags in the prompt, through [`~Nemotron3_5TranscribeProcessor.apply_chat_template`], and the model
starts each speaker turn with a `<spk:N>` tag. Decode with `skip_special_tokens=False` to keep the tags. The processor
chat template reproduces the training prompt format: an empty system turn, the text and the audio of a user turn joined
by a space, and thinking disabled. Put the instruction before the audio and do not add a system prompt.

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

The output is a single string, for example `<spk:0> to celebrate the release ... <spk:1> hi everybody ...`.
[`~Nemotron3_5TranscribeProcessor.extract_speaker_dict`] splits it into one dictionary per speaker turn, with the
`Speaker` and `Content` keys of [VibeVoice-ASR](./vibevoice_asr) (the generated text has no timestamps, see
[Word timestamps](#word-timestamps) for word-level times):

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

Pass `return_word_timestamp_logits=True` to `generate`: the audio model runs once, and the output adds the logits of
the word timestamp head and of the diarization model. [`~Nemotron3_5TranscribeProcessor.align_word_timestamps`] aligns
the decoded text to them. For each recording it returns a dictionary with:

- `chunks`: every generated word, as the word timestamps of the speech recognition pipeline: `text`, `timestamp`
  (start and end in seconds) and its speaker tag `speaker`,
- `diarization`: the speaker activity segments predicted by the diarization model (`speaker`, `timestamp`),
- `speaker_tag_to_diarization_speaker`: the diarization speaker matched to each transcript tag,
- `failed_speakers`: the tags whose words could not be aligned.

Use the speaker-tagged transcription prompt of the previous section and decode with `skip_special_tokens=False`: a
transcript without speaker tags is aligned as a single speaker. The alignment requires `torchaudio` and, to match tags to diarization speakers, `scipy`.

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

```text
[{'text': 'to', 'timestamp': (0.32, 0.35000000000000003), 'speaker': 0}, {'text': 'celebrate', 'timestamp': (0.35000000000000003, 0.96), 'speaker': 0}, {'text': 'the', 'timestamp': (0.96, 1.04), 'speaker': 0}]
```

The processor aligns the generated words to the head's per-frame piece probabilities with a CTC search, one search per
speaker tag, restricted to the frames where the matched diarization speaker is active, then snaps the word boundaries
to the 10 ms diarization segments. The head runs once, on the encoder outputs already computed for generation, and the
alignment is a search on the CPU.

To align a transcript that was not generated (a reference or an edited transcript), run the audio model and the head
without `generate`:

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

The duration of each recording, hence the time step of the head's frames, is read from `input_features_mask` to the
10 ms hop of the spectrogram below. `language` is detected from the transcript when it is not given. Chinese and Japanese are aligned per character; Thai is not
supported (it needs `pythainlp`). A transcript that cannot be aligned (empty, too long for the audio, or with a
character outside the alignment vocabulary) raises a `ValueError`. A speaker tag whose words find no valid path through
the audio is listed in `failed_speakers`, and its words are left out. To evaluate diarization, use the `diarization`
segments rather than the word timestamps.

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
