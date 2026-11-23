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

# Audio processors

An audio processor turns a raw audio signal into the tensors a model expects. The output shape depends
on the model — a waveform model wants samples, a spectrogram model wants mel frames — and the
processor produces the right one for whichever model you loaded. It also handles padding, truncation
and resampling.

Load one with [`~PreprocessingMixin.from_pretrained`], then pass the signal along with the
`sampling_rate` of the arrays you are passing.

```py
from transformers import AutoAudioProcessor

audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base")
processed = audio_processor(audio_array, sampling_rate=16000)
processed.keys()
dict_keys(['audio_values'])
```

## Inputs

### Shapes and batches

Pass a single waveform as a NumPy array or PyTorch tensor. Mono waveforms have shape `(num_samples,)`.
Multichannel waveforms use the channel-first shape `(num_channels, num_samples)` and are downmixed to mono
before further processing.

Pass a batch as a list of waveforms. The waveforms may have different lengths; padding can make their output
shapes uniform later in the pipeline.

You can also pass a local file path or HTTP(S) URL directly, or a list of paths or URLs for a batch. Referenced
audio is decoded at the processor's native sampling rate and returned as mono audio by the decoder.

| input | meaning |
|---|---|
| `(num_samples,)` | one mono waveform |
| `(num_channels, num_samples)` | one multichannel waveform |
| list of `(num_samples,)` arrays | batch of mono waveforms |
| list of `(num_channels, num_samples)` arrays | batch of multichannel waveforms |
| local path or HTTP(S) URL | one referenced waveform |
| list of paths or URLs | batch of referenced waveforms |

A bare two-dimensional array always represents one multichannel waveform. It does not represent a mono batch
with shape `(batch_size, num_samples)`; wrap the individual waveforms in a list to express a batch. This avoids
the ambiguity between a batch axis and a channel axis when both would occupy axis 0.

### Sampling rate and resampling

Audio models are trained on waveforms sampled at a specific rate. The rate is measured in hertz (Hz), or samples
per second, and is part of the model's input contract. For example, a model trained at 16,000 Hz expects 16,000
samples to represent one second of audio.

An array contains only sample values, so it does not fully describe a waveform on its own. The same array of
16,000 values could represent one second at 16,000 Hz or two seconds at 8,000 Hz. The processor cannot infer which
interpretation is correct from the array's shape or values.

An audio processor prepares waveforms for a particular model, so every audio processor has a native
`sampling_rate`, expressed in Hz. Pass the actual rate of your arrays to `__call__` through the `sampling_rate`
argument. The processor compares the two rates and resamples the waveform when necessary.

That is why we recommend—and expect—users to pass `sampling_rate` together with audio provided as an array. By
default, when the provided rate does not match the processor's native `sampling_rate`, the processor resamples the
audio to its native rate before preparing the model inputs.

```py
audio_processor.sampling_rate
16000
```

If `sampling_rate` is omitted for an array, the processor logs a warning and assumes the array already uses its
native rate. A wrong assumption changes the apparent duration and frequency content of the audio, which can lead
to incorrect model results.

```py
processed = audio_processor(audio_array)
# `sampling_rate` was not provided. The audio arrays will be assumed to be sampled at 16000 Hz, which can
# produce incorrect results if that assumption is wrong.
```

If the provided rate differs from the model's native rate, the processor resamples to the native rate by default
and warns that it did.

```py
audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base")
processed = audio_processor(eight_khz_audio, sampling_rate=8000)
# Resampling audio from 8000 Hz to Wav2Vec2AudioProcessor's native sampling rate of 16000 Hz.
processed["audio_values"].shape
(1, 16000)
```

Resampling on every call costs time. When you are processing a whole dataset, resample once at load
instead — [Datasets](https://hf.co/docs/datasets/index) does it on the fly as samples are read:

```py
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset[0]["audio"]["sampling_rate"]
16000
```

`sampling_rate` describes passed arrays only. Audio given as a URL or a path is decoded at the
model's native rate by construction, so a `sampling_rate` passed alongside it has nothing to describe
and is ignored, with a warning saying so.

> [!WARNING]
> Omitting `sampling_rate` entirely means the processor cannot check the rate, and mismatched audio
> will be processed as though it were correct. The processor warns and assumes its native rate. Always pass the
> rate when providing arrays.

> [!TIP]
> `AudioProcessor` replaces the legacy feature extractors. `AutoFeatureExtractor` and the
> `XxxFeatureExtractor` classes still load and still work, with a deprecation warning. The output keys
> changed: `input_values` is now `audio_values` and `input_features` is now `audio_features`.

For the contract these classes follow — configuration files, option precedence, and the pipeline
itself — see [Preprocessing](./preprocessing).

## Output keys

A processor emits one of two families, depending on whether it extracts a spectrogram:

| model kind | key | shape |
|---|---|---|
| waveform, e.g. [Wav2Vec2](./model_doc/wav2vec2) | `audio_values` | `(batch_size, num_samples)` |
| spectrogram, e.g. [Whisper](./model_doc/whisper) | `audio_features` | `(batch_size, num_mel_bins, num_frames)` |

Each comes with a matching `_mask` when `return_padding_mask` is enabled. `model_input_names` tells
you which keys a given processor emits:

```py
audio_processor.model_input_names
['audio_values']
```

## Loading

<hfoptions id="audio-processor-classes">
<hfoption id="AutoAudioProcessor">

The [AutoClass](./model_doc/auto) API loads the correct processor for a model.

```py
from transformers import AutoAudioProcessor

audio_processor = AutoAudioProcessor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
<hfoption id="model-specific class">

Each audio model has its own class, which reads its settings — mel bins, hop length, spectrogram
geometry — from [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json).

```py
from transformers import WhisperAudioProcessor

audio_processor = WhisperAudioProcessor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
</hfoptions>

Every audio processor has a torch and a numpy implementation. They share one configuration and
produce bit-identical output, so which one you get does not change your results.

## Padding and truncation

Models are batched, so sequences in a batch must be the same length. Real audio is not.

```py
short.shape, long.shape
((53248,), (86699,))
```

Padding and truncation control opposite ends of a waveform's length. Truncation removes samples from waveforms
that are too long; padding adds silence to waveforms that are too short. When both are enabled, the processor
truncates first and pads second.

```text
input waveforms
      │
      ├── truncation=True
      │       └── require max_length
      │               └── pad_to_multiple_of set? round max_length up
      │                       └── shorten items above the resolved length
      └── truncation=False ──► keep longer items unchanged
      │
      ▼
resolved padding strategy
      ├── True or "longest"
      │       └── target = longest remaining item
      │               └── pad_to_multiple_of set? round target up
      │                       └── pad shorter items to target
      ├── False or "do_not_pad" ─► do not pad
      └── "max_length"
              └── require max_length
                      └── pad_to_multiple_of set? round max_length up
                              └── pad shorter items to the resolved length
```

Audio processors pad by default. The base strategy, `padding=True` (equivalent to `"longest"`), pads to the
longest sequence in the batch. The padding value is silence. A model can configure a different default; inspect
`audio_processor.padding` to see the strategy in use. Omitting `padding` uses that configured value.

Prefer the explicit string strategies—`"longest"`, `"max_length"`, and `"do_not_pad"`—in new code. The boolean
forms are retained for compatibility: `True` means `"longest"`, and `False` means `"do_not_pad"`.

```py
audio_processor.padding
True

processed = audio_processor([long, short], sampling_rate=16000)
processed["audio_values"].shape
(2, 86699)
```

`truncation=True` always requires `max_length`. Padding alone never shortens a waveform, and truncation alone
never lengthens one. Their combinations behave as follows:

| `padding` | `truncation` | result |
|---|---|---|
| `True` or `"longest"` | `False` | Pad every item to the longest item in the batch. |
| `False` or `"do_not_pad"` | `False` | Keep every length unchanged; list items must already have compatible shapes. |
| `"max_length"` | `False` | Pad shorter items to `max_length`, but leave longer items unchanged. |
| `True` or `"longest"` | `True` | Truncate above `max_length`, then pad to the longest remaining item. |
| `False` or `"do_not_pad"` | `True` | Truncate above `max_length`, but leave shorter items unchanged. |
| `"max_length"` | `True` | Truncate longer items and pad shorter items so every item has exactly `max_length` samples. |

Disabling padding does not return a ragged batch. Audio processors produce dense batched tensors, so a list whose
items still have different shapes raises `ValueError` with a suggestion to enable padding. A dense array shaped
`(batch_size, num_samples)` is not a supported batch representation; pass a list of individual waveforms instead.

```py
# Pad to the longest waveform explicitly.
processed = audio_processor([long, short], sampling_rate=16000, padding="longest")
processed["audio_values"].shape
(2, 86699)

# Disable padding when the waveforms already have the same length.
processed = audio_processor([short, short], sampling_rate=16000, padding=False)
processed["audio_values"].shape
(2, 53248)

# Pad shorter waveforms to a fixed length.
processed = audio_processor([long, short], sampling_rate=16000, padding="max_length", max_length=90000)
processed["audio_values"].shape
(2, 90000)

# Produce an exact length by truncating and padding around the same boundary.
processed = audio_processor(
    [long, short],
    sampling_rate=16000,
    padding="max_length",
    truncation=True,
    max_length=50000,
)
processed["audio_values"].shape
(2, 50000)
```

Use `pad_to_multiple_of` to round the selected padding or truncation boundary up to a convenient multiple. With
`padding="longest"`, it rounds the longest waveform length; with `padding="max_length"`, it rounds `max_length`.
The truncation boundary is rounded up as well when truncation is enabled. This is useful for tensor shapes that
perform better on specific hardware.

```py
processed = audio_processor(
    [long, short], sampling_rate=16000, padding="longest", pad_to_multiple_of=1024
)
processed["audio_values"].shape
(2, 87040)
```

Pass `return_padding_mask=True` to get a mask marking which positions are real audio. For a
spectrogram model the mask is on the frame axis, not the sample axis — the processor converts between
them using the model's `hop_length`, so you do not have to.

```py
processed = audio_processor(
    [long, short], sampling_rate=16000, padding=True, return_padding_mask=True
)
processed["audio_values_mask"].shape
(2, 86699)
```

## Preprocessing a dataset

```py
from datasets import load_dataset, Audio
from transformers import AutoAudioProcessor

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base")


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    return audio_processor(audio_arrays, sampling_rate=16000, padding=True)


processed_dataset = preprocess_function(dataset[:5])
```

## Writing one

An audio processor declares its configuration and overrides only the steps that differ from the
shared pipeline. Both are covered in [Preprocessing](./preprocessing), which describes the pipeline,
the override points, and the rules every declared option follows.
