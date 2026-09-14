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

## Padding

Models are batched, so sequences in a batch must be the same length. Real audio is not.

```py
short.shape, long.shape
((53248,), (86699,))
```

Set `padding=True` to pad to the longest sequence in the batch. The padding value is silence.

```py
processed = audio_processor([long, short], sampling_rate=16000, padding=True)
processed["audio_values"].shape
(2, 86699)
```

Pass `return_padding_mask=True` to get a mask marking which positions are real audio. For a
spectrogram model the mask is on the frame axis, not the sample axis — the processor converts between
them using the model's `hop_length`, so you do not have to.

## Truncation

Set `truncation=True` with a `max_length` to cut sequences that are too long for the model.

```py
processed = audio_processor(
    [long, short],
    sampling_rate=16000,
    max_length=50000,
    truncation=True,
    padding="max_length",
)
processed["audio_values"].shape
(2, 50000)
```

`truncation=True` requires `max_length`; passing it alone raises.

## Resampling

The `sampling_rate` argument states the rate of the arrays you are passing. If it differs from the
model's native rate, the processor resamples for you and warns that it did.

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
> will be processed as though it were correct. Always pass it.

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
