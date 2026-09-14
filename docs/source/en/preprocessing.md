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

# Preprocessing

Preprocessors turn raw modality inputs — images, audio, video — into the tensors a model's `forward`
accepts.

```python
inputs = audio_processor(audio, sampling_rate=16000)
inputs = image_processor(images, do_resize=False)
```

Audio processors replace the legacy feature extractors; they all derive from
[`~audio_processing_utils.BaseAudioProcessor`]. `PreprocessingMixin` shares
configuration, loading, saving and argument resolution across every modality, so the rules below hold
for image and video processors too except where noted.

Model files are deliberately terse: an audio processor declares its configuration and overrides only
the steps that differ from the shared pipeline. The pipeline itself is explained once, here.

## The audio pipeline

An audio processor is a fixed sequence of steps with named override points. A model changes behaviour
by overriding a step, never by rewriting the sequence.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/audio-processor-call-flow.png" alt="AudioProcessor call flow, from __call__ to BatchFeature"/>
</div>

Three properties of that flow are worth stating outright, because they are what let model files stay
short.

**Options are resolved once, before any audio is touched.** `preprocess` fills every declared option
from the instance, validates the merged set, canonicalises it, and only then dispatches. Each step
below receives its options as named parameters. A step that reads `self.<option>` for a declared
option is a bug: it means the caller's value was computed, validated and then discarded.

**The per-utterance boundary sits above the branch.** `_downmix_to_mono` and `_resample` run once per
clip, on one waveform at a time. Everything after the branch is either explicitly per-clip or
explicitly batched, and a step written for one will not work in the other.

**The two paths differ on one question:** whether padding happens before or after feature extraction.
A model whose filterbank must see unpadded audio sets `do_batch_spectrogram = False` and takes the
per-utterance path; everything else pads first and extracts on the batch, which is faster.

### The spectrogram core

`compute_features` is where a mel spectrogram is produced. Its shape is driven entirely by
[`~audio_utils.SpectrogramConfig`], so most models configure it rather than override it.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/audio-processor-spectrogram-core.png" alt="The spectrogram core, expanding compute_features"/>
</div>

The manual-framing branch exists because a native STFT frames on `n_fft`. Per-frame preemphasis, DC
offset removal, an extended frame, or a window left-aligned inside the FFT buffer all require framing
on `win_length`, which only the manual path can do. The choice is derived from the configuration, not
declared: you get the manual path by setting one of those fields.

Two configuration fields describe framing geometry rather than a transform, and are easiest to read
next to the figure:

- `extra_samples_per_frame` cuts each frame that many samples longer, so per-frame preemphasis has a
  sample to look back at.
- `count_frames_by_hop` counts frames as `ceil(samples / hop_length)` instead of deriving them from
  the window geometry — some extractors report the shorter count, and the padding mask has to agree
  with whichever the model expects.

### Which models override what

Most models need no code at all: **9 of the 33 audio processors override nothing**, and are a
configuration block and nothing else — including [Whisper](./model_doc/whisper), [Gemma3n](./model_doc/gemma3n),
[EnCodec](./model_doc/encodec), [DAC](./model_doc/dac), [Dia](./model_doc/dia), [SpeechT5](./model_doc/speecht5),
LASR, PE-Audio and [Pop2Piano](./model_doc/pop2piano).

The remaining 24 override a hook or two. The table below is the whole surface in use — if you are
changing a hook, this is who else depends on its behaviour.

| hook | models that override it |
|---|---|
| `_finalize_output` | 17 — AST, CLAP, Cohere-ASR, Fun-ASR-Nano, Gemma4, Granite-Speech, Granite-Speech5, Inkling, Kyutai-STT, Nemotron-ASR-Streaming, NeuCodec, Parakeet, Phi4-Multimodal, Qwen3-ASR, SeamlessM4T, Speech2Text, XCodec2 |
| `compute_features` | CLAP, Gemma4-Unified, Granite-Speech, Musicgen-Melody, SeamlessM4T |
| `_downmix_to_mono` | NeuCodec, Qwen3-ASR, VibeVoice, Wav2Vec2, XCodec2 |
| `_log_compress` | CLVP, UnivNet, Voxtral-Realtime |
| `_padded_frame_count` | Inkling, Qwen3-ASR, UnivNet |
| `_finalize_features` | Fun-ASR-Nano, SeamlessM4T |
| `_waveform_to_spectrum` | Inkling, UnivNet |
| `_spectrum_magnitude` | Inkling, UnivNet |
| `_pad_feature_single` | NeuCodec, XCodec2 |
| `_project_to_mel` · `_valid_frame_counts` | UnivNet |
| `_process_frames` · `_stft_framed` | Phi4-Multimodal |
| `_pad_features` | AST |
| `pad` · `_pad_waveform` · `_truncate_waveform` · `_stack_waveforms` · `_resolve_padding_strategy` · `_set_attributes` | CLAP |
| `_dither_waveform` · `_preprocess_audio_like_inputs` | Cohere-ASR |
| `_preprocess` | NeuCodec |

Two patterns are worth reading off it. `_finalize_output` dominates because most model-specific work
is *after* the features exist — an extra output key, a per-utterance normalisation, a reshape for the
encoder. And the two models with the widest surface are the two with genuinely unusual pipelines:
CLAP, whose fusion mode crops the mel rather than the waveform, and UnivNet, which is a vocoder and
runs the spectrogram in float64 throughout.

## Configuration files

`preprocessor_config.json` stores a standalone preprocessor's settings: the defaults for the options
it accepts, plus metadata identifying its class. It stores neither the input nor the output.

```python
from transformers import NemotronAsrStreamingAudioProcessor

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
audio_processor = NemotronAsrStreamingAudioProcessor.from_pretrained(
    model_id, return_padding_mask=True
)
audio_processor.save_pretrained("./audio-preprocessor")
audio_processor = NemotronAsrStreamingAudioProcessor.from_pretrained("./audio-preprocessor")
```

An excerpt from the saved `preprocessor_config.json`:

```json
{
  "audio_processor_type": "NemotronAsrStreamingAudioProcessor",
  "sampling_rate": 16000,
  "return_padding_mask": true
}
```

`processor_config.json` stores a composite processor's own settings and its nested preprocessor
configurations. Each nested configuration follows the same rules as a standalone one. Tokenizers and
chat templates are saved separately.

```json
{
  "processor_class": "Nemotron3_5AsrProcessor",
  "audio_processor": {
    "audio_processor_type": "NemotronAsrStreamingAudioProcessor",
    "sampling_rate": 16000,
    "return_padding_mask": true
  }
}
```

Audio composite processors use `audio_processor` as their component name. Configurations using the
legacy `feature_extractor` key are still accepted when loading; saving writes `audio_processor`. The
old constructor keyword and attribute remain available with a deprecation warning. If both keys are
present, `audio_processor` wins.

These files store instance settings. An override passed for a single call is never saved.

## Supported options: `valid_kwargs`

`valid_kwargs` declares the supported options and their types, as `ImagesKwargs`, `AudioKwargs`, or a
model-specific extension of one.

It is the boundary in both directions. Every option in it is materialised onto the instance and
serialised; nothing outside it reaches a saved configuration. A key that a configuration carries but
the class does not declare is still set on the instance — remote code and legacy spellings keep
working — but it is not written back out, so it cannot become a permanent field of every checkpoint
saved downstream. Loading metadata such as `audio_processor_type` is exempt.

Defaults live on the class, in its implementation file:

```python
from transformers.audio_processing_backends import TorchAudioBackend
from transformers.processing_utils import AudioKwargs, Unpack


class MyAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    gain (`float`, *optional*, defaults to 1.0):
        Multiplier applied to the audio samples.
    """

    gain: float


class MyAudioProcessor(TorchAudioBackend):
    valid_kwargs = MyAudioProcessorKwargs
    sampling_rate = 16000
    gain = 1.0

    def __init__(self, **kwargs: Unpack[MyAudioProcessorKwargs]):
        super().__init__(**kwargs)

    def _downmix_to_mono(self, audio_el, *, gain, **kwargs):
        return super()._downmix_to_mono(audio_el, **kwargs) * gain
```

Document custom parameters on the `TypedDict`, assign `valid_kwargs`, and annotate the constructor
with `Unpack` — the same pattern as [image processors](./image_processors). `total=False` allows keys
to be omitted.

Declaring an option is not enough. **The step that acts on it must name it as a parameter**, as
`_downmix_to_mono` does above. The parameter is un-defaulted on purpose: you cannot override the step
without typing the option's name, so an option cannot quietly stop being read. `preprocess`
guarantees the value is always supplied, so no fallback is needed and none should be written.

Option kinds:

- Boolean flags such as `do_normalize`: `True` enables an operation, `False` disables it.
- Numbers such as `sampling_rate` or `gain`, with documented ranges where they matter.
- Strings with restricted choices, documented and validated.
- Structured values such as a size dictionary or a [`~audio_utils.SpectrogramConfig`].

## Defaults and overrides

Explicit initialization or loading arguments beat the saved configuration, which beats the class
defaults. At call time, arguments beat the instance settings — for that call only.

```text
call kwargs > instance settings
              explicit init/loading kwargs > saved config > class defaults
```

```python
# The saved config above has return_padding_mask=True.
audio_processor = NemotronAsrStreamingAudioProcessor.from_pretrained(
    "./audio-preprocessor", return_padding_mask=False
)
assert audio_processor.return_padding_mask is False

# Enable the mask for this call only.
inputs = audio_processor(audio, sampling_rate=16000, return_padding_mask=True)
assert audio_processor.return_padding_mask is False

# The next call uses the instance setting again: no padding mask.
inputs = audio_processor(audio, sampling_rate=16000)
```

Composite processors add a layer above this. `ModelProcessorKwargs`, a subclass of `ProcessingKwargs`,
declares modality defaults in `_defaults`; the composite merges those with the call arguments before
forwarding them, and the precedence above then applies to what each preprocessor receives.

```python
inputs = processor(audio=audio, sampling_rate=16000, audio_kwargs={"return_padding_mask": True})
```

See [Processing kwargs](./main_classes/processors#processing-kwargs).

## What `None` means

`None` means *not provided*, in every direction. Passing it is the same as omitting the option: the
value falls back to the instance setting at call time, and to the next available default at
initialization.

```python
audio_processor.return_padding_mask = True
inputs = audio_processor(audio, sampling_rate=16000, return_padding_mask=None)   # mask stays enabled
inputs = audio_processor(audio, sampling_rate=16000, return_padding_mask=False)  # explicitly disabled
```

This is what makes the common forwarding wrapper behave:

```python
def transcribe(audio, mask=None):
    return audio_processor(audio, sampling_rate=16000, return_padding_mask=mask)
```

`None` therefore never reaches an operation as a false-like value, and a boolean flag is only ever
`True` or `False` by the time a step reads it. Explicit `False` and `0` are values, not omissions.

The cost is that an option's default cannot be overridden *to* `None` for a single call. An explicit
`None` at initialization is kept and survives a save/load round trip; per call, it resolves to the
instance value.

`sampling_rate` is worth calling out because the name carries two meanings. `self.sampling_rate` is
the model's native rate, part of the processor's identity and fixed at initialization. The
`sampling_rate` argument is the caller's assertion about the rate of the arrays being passed. A
mismatch resamples to the native rate with a warning; audio referenced by URL or path is decoded at
the native rate, so an argument given alongside it is ignored, also with a warning.

## Backends

Each audio processor has a torch and a numpy implementation, `TorchAudioBackend` and
`NumpyAudioBackend`, sharing one configuration. They are required to produce bit-identical output for
the same input, so a model's behaviour does not depend on which is installed.
