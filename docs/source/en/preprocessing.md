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

Audio processors either customize the default workflow or own their processing order. Both reuse
input preparation, configuration handling and numerical backend operations.

## Choosing an audio integration

Choose execution and authoring independently:

| Processing requirements | Integration | Reference |
|---|---|---|
| Default order, different numerical parameters | Declare a spectrogram configuration | `models/whisper/audio_processing_whisper.py` |
| Default order, different individual operation | Override a meaningful hook | `models/wav2vec2/audio_processing_wav2vec2.py` |
| Different order, branching, or coordination between stages | Override `_preprocess` and call backend operations | `models/clap/audio_processing_clap.py` |
| A variation of an existing model family | Author the differences in modular, then generate | `models/neucodec/modular_neucodec.py` |

Modular is an authoring choice for either execution style. Numerical backends remain shared at
runtime. A new model-specific recipe does not require adding flags or hooks to the universal
workflow. Prefer a main-method override when several hooks would have to coordinate through
instance state or compensate for earlier stages.

### Declare the workflow's options

Audio processors extend the full `AudioKwargs` schema, including when they override
`_preprocess`. This follows image processors, which inherit `ImagesKwargs` for custom workflows.
Model-specific kwargs classes add their own options; validators check constraints required by
the workflow. Legacy configuration fields can be mapped or explicitly dropped through
`legacy_field_mapping`.

A shared schema may include options a custom workflow does not consume. Whether to narrow those
schemas or change how unused options are handled is an unresolved cross-modality API question,
separate from choosing hooks versus a main-method override. The schema controls defaults and
serialization; read the workflow to establish which operations it performs.

Read the current declarations in `processing_utils.py` and the model's kwargs class. For an
instantiated processor, `processor.valid_kwargs.__annotations__` includes inherited names; it is
more reliable than maintaining a separate list of supported options. Resolved options must reach
the methods that consume them as named parameters. Read call options from those parameters;
`self.sampling_rate` remains the processor's native-rate identity.

### Keep operation contracts explicit

A custom `_preprocess` receives a list of mono waveforms already converted to its backend and
resampled to the processor's native rate, plus resolved call options. It owns cropping, padding,
feature extraction order and output assembly. Return `BatchFeature` with the requested tensor
conversion and the keys expected by the model.

Reuse `compute_features` for STFT/mel/log processing, or narrower backend operations when the
recipe needs them. Check each operation's docstring and the spectrogram configuration for axis
order and dtype. Keep waveform lengths in samples distinct from feature lengths in frames.
Return per-call metadata alongside the features it describes. Numerical caches are internal
implementation state and must stay out of saved configuration.

The common hooks remain appropriate for independent changes to the default order. Their
contracts specify whether data is per waveform, per frame, or padded and batched; choose the hook
by those conditions rather than by a similar-sounding name.

### Author related models with modular

NeuCodec and Nemotron streaming audio processors are generated from their existing modular
files, alongside their models and configurations. Edit that source and regenerate with the existing modular converter; inspect all
resulting files. The generated audio classes reuse the core backends without runtime inheritance
from XCodec2 or Parakeet. Read [the modular guide](./modular_transformers) for converter conventions.

The converter routes `AudioProcessor`, `AudioProcessorMixin` and `AudioProcessorKwargs` to
`audio_processing_<model>.py`, and `AudioProcessorNumpy` to `audio_processing_numpy_<model>.py`.
Keep optional dependency imports safe: the NumPy sibling must remain importable when Torch is
unavailable. Modular expands model-family inheritance; imports of shared runtime backends remain
imports. When a parent model does not define a method itself, an explicit call to the shared
implementation can express delegation without asking the converter to expand an absent method
(see Nemotron streaming's `_validate_preprocess_kwargs`).

### Verify the chosen integration

Test through the processor call: numerical compatibility, metadata and feature alignment,
non-default per-call options, and save/load behavior. Exercise both backends and the cases that
change workflow order, such as long versus short inputs. For modular sources, also verify
regeneration and optional-dependency imports. `tests/repo_utils/test_audio_modular_conversion.py`
checks those properties for both families. Follow the model's established numerical
comparison policy; generation does not change that policy.

## The default audio pipeline

The default workflow is a fixed sequence with named override points. Models using it override
individual steps; models owning their sequence override `_preprocess`.

In both figures the highlighted steps are the override points, each labelled with the models that
actually override it — so the diagram shows not just where you *may* intervene, but who does.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/audio-processor-call-flow.png" alt="AudioProcessor call flow, from __call__ to BatchFeature"/>
</div>

Three properties of that flow are worth stating outright, because they are what let model files stay
short.

**Options are resolved once, before any audio is touched.** `preprocess` fills every declared option
from the instance, validates the merged set, canonicalises it, and only then dispatches. Each step
below receives its options as named parameters. A step that reads `self.<option>` for a declared
option is a bug: it means the caller's value was computed, validated and then discarded.

**The per-utterance boundary sits above the branch.** `_downmix_to_mono`, `_prepare_waveform` and
`_resample` run once per clip, on one waveform at a time. Everything after the branch is either explicitly per-clip or
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

### Examples to read

Use these references to choose an integration; each model's source and `valid_kwargs` define its
current behavior. The pipeline figures above illustrate the shared workflow; their model labels
may reflect older integrations.

| Requirement | Example |
|---|---|
| Configuration alone | Whisper, Gemma3n, EnCodec |
| Independent waveform preparation | Wav2Vec2, VibeVoice |
| Independent normalization or numerical operation | Parakeet, CLVP, UnivNet, Voxtral Realtime |
| Independent output metadata, masking or frame grouping | Phi4 Multimodal, Gemma4, Kyutai, SeamlessM4T |
| Chunk before delegating to the default workflow | Cohere-ASR `_preprocess` |
| Branch between crop and fusion | CLAP `_preprocess` |
| Frame raw audio into tokens | Gemma4 Unified `_preprocess` |
| Coordinate mel extraction, frame grouping and model masks | Granite Speech and Granite Speech 5 `_preprocess` |
| Resolve a chroma recipe for the current call | MusicGen Melody `_preprocess` |
| Assemble acoustic and semantic inputs together | XCodec2 `_preprocess`; NeuCodec modular source |
| Reuse a related model's configuration with a different output hook | Nemotron streaming modular source |

Custom workflows document their length units and required outputs. Gemma4 Unified pads in tokens.
MusicGen Melody pads in samples, with `chunk_length` supplying a default maximum in seconds.
XCodec2 and NeuCodec preserve the legacy dual meaning of `max_length`: samples for the acoustic
branch, frames for semantic padding. Their semantic mask is mandatory; `return_padding_mask`
controls only the acoustic mask. Granite's masks describe model-specific frame or projector
geometry and are always returned.

CLAP expresses its processing order in one `_preprocess` workflow: `rand_trunc` crops waveforms
and returns one mel view; `fusion` keeps long waveforms and returns four mel views. Both modes fill
short waveforms to `max_length` using `padding_mode="repeatpad"`, `"repeat"`, or `"pad"`. A
per-call `truncation_mode` selects the matching mel filter bank. Each clip returns its views and
`is_longer` together; the workflow keeps no temporary metadata on the processor.

CLAP always returns mel views and `is_longer`. Its custom workflow consumes `max_length` in
waveform samples, `padding_mode` for short clips, and `truncation_mode` for long clips. Legacy
fill-method spellings passed as `padding` remain accepted. It still inherits the generic
`AudioKwargs` options; those declarations do not add raw output, generic padding or mask
operations to CLAP's workflow.

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

    def _prepare_waveform(self, audio_el, *, gain, **kwargs):
        return audio_el * gain
```

Document custom parameters on the `TypedDict`, assign `valid_kwargs`, and annotate the constructor
with `Unpack` — the same pattern as [image processors](./image_processors). `total=False` allows keys
to be omitted.

Declaring an option is not enough. **The step that acts on it must name it as a parameter**, as
`_prepare_waveform` does above. The parameter is un-defaulted on purpose: you cannot override the step
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

Audio processors provide paired implementations sharing one configuration and workflow. Most use
`TorchAudioBackend` and `NumpyAudioBackend`; DAC intentionally uses NumPy for both entry points.
Preserve each model's tested numerical contract: some legacy accumulation orders require exact
comparisons, while cross-backend tests use model-specific tolerances. CLAP's NumPy fusion path
currently uses Torch interpolation, so a NumPy class name alone does not guarantee Torch-free
execution.
