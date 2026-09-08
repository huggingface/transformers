# Preprocessing

Draft API contract for this PR. The implementation gaps listed below still need to be resolved.

Preprocessors prepare modality-specific inputs (images, audio, or video) for a model:

```python
inputs = audio_processor(audio, sampling_rate=16000)
inputs = image_processor(images, do_resize=False)
```

`AudioProcessor` replaces the legacy feature extractors. `PreprocessingMixin` shares configuration, loading, saving, and argument handling across preprocessors.

## Configuration files

`preprocessor_config.json` stores a standalone preprocessor's settings: defaults for the options passed when calling it, plus metadata identifying its class. It does not store the input data or the processed output.

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

`processor_config.json` stores a composite processor's own settings and its nested preprocessor configurations. Each nested configuration follows the same rules as a standalone preprocessor configuration. Tokenizers and chat templates are saved separately.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained("./processor")
```

For example, a composite configuration may contain the following excerpt:

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

Audio composite processors use `audio_processor` as their component name. Legacy configurations using `feature_extractor` are accepted when loading; saving writes `audio_processor`. The old constructor keyword and attribute remain available with a deprecation warning. If both config keys are present, `audio_processor` takes precedence. These files store instance settings. An override used for a single call is not saved.

## Supported options: `valid_kwargs`

`valid_kwargs` declares the supported preprocessing options and their types. It uses `ImagesKwargs`, `AudioKwargs`, or a model-specific extension.

This PR proposes an additional guarantee: every preprocessing setting in the configuration must belong to this schema. Loading metadata such as `audio_processor_type` is exempt. Existing documentation describes supported arguments without defining this serialization boundary.

Defaults live on the preprocessor class, in its implementation file. For example:

```python
from transformers.audio_processing_backends import TorchAudioBackend
from transformers.processing_utils import AudioKwargs, Unpack
from transformers.utils import auto_docstring


class MyAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    gain (`float`, *optional*, defaults to `self.gain`):
        Multiplier applied to the audio samples.
    """

    gain: float


@auto_docstring
class MyAudioProcessor(TorchAudioBackend):
    valid_kwargs = MyAudioProcessorKwargs
    sampling_rate = 16000
    gain = 1.0

    def __init__(self, **kwargs: Unpack[MyAudioProcessorKwargs]):
        super().__init__(**kwargs)

    # Processing implementation omitted: it must apply the resolved gain.
```

This follows the existing [image processor documentation pattern](docs/source/en/auto_docstring.md): document custom parameters on the TypedDict, assign `valid_kwargs`, and annotate the constructor with `Unpack`. `total=False` allows keys to be omitted; accepting `None` requires a separate type annotation and defined behavior.

The intended contract is that these options are accepted both at initialization and when calling the preprocessor. Declaring an option is not enough: the processing implementation must use its resolved value. Helper methods may have their own signatures.

Options include:

- Boolean flags, such as `do_resize`: `True` enables an operation and `False` disables it.
- Numbers, such as `sampling_rate` or `rescale_factor`, with documented ranges where needed.
- Strings: restricted choices should use `Literal` or an enum and document the accepted values.
- Structured values, such as a size dictionary or spectrogram configuration, with a defined schema.

`None` is valid only when the option explicitly defines its meaning. For example, `return_tensors=None` means no conversion to a tensor framework.

## Defaults and overrides

For a preprocessor called directly, explicit initialization/loading kwargs override saved settings, which override class defaults. At call time, explicit kwargs override the instance settings for that call only:

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

# The next call uses the instance default again: no padding mask.
inputs = audio_processor(audio, sampling_rate=16000)
```

Omitting an option uses its default. Explicit `False` and `0` are values, not missing options.

Composite processors add another layer. `ModelProcessorKwargs`, a subclass of `ProcessingKwargs`, can declare modality-specific defaults in `_defaults`. The composite processor merges these with call-time kwargs before forwarding options to each preprocessor. The precedence above applies after that merge, to the arguments the preprocessor receives.

```python
# Recommended composite-call syntax: group options by modality.
inputs = processor(audio=audio, sampling_rate=16000, audio_kwargs={"return_padding_mask": True})
```

These processor-level `_defaults` are defined in code; they are distinct from the preprocessor's class attributes and saved instance settings. See [Processing kwargs](docs/source/en/main_classes/processors.md#processing-kwargs).

## Boolean flags and `None`

This PR proposes a stricter rule for boolean operation flags: they must be `True` or `False`. Existing `ImagesKwargs` annotations explicitly accept `bool | None`, so this is an API change, not a restatement of the current contract.

Passing `None` will be deprecated. During the transition, log a deprecation warning and treat it as an omitted option: use the instance default at call time, or the next available default during initialization/loading.

```python
# Intended behavior during the deprecation period:
image_processor.do_resize = True
inputs = image_processor(images, do_resize=None)   # Warns; resizing stays enabled.
inputs = image_processor(images, do_resize=False)  # Explicitly disables resizing.
assert image_processor.do_resize is True
```

This fallback applies to boolean flags. It must not discard a meaningful `None` for another option.

## Implementation gaps

- Audio currently limits call-time options through `per_call_kwargs`; some `valid_kwargs` are initialization-only. Supporting every declared option at call time requires updating the processing implementations.
- Boolean `None` needs the warning and default fallback described above. It must not reach an operation as a false-like value.
- Initialization currently treats `None` as missing for every declared option. Options with a meaningful `None` need to distinguish it from omission.
- Serialization currently starts from instance attributes. The schema boundary above needs enforcement so internal state does not become preprocessing configuration.
