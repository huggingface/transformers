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

# Add audio processing components

Audio models require a feature extractor which is accessible behind the [`AutoFeatureExtractor`] entry point.

> [!NOTE]
> For the model and configuration steps, follow the [modular](./modular_transformers) guide first.

## Feature extractor

Add a feature extractor when the model consumes raw audio or audio-derived features.

Create `feature_extraction_<model_name>.py` in the model directory. Inherit from [`SequenceFeatureExtractor`] so the new class gets shared padding, truncation, saving, and loading behavior.

```py
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor


class MyModelFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(self, feature_size=80, sampling_rate=16000, padding_value=0.0, **kwargs):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    def __call__(self, raw_speech, sampling_rate=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"`sampling_rate` must be {self.sampling_rate}, but got {sampling_rate}.")

        # Convert raw_speech to model features here.
        ...
```

Keep the constructor small and serializable. Store every value needed to reproduce preprocessing as an instance attribute, and avoid storing runtime-only values such as open files, devices, or decoded audio arrays.

The `__call__` method must validate the input sampling rate when users pass `sampling_rate`. If the input rate differs from the model's expected rate, raise an error instead of silently resampling.

Save the feature extractor with the checkpoint by instantiating it in the conversion script and calling [`~FeatureExtractionMixin.save_pretrained`]. Do not manually create or edit preprocessing config files.

> [!TIP]
> See [`Gemma4AudioFeatureExtractor`] for reference.

## Register the classes

Expose the new classes from the model package `__init__.py`. Follow the lazy import pattern used by nearby models and guard imports with the same optional dependencies required by the class.

Map the new class to the model config so [`AutoFeatureExtractor`] can load it. Add an entry to `FEATURE_EXTRACTOR_MAPPING_NAMES` in `src/transformers/models/auto/feature_extraction_auto.py`, following the pattern of nearby entries. Then verify the model type appears there under `FEATURE_EXTRACTOR_MAPPING_NAMES` for [`AutoFeatureExtractor`].

- `FEATURE_EXTRACTOR_MAPPING_NAMES` for [`AutoFeatureExtractor`]

## Testing

Add tests for each audio processing component in the model test directory. Feature extractor tests usually live in `tests/models/<model_name>/test_feature_extraction_<model_name>.py`.

For feature extractors that inherit from [`SequenceFeatureExtractor`], inherit from [`SequenceFeatureExtractionTestMixin`]. The mixin covers save and load behavior, padding, truncation, tensor conversion, and common feature extractor properties. Provide a tester object with `prepare_feat_extract_dict()` and `prepare_inputs_for_common()` so the mixin can instantiate the feature extractor and build short dummy audio inputs.

```py
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin

class MyModelFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = MyModelFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = MyModelFeatureExtractionTester(self)
```

Add focused tests for model-specific behavior that the mixin doesn't know about. For audio feature extractors, that usually means checking the feature shape returned by `__call__`, validating that an incorrect `sampling_rate` raises an error, and checking any custom normalization or feature computation.

If the model also has a [`ProcessorMixin`] that wraps the feature extractor, add `tests/models/<model_name>/test_processing_<model_name>.py` and inherit from [`ProcessorTesterMixin`]. Set `processor_class` and override `_setup_<component>()` class methods for components that can't be constructed without arguments. Use `_setup_test_attributes()` to expose placeholder tokens used by the common processor tests.

```py
from ...test_processing_common import ProcessorTesterMixin

class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    @classmethod
    def _setup_feature_extractor(cls):
        return cls._get_component_class_from_processor("feature_extractor")(sampling_rate=16000)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.audio_token = getattr(processor, "audio_token", "")
```

## Next steps

- Read the [Auto-generating docstrings](./auto_docstring) guide to auto-generate consistent docstrings with `@auto_docstring`.
- Read the [Feature extractors](./feature_extractors) guide for user-facing preprocessing behavior.
