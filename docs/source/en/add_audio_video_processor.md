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

# Add audio or video processors

Audio models require a feature extractor and video models require a video processor. Multimodal models use a processor that wraps some combination of a tokenizer, feature extractor, image processor, or video processor behind one [`AutoProcessor`] entry point.

> [!NOTE]
> For the model and configuration steps, follow the [modular](./modular_transformers) guide first.

## Feature extractor

Add a feature extractor when the model consumes raw audio or audio-derived features. Feature extractors inherit from [`SequenceFeatureExtractor`], save their configuration to `preprocessor_config.json`, and load through [`AutoFeatureExtractor`].

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

> [!TIP]
> See [`Gemma4AudioFeatureExtractor`] for reference.

## Video processor

Add a video processor when the model consumes videos or sampled video frames. Video processors inherit from [`BaseVideoProcessor`], save their configuration to `video_preprocessor_config.json`, and load through [`AutoVideoProcessor`].

Create `video_processing_<model_name>.py` in the model directory. Inherit from [`BaseVideoProcessor`] so the new class gets shared decoding, frame sampling, resizing, rescaling, normalization, saving, and loading behavior.

```py
from ...video_processing_utils import BaseVideoProcessor

class MyModelVideoProcessor(BaseVideoProcessor):
    # Saved defaults.
    size = {"shortest_edge": 224}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_normalize = True
    do_sample_frames = True
    num_frames = 16
    model_input_names = ["pixel_values_videos"]
```

The class attributes are the saved defaults. Users can override them at initialization or call time. Use the same names as [`VideosKwargs`] when possible, such as `size`, `crop_size`, `do_resize`, `do_sample_frames`, `num_frames`, and `fps`.

Override [`~BaseVideoProcessor.sample_frames`] only when the model requires a sampling rule that the base uniform sampler can't express. For example, some models enforce a minimum or maximum number of frames, or sample based on model-specific constraints.

If the model's forward method expects a legacy input name, override `preprocess` and rename the key after calling the base implementation.

```py
class MyModelVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        # Rename the key only when the model expects a legacy input name.
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch
```

> [!TIP]
> See [`Qwen3VLVideoProcessor`] for reference.

## Multimodal processor

Add a model processor when users need one object that combines multiple preprocessing components. A processor inherits from [`ProcessorMixin`] and usually wraps a tokenizer plus one or more processors. The processor loads through [`AutoProcessor`].

Create `processing_<model_name>.py` when users need one object for more than one preprocessing component. The processor holds the cross-modal logic, such as expanding chat templates, inserting audio or video placeholders, or merging tokenizer outputs with media tensors.

```py
from ...processing_utils import ProcessorMixin


class MyModelProcessor(ProcessorMixin):
    # Components saved and loaded by ProcessorMixin.
    feature_extractor_class = "MyModelFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "MyModelVideoProcessor"

    def __init__(self, feature_extractor=None, tokenizer=None, video_processor=None, **kwargs):
        super().__init__(feature_extractor, tokenizer, video_processor, **kwargs)
```

Keep modality-specific tensor creation in the feature extractor or video processor. Use the processor for orchestration across components.

> [!TIP]
> See [`Gemma4Processor`] and [`Qwen3OmniMoeProcessor`] for reference.

## Register the classes

Expose the new classes from the model package `__init__.py`. Follow the lazy import pattern used by nearby models and guard imports with the same optional dependencies required by the class.

Map the new classes to the model config so the `Auto` classes can load them. The generated auto mapping file has a warning at the top. Do not edit it by hand. Add or update the model config, then run:

```bash
python utils/check_auto.py --fix_and_overwrite
```

After the mapping is generated, verify the model type appears in the relevant mapping:

- `FEATURE_EXTRACTOR_MAPPING_NAMES` for [`AutoFeatureExtractor`]
- `VIDEO_PROCESSOR_MAPPING_NAMES` for [`AutoVideoProcessor`]
- `PROCESSOR_MAPPING_NAMES` for [`AutoProcessor`]

## Next steps

- Read the [Auto-generating docstrings](./auto_docstring) guide to auto-generate consistent docstrings with `@auto_docstring`.
- Read the [Writing model tests](./testing#preprocessing-component-tests) guide to test feature extractors, video processors, and model processors.
