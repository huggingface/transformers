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

# Audio processor

An audio processor prepares input features for audio models: resampling and mono conversion, padding
and truncation, and — for spectrogram models — the STFT and mel filterbank. See
[Audio processors](../audio_processors) for usage and [Preprocessing](../preprocessing) for the
contract these classes follow.

## PreprocessingMixin

Shared across audio, image and video processors: configuration, loading, saving, and option
resolution.

[[autodoc]] preprocessing_base.PreprocessingMixin
    - from_pretrained
    - save_pretrained

## BaseAudioProcessor

[[autodoc]] audio_processing_utils.BaseAudioProcessor
    - __call__
    - pad

## Backends

Each audio processor has a torch and a numpy implementation. They share one configuration and produce
bit-identical output for the same input.

[[autodoc]] audio_processing_backends.TorchAudioBackend

[[autodoc]] audio_processing_backends.NumpyAudioBackend

## Spectrogram configuration

The frozen dataclasses describing the STFT and mel stages. A model declares these as a plain dict;
they are coerced to these types when the processor is built.

[[autodoc]] audio_utils.SpectrogramConfig

[[autodoc]] audio_utils.StftConfig

[[autodoc]] audio_utils.MelScaleConfig

## BatchFeature

[[autodoc]] BatchFeature
