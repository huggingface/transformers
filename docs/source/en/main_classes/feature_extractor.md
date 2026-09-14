<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Feature extractor

> [!WARNING]
> Feature extractors are deprecated for audio. [`AudioProcessor`](./audio_processor) replaces them:
> `AutoFeatureExtractor` and the `XxxFeatureExtractor` classes still load and still work, with a
> deprecation warning. See [Audio processors](../audio_processors).
>
> The classes below remain documented for the image and legacy paths.

A feature extractor is in charge of preparing input features for audio models. This includes feature extraction from sequences, e.g., pre-processing audio files to generate Log-Mel Spectrogram features, and conversion to NumPy and PyTorch tensors.

## FeatureExtractionMixin

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin
    - from_pretrained
    - save_pretrained

## SequenceFeatureExtractor

[[autodoc]] SequenceFeatureExtractor
    - pad

## BatchFeature

[[autodoc]] BatchFeature

## ImageFeatureExtractionMixin

[[autodoc]] image_utils.ImageFeatureExtractionMixin
