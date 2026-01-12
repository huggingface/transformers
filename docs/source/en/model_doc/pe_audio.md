<!--Copyright 2025 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-16.*

# PE Audio (Perception Encoder Audio)

## Overview

PE Audio (Perception Encoder Audio) is a state-of-the-art multimodal model that embeds audio and text into a shared (joint) embedding space.
The model enables cross-modal retrieval and understanding between audio and text.

**Text input**

- Produces a single embedding representing the full text.

**Audio input**

- **PeAudioFrameLevelModel**
  - Produces a sequence of embeddings, one every 40 ms of audio.
  - Suitable for audio event localization and fine-grained temporal analysis.
- **PeAudioModel**
  - Produces a single embedding for the entire audio clip.
  - Suitable for global audio-text retrieval tasks.

**The resulting embeddings can be used for:**

- Audio event localization
- Cross-modal (audio–text) retrieval and matching

## Usage

### Basic usage

```py
TODO
```

## PeAudioFeatureExtractor

[[autodoc]] PeAudioFeatureExtractor
    - __call__

## PeAudioProcessor

[[autodoc]] PeAudioProcessor
    - __call__

## PeAudioConfig

[[autodoc]] PeAudioConfig

## PeAudioEncoderConfig

[[autodoc]] PeAudioEncoderConfig

## PeAudioEncoder

[[autodoc]] PeAudioEncoder
    - forward

## PeAudioFrameLevelModel

[[autodoc]] PeAudioFrameLevelModel
    - forward

## PeAudioModel

[[autodoc]] PeAudioModel
    - forward
