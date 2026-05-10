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
*This model was released on 2025-12-16 and added to Hugging Face Transformers on 2025-12-16.*

# PE Audio-Visual (Perception Encoder Audio-Visual)

## Overview

Perception Encoder Audio-Visual (PE-AV) was proposed in [Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning](https://huggingface.co/papers/2512.19687) by Apoorv Vyas et al. It extends the Perception Encoder framework to multiple modalities (text, audio, and video).

PE-AV is a family of encoders trained on O(100M) audio-video pairs with synthetic captions, using ten pairwise contrastive objectives to align all three modalities in a shared embedding space.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/pe-av.png"
alt="PE-AV architecture" width="600"/>

<small>PE-AV architecture. Taken from the <a href="https://ai.meta.com/blog/sam-audio/">Meta AI blog post.</a></small>

Audio and video are processed by dedicated encoders and combined through an Audio-Visual Fusion Encoder, while captions pass through a separate Text Encoder. Each branch produces CLS embeddings: `CLS-A`, `CLS-V`, `CLS-AV` for the three modality views, and `CLS-AT`, `CLS-VT`, `CLS-AVT` for text-projection variants aligned to each target. The model is trained with a combination of single-modality and fused-modality alignment losses.

See the [PE-AV collection](https://huggingface.co/collections/facebook/perception-encoder-audio-visual) on the Hub for the full checkpoint family and usage examples.

## PeAudioVideoProcessor

[[autodoc]] PeAudioVideoProcessor
    - __call__

## PeAudioVideoConfig

[[autodoc]] PeAudioVideoConfig

## PeAudioVideoEncoderConfig

[[autodoc]] PeAudioVideoEncoderConfig

## PeAudioVideoModel

[[autodoc]] PeAudioVideoModel
    - forward

## PeAudioVideoEncoder

[[autodoc]] PeAudioVideoEncoder
    - forward
