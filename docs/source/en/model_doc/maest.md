<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MAEST

## Overview

The MAEST model was proposed in [Efficient Supervised Training of Audio Transformers for Music
Representation Learning](https://arxiv.org/abs/2309.16418) by Pablo Alonso Jiménez, Xavier Serra, and Dimitry Bogdanov.
MAEST is a family of Transformer models based on [PASST](https://github.com/kkoutini/PaSST) (or AST) pre-trained in music style labels and intended to procude semantic music embeddigns.

The abstract from the paper is the following:

*In this work, we address music representation learning using convolution-free transformers. We
build on top of existing spectrogram-based audio transformers such as AST and train our models on a
supervised task using patchout training similar to PaSST. In contrast to previous works, we study
how specific design decisions affect downstream music tagging tasks instead of focusing on the
training task. We assess the impact of initializing the models with different pre-trained weights,
using various input audio segment lengths, using learned representations from different blocks and
tokens of the transformer for downstream tasks, and applying patchout at inference to speed up
feature extraction. We find that 1) initializing the model from ImageNet or AudioSet weights and
using longer input segments are beneficial both for the training and downstream tasks, 2) the best
representations for the considered downstream tasks are located in the middle blocks of the
transformer, and 3) using patchout at inference allows faster processing than our convolutional
baselines while maintaining superior performance. The resulting models, MAEST, are publicly
available and obtain the best performance among open models in music tagging tasks.*

This model was contributed by [p-alonso](https://huggingface.co/p-alonso).
The original code can be found [here](https://github.com/palonso/MAEST).

Sice MAEST features the same architecture as AST, the [ASTConfig](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTConfig), [ASTModel](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTModel), and [ASTForAudioClassification](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification) classes are used for inference.

## MAESTFeatureExtractor

[[autodoc]] MAESTFeatureExtractor
    - __call__
