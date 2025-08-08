<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# M-CTC-T

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The M-CTC-T model was proposed in [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://huggingface.co/papers/2111.00161) by Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve, and Ronan Collobert. The model is a 1B-param transformer encoder, with a CTC head over 8065 character labels and a language identification head over 60 language ID labels. It is trained on Common Voice (version 6.1, December 2020 release) and VoxPopuli. After training on Common Voice and VoxPopuli, the model is trained on Common Voice only. The labels are unnormalized character-level transcripts (punctuation and capitalization are not removed). The model takes as input Mel filterbank features from a 16Khz audio signal.

The abstract from the paper is the following:

*Semi-supervised learning through pseudo-labeling has become a staple of state-of-the-art monolingual
speech recognition systems. In this work, we extend pseudo-labeling to massively multilingual speech
recognition with 60 languages. We propose a simple pseudo-labeling recipe that works well even
with low-resource languages: train a supervised multilingual model, fine-tune it with semi-supervised
learning on a target language, generate pseudo-labels for that language, and train a final model using
pseudo-labels for all languages, either from scratch or by fine-tuning. Experiments on the labeled
Common Voice and unlabeled VoxPopuli datasets show that our recipe can yield a model with better
performance for many languages that also transfers well to LibriSpeech.*

This model was contributed by [cwkeam](https://huggingface.co/cwkeam). The original code can be found [here](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl).

## Usage tips

The PyTorch version of this model is only available in torch 1.9 and higher.

## Resources

- [Automatic speech recognition task guide](../tasks/asr)

## MCTCTConfig

[[autodoc]] MCTCTConfig

## MCTCTFeatureExtractor

[[autodoc]] MCTCTFeatureExtractor
    - __call__

## MCTCTProcessor

[[autodoc]] MCTCTProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## MCTCTModel

[[autodoc]] MCTCTModel
    - forward

## MCTCTForCTC

[[autodoc]] MCTCTForCTC
    - forward
