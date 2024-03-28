<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# WavLM

## Overview

The WavLM model was proposed in [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) by Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen,
Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu,
Michael Zeng, Furu Wei.

The abstract from the paper is the following:

*Self-supervised learning (SSL) achieves great success in speech recognition, while limited exploration has been
attempted for other speech processing tasks. As speech signal contains multi-faceted information including speaker
identity, paralinguistics, spoken content, etc., learning universal representations for all speech tasks is
challenging. In this paper, we propose a new pre-trained model, WavLM, to solve full-stack downstream speech tasks.
WavLM is built based on the HuBERT framework, with an emphasis on both spoken content modeling and speaker identity
preservation. We first equip the Transformer structure with gated relative position bias to improve its capability on
recognition tasks. For better speaker discrimination, we propose an utterance mixing training strategy, where
additional overlapped utterances are created unsupervisedly and incorporated during model training. Lastly, we scale up
the training dataset from 60k hours to 94k hours. WavLM Large achieves state-of-the-art performance on the SUPERB
benchmark, and brings significant improvements for various speech processing tasks on their representative benchmarks.*

Relevant checkpoints can be found under https://huggingface.co/models?other=wavlm.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/microsoft/unilm/tree/master/wavlm).

## Usage tips

- WavLM is a speech model that accepts a float array corresponding to the raw waveform of the speech signal. Please use
  [`Wav2Vec2Processor`] for the feature extraction.
- WavLM model can be fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded
  using [`Wav2Vec2CTCTokenizer`].
- WavLM performs especially well on speaker verification, speaker identification, and speaker diarization tasks.

## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel
    - forward

## WavLMForCTC

[[autodoc]] WavLMForCTC
    - forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification
    - forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification
    - forward

## WavLMForXVector

[[autodoc]] WavLMForXVector
    - forward
