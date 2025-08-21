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
*This model was released on 2021-11-17 and added to Hugging Face Transformers on 2023-06-20.*

# XLS-R

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The XLS-R model was proposed in [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://huggingface.co/papers/2111.09296) by Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman
Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli.

The abstract from the paper is the following:

*This paper presents XLS-R, a large-scale model for cross-lingual speech representation learning based on wav2vec 2.0.
We train models with up to 2B parameters on nearly half a million hours of publicly available speech audio in 128
languages, an order of magnitude more public data than the largest known prior work. Our evaluation covers a wide range
of tasks, domains, data regimes and languages, both high and low-resource. On the CoVoST-2 speech translation
benchmark, we improve the previous state of the art by an average of 7.4 BLEU over 21 translation directions into
English. For speech recognition, XLS-R improves over the best known prior work on BABEL, MLS, CommonVoice as well as
VoxPopuli, lowering error rates by 14-34% relative on average. XLS-R also sets a new state of the art on VoxLingua107
language identification. Moreover, we show that with sufficient model size, cross-lingual pretraining can outperform
English-only pretraining when translating English speech into other languages, a setting which favors monolingual
pretraining. We hope XLS-R can help to improve speech processing tasks for many more languages of the world.*

Relevant checkpoints can be found under https://huggingface.co/models?other=xls_r.

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

## Usage tips

- XLS-R is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- XLS-R model was trained using connectionist temporal classification (CTC) so the model output has to be decoded using
  [`Wav2Vec2CTCTokenizer`].

<Tip>

XLS-R's architecture is based on the Wav2Vec2 model, refer to [Wav2Vec2's documentation page](wav2vec2) for API reference.

</Tip>
