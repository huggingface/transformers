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

# Wav2Vec2Phoneme

## Overview

The Wav2Vec2Phoneme model was proposed in [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al.,
2021](https://arxiv.org/abs/2109.11680) by Qiantong Xu, Alexei Baevski, Michael Auli.

The abstract from the paper is the following:

*Recent progress in self-training, self-supervised pretraining and unsupervised learning enabled well performing speech
recognition systems without any labeled data. However, in many cases there is labeled data available for related
languages which is not utilized by these methods. This paper extends previous work on zero-shot cross-lingual transfer
learning by fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen languages. This is done by
mapping phonemes of the training languages to the target language using articulatory features. Experiments show that
this simple method significantly outperforms prior work which introduced task-specific architectures and used only part
of a monolingually pretrained model.*

Tips:

- Wav2Vec2Phoneme uses the exact same architecture as Wav2Vec2
- Wav2Vec2Phoneme is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- Wav2Vec2Phoneme model was trained using connectionist temporal classification (CTC) so the model output has to be
  decoded using [`Wav2Vec2PhonemeCTCTokenizer`].
- Wav2Vec2Phoneme can be fine-tuned on multiple language at once and decode unseen languages in a single forward pass
  to a sequence of phonemes
- By default the model outputs a sequence of phonemes. In order to transform the phonemes to a sequence of words one
  should make use of a dictionary and language model.

Relevant checkpoints can be found under https://huggingface.co/models?other=phoneme-recognition.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten)

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

Wav2Vec2Phoneme's architecture is based on the Wav2Vec2 model, so one can refer to [`Wav2Vec2`]'s documentation page except for the tokenizer.


## Wav2Vec2PhonemeCTCTokenizer

[[autodoc]] Wav2Vec2PhonemeCTCTokenizer
	- __call__
	- batch_decode
	- decode
	- phonemize
