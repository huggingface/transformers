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
*This model was released on 2020-06-24 and added to Hugging Face Transformers on 2023-06-20.*

# XLSR-Wav2Vec2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The XLSR-Wav2Vec2 model was proposed in [Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://huggingface.co/papers/2006.13979) by Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael
Auli.

The abstract from the paper is the following:

*This paper presents XLSR which learns cross-lingual speech representations by pretraining a single model from the raw
waveform of speech in multiple languages. We build on wav2vec 2.0 which is trained by solving a contrastive task over
masked latent speech representations and jointly learns a quantization of the latents shared across languages. The
resulting model is fine-tuned on labeled data and experiments show that cross-lingual pretraining significantly
outperforms monolingual pretraining. On the CommonVoice benchmark, XLSR shows a relative phoneme error rate reduction
of 72% compared to the best known results. On BABEL, our approach improves word error rate by 16% relative compared to
a comparable system. Our approach enables a single multilingual speech recognition model which is competitive to strong
individual models. Analysis shows that the latent discrete speech representations are shared across languages with
increased sharing for related languages. We hope to catalyze research in low-resource speech understanding by releasing
XLSR-53, a large model pretrained in 53 languages.*

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

Note: Meta (FAIR) released a new version of [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - it's pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, e.g. as per [this guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Usage tips

- XLSR-Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- XLSR-Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be
  decoded using [`Wav2Vec2CTCTokenizer`].

<Tip>

XLSR-Wav2Vec2's architecture is based on the Wav2Vec2 model, so one can refer to [Wav2Vec2's documentation page](wav2vec2).

</Tip>
