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

# UniSpeech

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The UniSpeech model was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://huggingface.co/papers/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael
Zeng, Xuedong Huang .

The abstract from the paper is the following:

*In this paper, we propose a unified pre-training approach called UniSpeech to learn speech representations with both
unlabeled and labeled data, in which supervised phonetic CTC learning and phonetically-aware contrastive
self-supervised learning are conducted in a multi-task learning manner. The resultant representations can capture
information more correlated with phonetic structures and improve the generalization across languages and domains. We
evaluate the effectiveness of UniSpeech for cross-lingual representation learning on public CommonVoice corpus. The
results show that UniSpeech outperforms self-supervised pretraining and supervised transfer learning for speech
recognition by a maximum of 13.4% and 17.8% relative phone error rate reductions respectively (averaged over all
testing languages). The transferability of UniSpeech is also demonstrated on a domain-shift speech recognition task,
i.e., a relative word error rate reduction of 6% against the previous approach.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech).

## Usage tips

- UniSpeech is a speech model that accepts a float array corresponding to the raw waveform of the speech signal. Please
  use [`Wav2Vec2Processor`] for the feature extraction.
- UniSpeech model can be fine-tuned using connectionist temporal classification (CTC) so the model output has to be
  decoded using [`Wav2Vec2CTCTokenizer`].

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than "eager". If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

## UniSpeechConfig

[[autodoc]] UniSpeechConfig

## UniSpeech specific outputs

[[autodoc]] models.unispeech.modeling_unispeech.UniSpeechForPreTrainingOutput

## UniSpeechModel

[[autodoc]] UniSpeechModel
    - forward

## UniSpeechForCTC

[[autodoc]] UniSpeechForCTC
    - forward

## UniSpeechForSequenceClassification

[[autodoc]] UniSpeechForSequenceClassification
    - forward

## UniSpeechForPreTraining

[[autodoc]] UniSpeechForPreTraining
    - forward
