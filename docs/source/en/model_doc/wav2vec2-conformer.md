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

# Wav2Vec2-Conformer

## Overview

The Wav2Vec2-Conformer was added to an updated version of [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino.

The official results of the model can be found in Table 3 and Table 4 of the paper.

The Wav2Vec2-Conformer weights were released by the Meta AI team within the [Fairseq library](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
The original code can be found [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

## Usage tips

- Wav2Vec2-Conformer follows the same architecture as Wav2Vec2, but replaces the *Attention*-block with a *Conformer*-block
  as introduced in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100).
- For the same number of layers, Wav2Vec2-Conformer requires more parameters than Wav2Vec2, but also yields 
an improved word error rate.
- Wav2Vec2-Conformer uses the same tokenizer and feature extractor as Wav2Vec2.
- Wav2Vec2-Conformer can use either no relative position embeddings, Transformer-XL-like position embeddings, or
  rotary position embeddings by setting the correct `config.position_embeddings_type`.

## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

## Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

## Wav2Vec2Conformer specific outputs

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

## Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel
    - forward

## Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC
    - forward

## Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification
    - forward

## Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification
    - forward

## Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector
    - forward

## Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining
    - forward
