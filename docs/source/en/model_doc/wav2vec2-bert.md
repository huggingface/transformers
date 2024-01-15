<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Wav2Vec2-BERT

## Overview

The Wav2Vec2-BERT model was proposed in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team from Meta AI.

This model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. It requires finetuning to be used for downstream tasks such as Automatic Speech Recognition (ASR), or Audio Classification.

The official results of the model can be found in Section 3.2.1 of the paper.

The abstract from the paper is the following:

*Recent advancements in automatic speech translation have dramatically expanded language coverage, improved multimodal capabilities, and enabled a wide range of tasks and functionalities. That said, large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model—SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. The expanded version of SeamlessAlign adds 114,800 hours of automatically aligned data for a total of 76 languages. SeamlessM4T v2 provides the foundation on which our two newest models, SeamlessExpressive and SeamlessStreaming, are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one’s voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention (EMMA) mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To understand the performance of these models, we combined novel and modified versions of existing automatic metrics to evaluate prosody, latency, and robustness. For human evaluations, we adapted existing protocols tailored for measuring the most relevant attributes in the preservation of meaning, naturalness, and expressivity. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. In sum, Seamless gives us a pivotal look at the technical foundation needed to turn the Universal Speech Translator from a science fiction concept into a real-world technology. Finally, contributions in this work—including models, code, and a watermark detector—are publicly released and accessible at the link below.*

This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## Usage tips

- Wav2Vec2-BERT follows the same architecture as Wav2Vec2-Conformer, but employs a causal depthwise convolutional layer and uses as input a mel-spectrogram representation of the audio instead of the raw waveform.
- Wav2Vec2-BERT can use either no relative position embeddings, Shaw-like position embeddings, Transformer-XL-like position embeddings, or
  rotary position embeddings by setting the correct `config.position_embeddings_type`.
- Wav2Vec2-BERT also introduces a Conformer-based adapter network instead of a simple convolutional network.

## Resources

<PipelineTag pipeline="automatic-speech-recognition"/>

- [`Wav2Vec2BertForCTC`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition).
- You can also adapt these notebooks on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).

<PipelineTag pipeline="audio-classification"/>

- [`Wav2Vec2BertForSequenceClassification`] can be used by adapting this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification).
- See also: [Audio classification task guide](../tasks/audio_classification)


## Wav2Vec2BertConfig

[[autodoc]] Wav2Vec2BertConfig

## Wav2Vec2BertProcessor

[[autodoc]] Wav2Vec2BertProcessor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Wav2Vec2BertModel

[[autodoc]] Wav2Vec2BertModel
    - forward

## Wav2Vec2BertForCTC

[[autodoc]] Wav2Vec2BertForCTC
    - forward

## Wav2Vec2BertForSequenceClassification

[[autodoc]] Wav2Vec2BertForSequenceClassification
    - forward

## Wav2Vec2BertForAudioFrameClassification

[[autodoc]] Wav2Vec2BertForAudioFrameClassification
    - forward

## Wav2Vec2BertForXVector

[[autodoc]] Wav2Vec2BertForXVector
    - forward
