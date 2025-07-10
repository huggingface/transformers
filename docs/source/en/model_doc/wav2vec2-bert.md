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

<div style="float: right">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>
</div>

# Wav2Vec2-BERT

[Wav2Vec2-BERT](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) is a powerful speech model that was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. What makes it unique is its ability to learn from a massive amount of multilingual audio data and achieve state-of-the-art results across various speech tasks. It uses a causal depthwise convolutional layer and takes mel-spectrogram representations as input, making it more efficient than raw waveform models.

You can find all the original Wav2Vec2-BERT checkpoints under the [Wav2Vec2-BERT](https://huggingface.co/models?other=wav2vec2-bert) collection.

> [!TIP]
> Click on the Wav2Vec2-BERT models in the right sidebar for more examples of how to apply Wav2Vec2-BERT to different speech tasks like automatic speech recognition (ASR) and audio classification.

The example below demonstrates how to use Wav2Vec2-BERT for automatic speech recognition with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-bert-base-960h")
>>> transcriber("path/to/audio.wav")
{'text': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'}
```

</hfoption>
<hfoption id="AutoModel">

```python
>>> from transformers import AutoProcessor, AutoModelForCTC
>>> import torch
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-bert-base-960h")
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-bert-base-960h")

>>> # Load example audio
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio = dataset[0]["audio"]["array"]

>>> # Process audio
>>> inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

>>> # Get logits
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Decode
>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli transcribe facebook/wav2vec2-bert-base-960h path/to/audio.wav
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to quantize the weights to 8-bit integers:

```python
>>> from transformers import AutoModelForCTC, AutoProcessor
>>> import torch

>>> # Load model in 8-bit
>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-bert-large-960h",
...     load_in_8bit=True,
...     device_map="auto"
... )
>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-bert-large-960h")

>>> # Process audio and generate transcription
>>> inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to:

```python
>>> from transformers.utils.attention_visualizer import AttentionMaskVisualizer

>>> visualizer = AttentionMaskVisualizer("facebook/wav2vec2-bert-base-960h")
>>> visualizer("path/to/audio.wav")
```

## Overview

The Wav2Vec2-BERT model was proposed in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team from Meta AI.

This model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. It requires finetuning to be used for downstream tasks such as Automatic Speech Recognition (ASR), or Audio Classification.

The official results of the model can be found in Section 3.2.1 of the paper.

The abstract from the paper is the following:

*Recent advancements in automatic speech translation have dramatically expanded language coverage, improved multimodal capabilities, and enabled a wide range of tasks and functionalities. That said, large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model—SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. The expanded version of SeamlessAlign adds 114,800 hours of automatically aligned data for a total of 76 languages. SeamlessM4T v2 provides the foundation on which our two newest models, SeamlessExpressive and SeamlessStreaming, are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one's voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention (EMMA) mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To understand the performance of these models, we combined novel and modified versions of existing automatic metrics to evaluate prosody, latency, and robustness. For human evaluations, we adapted existing protocols tailored for measuring the most relevant attributes in the preservation of meaning, naturalness, and expressivity. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. In sum, Seamless gives us a pivotal look at the technical foundation needed to turn the Universal Speech Translator from a science fiction concept into a real-world technology. Finally, contributions in this work—including models, code, and a watermark detector—are publicly released and accessible at the link below.*

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
