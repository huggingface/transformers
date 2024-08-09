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

# MSClap

## Overview

The MSClap model was proposed in [Natural Language Supervision for General-Purpose Audio Representations](Benjamin Elizalde and Soham Deshmukh and Huaming Wang) by Benjamin Elizalde and Soham Deshmukh and Huaming Wang.

MSCLAP (Microsoft Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text snippet, given an audio, without directly optimizing for the task. The MSCLAP model uses a SWINTransformer to get audio features from a log-Mel spectrogram input, and a GPT2 model to get text features. Both the text and audio features are then projected to a latent space with identical dimension. The dot product between the projected audio and text features is then used as a similar score.

The abstract from the paper is the following:

*Mainstream Audio Analytics models are trained to learn under the paradigm of one class label to many recordings focusing on one task. Learning under such restricted supervision limits the flexibility of models because they require labeled audio for training and can only predict the predefined categories. Instead, we propose to learn audio concepts from natural language supervision. We call our approach Contrastive Language-Audio Pretraining (CLAP), which learns to connect language and audio by using two encoders and a contrastive learning to bring audio and text descriptions into a joint multimodal space. We trained CLAP with 128k audio and text pairs and evaluated it on 16 downstream tasks across 8 domains, such as Sound Event Classification, Music tasks, and Speech-related tasks. Although CLAP was trained with significantly less pairs than similar computer vision models, it establishes SoTA for Zero-Shot performance. Additionally, we evaluated CLAP in a supervised learning setup and achieve SoTA in 5 tasks. Hence, CLAP’s Zero-Shot capability removes the need of training with class labels, enables flexible class prediction at inference time, and generalizes to multiple downstream tasks.*


This model was contributed by [kamilakesbi](https://huggingface.co/<kamilakesbi>).
The original code can be found [here](https://github.com/microsoft/CLAP/tree/main).


## Model structure

The model is designed for zero-shot audio classification and retrieval. It consists of:

- GPT-2 Text Encoder: Utilizes the transformer-based GPT-2 architecture for text encoding.
- Audio Encoder: Leverages the SWIN Transformer architecture for audio encoding.

## Usage example

Here's an example of how to perform zero-shot audio classification: 

```
from datasets import load_dataset
from transformers import AutoProcessor, MSClapModel

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
audio_sample = dataset["train"]["audio"][0]["array"]

model = MSClapModel.from_pretrained("microsoft/ms_clap")
processor = AutoProcessor.from_pretrained("microsoft/ms_clap")

input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
```

## MSClapConfig

[[autodoc]] MSClapConfig
    - from_text_audio_configs

## MSClapTextConfig

[[autodoc]] MSClapTextConfig

## MSClapAudioConfig

[[autodoc]] MSClapAudioConfig

## MSClapFeatureExtractor

[[autodoc]] MSClapFeatureExtractor

## MSClapProcessor

[[autodoc]] MSClapProcessor

## MSClapModel

[[autodoc]] MSClapModel
    - forward
    - get_text_features
    - get_audio_features

## MSClapTextModel

[[autodoc]] MSClapTextModel
    - forward

## MSClapTextModelWithProjection

[[autodoc]] MSClapTextModelWithProjection
    - forward

## MSClapAudioModel

[[autodoc]] MSClapAudioModel
    - forward

## MSClapAudioModelWithProjection

[[autodoc]] MSClapAudioModelWithProjection
    - forward
