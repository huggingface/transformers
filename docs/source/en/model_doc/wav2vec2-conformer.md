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

<div style="float: right">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>
</div>

# Wav2Vec2-Conformer

[Wav2Vec2-Conformer](https://arxiv.org/abs/2010.05171) is a speech model that builds upon Wav2Vec2 by replacing the standard attention blocks with Conformer blocks. What makes it unique is its ability to capture both local and global dependencies in speech through a combination of self-attention and convolution layers, leading to improved word error rates compared to the original Wav2Vec2 model. The Conformer architecture was introduced in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) and has become a popular choice for speech recognition tasks.

You can find all the original Wav2Vec2-Conformer checkpoints under the [Wav2Vec2-Conformer](https://huggingface.co/models?other=wav2vec2-conformer) collection.

> [!TIP]
> Click on the Wav2Vec2-Conformer models in the right sidebar for more examples of how to apply Wav2Vec2-Conformer to different speech tasks like automatic speech recognition (ASR) and audio classification.

The example below demonstrates how to use Wav2Vec2-Conformer for automatic speech recognition with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-conformer-base-960h")
>>> transcriber("path/to/audio.wav")
{'text': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'}
```

</hfoption>
<hfoption id="AutoModel">

```python
>>> from transformers import AutoProcessor, AutoModelForCTC
>>> import torch
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-base-960h")
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-conformer-base-960h")

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
transformers-cli transcribe facebook/wav2vec2-conformer-base-960h path/to/audio.wav
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
...     "facebook/wav2vec2-conformer-large-960h",
...     load_in_8bit=True,
...     device_map="auto"
... )
>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-large-960h")

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

>>> visualizer = AttentionMaskVisualizer("facebook/wav2vec2-conformer-base-960h")
>>> visualizer("path/to/audio.wav")
```

## Overview

The Wav2Vec2-Conformer was added to an updated version of [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino.

The official results of the model can be found in Table 3 and Table 4 of the paper.

The Wav2Vec2-Conformer weights were released by the Meta AI team within the [Fairseq library](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
The original code can be found [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

Note: Meta (FAIR) released a new version of [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - it's pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, e.g. as per [this guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

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
