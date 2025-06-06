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

# Wav2Vec2-Phoneme

<div style="float: right">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>
</div>

# Wav2Vec2-Phoneme

[Wav2Vec2-Phoneme](https://arxiv.org/abs/2010.05171) is a speech model that extends Wav2Vec2 by predicting phonemes instead of characters. What makes it unique is its ability to directly output phoneme sequences, which can be particularly useful for tasks like speech synthesis and pronunciation assessment. The model is trained using a phoneme vocabulary, making it more suitable for applications that require precise phonetic transcription.

You can find all the original Wav2Vec2-Phoneme checkpoints under the [Wav2Vec2-Phoneme](https://huggingface.co/models?other=wav2vec2-phoneme) collection.

> [!TIP]
> Click on the Wav2Vec2-Phoneme models in the right sidebar for more examples of how to apply Wav2Vec2-Phoneme to different speech tasks like phoneme recognition and pronunciation assessment.

The example below demonstrates how to use Wav2Vec2-Phoneme for phoneme recognition with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-phoneme")
>>> transcriber("path/to/audio.wav")
{'text': 'HH EH L OW W ER L D'}
```

</hfoption>
<hfoption id="AutoModel">

```python
>>> from transformers import AutoProcessor, AutoModelForCTC
>>> import torch
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-phoneme")
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-phoneme")

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
'HH EH L OW W ER L D'
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli transcribe facebook/wav2vec2-phoneme path/to/audio.wav
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
...     "facebook/wav2vec2-phoneme-large",
...     load_in_8bit=True,
...     device_map="auto"
... )
>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-phoneme-large")

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

>>> visualizer = AttentionMaskVisualizer("facebook/wav2vec2-phoneme")
>>> visualizer("path/to/audio.wav")
```

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

Relevant checkpoints can be found under https://huggingface.co/models?other=phoneme-recognition.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten)

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

## Usage tips

- Wav2Vec2Phoneme uses the exact same architecture as Wav2Vec2
- Wav2Vec2Phoneme is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- Wav2Vec2Phoneme model was trained using connectionist temporal classification (CTC) so the model output has to be
  decoded using [`Wav2Vec2PhonemeCTCTokenizer`].
- Wav2Vec2Phoneme can be fine-tuned on multiple language at once and decode unseen languages in a single forward pass
  to a sequence of phonemes
- By default, the model outputs a sequence of phonemes. In order to transform the phonemes to a sequence of words one
  should make use of a dictionary and language model.


<Tip>

Wav2Vec2Phoneme's architecture is based on the Wav2Vec2 model, for API reference, check out [`Wav2Vec2`](wav2vec2)'s documentation page 
except for the tokenizer.

</Tip>

## Wav2Vec2PhonemeCTCTokenizer

[[autodoc]] Wav2Vec2PhonemeCTCTokenizer
	- __call__
	- batch_decode
	- decode
	- phonemize
