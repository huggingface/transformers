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

# Wav2Vec2-With-LM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Wav2Vec2-With-LM

[Wav2Vec2-With-LM](https://arxiv.org/abs/2010.05171) is a speech model that combines Wav2Vec2 with a language model to improve transcription accuracy. What makes it unique is its ability to leverage both acoustic and linguistic information through a two-stage decoding process: first using Wav2Vec2 for acoustic modeling, then applying a language model to refine the transcription. This approach helps correct common speech recognition errors and produces more natural transcriptions.

You can find all the original Wav2Vec2-With-LM checkpoints under the [Wav2Vec2-With-LM](https://huggingface.co/models?other=wav2vec2-with-lm) collection.

> [!TIP]
> Click on the Wav2Vec2-With-LM models in the right sidebar for more examples of how to apply Wav2Vec2-With-LM to different speech tasks like automatic speech recognition (ASR) with improved accuracy.

The example below demonstrates how to use Wav2Vec2-With-LM for automatic speech recognition with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h-with-lm")
>>> transcriber("path/to/audio.wav")
{'text': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'}
```

</hfoption>
<hfoption id="AutoModel">

```python
>>> from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ProcessorWithLM
>>> import torch
>>> from datasets import load_dataset

>>> processor = Wav2Vec2ProcessorWithLM.from_pretrained("facebook/wav2vec2-base-960h-with-lm")
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h-with-lm")

>>> # Load example audio
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio = dataset[0]["audio"]["array"]

>>> # Process audio
>>> inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

>>> # Get logits
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Decode with LM
>>> transcription = processor.batch_decode(logits.numpy()).text
>>> transcription[0]
'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli transcribe facebook/wav2vec2-base-960h-with-lm path/to/audio.wav
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to quantize the weights to 8-bit integers:

```python
>>> from transformers import AutoModelForCTC, Wav2Vec2ProcessorWithLM
>>> import torch

>>> # Load model in 8-bit
>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-large-960h-with-lm",
...     load_in_8bit=True,
...     device_map="auto"
... )
>>> processor = Wav2Vec2ProcessorWithLM.from_pretrained("facebook/wav2vec2-large-960h-with-lm")

>>> # Process audio and generate transcription
>>> inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> transcription = processor.batch_decode(logits.numpy()).text
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to:

```python
>>> from transformers.utils.attention_visualizer import AttentionMaskVisualizer

>>> visualizer = AttentionMaskVisualizer("facebook/wav2vec2-base-960h-with-lm")
>>> visualizer("path/to/audio.wav")
``` 