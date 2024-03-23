<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pop2Piano

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/spaces/sweetcocoa/pop2piano">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The Pop2Piano model was proposed in [Pop2Piano : Pop Audio-based Piano Cover Generation](https://arxiv.org/abs/2211.00895) by Jongho Choi and Kyogu Lee.

Piano covers of pop music are widely enjoyed, but generating them from music is not a trivial task. It requires great 
expertise with playing piano as well as knowing different characteristics and melodies of a song. With Pop2Piano you 
can directly generate a cover from a song's audio waveform. It is the first model to directly generate a piano cover 
from pop audio without melody and chord extraction modules. 

Pop2Piano is an encoder-decoder Transformer model based on [T5](https://arxiv.org/pdf/1910.10683.pdf). The input audio 
is transformed to its waveform and passed to the encoder, which transforms it to a latent representation. The decoder 
uses these latent representations to generate token ids in an autoregressive way. Each token id corresponds to one of four 
different token types: time, velocity, note and 'special'. The token ids are then decoded to their equivalent MIDI file.

The abstract from the paper is the following:

*Piano covers of pop music are enjoyed by many people. However, the
task of automatically generating piano covers of pop music is still
understudied. This is partly due to the lack of synchronized
{Pop, Piano Cover} data pairs, which made it challenging to apply
the latest data-intensive deep learning-based methods. To leverage
the power of the data-driven approach, we make a large amount of
paired and synchronized {Pop, Piano Cover} data using an automated
pipeline. In this paper, we present Pop2Piano, a Transformer network
that generates piano covers given waveforms of pop music. To the best
of our knowledge, this is the first model to generate a piano cover
directly from pop audio without using melody and chord extraction
modules. We show that Pop2Piano, trained with our dataset, is capable
of producing plausible piano covers.*

This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/sweetcocoa/pop2piano).

## Usage tips

* To use Pop2Piano, you will need to install the 🤗 Transformers library, as well as the following third party modules:  
```bash
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```
Please note that you may need to restart your runtime after installation.
* Pop2Piano is an Encoder-Decoder based model like T5.
* Pop2Piano can be used to generate midi-audio files for a given audio sequence.
* Choosing different composers in `Pop2PianoForConditionalGeneration.generate()` can lead to variety of different results.
* Setting the sampling rate to 44.1 kHz when loading the audio file can give good performance.
* Though Pop2Piano was mainly trained on Korean Pop music, it also does pretty well on other Western Pop or Hip Hop songs.

## Examples

- Example using HuggingFace Dataset:

```python
>>> from datasets import load_dataset
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

>>> inputs = processor(
...     audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
... )
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example using your own audio file:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> audio, sr = librosa.load("<your_audio_file_here>", sr=44100)  # feel free to change the sr to a suitable value.
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example of processing multiple audio files in batch:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> # feel free to change the sr to a suitable value.
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt")
>>> # Since we now generating in batch(2 audios) we must pass the attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # Since we now have 2 generated MIDI files
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```


- Example of processing multiple audio files in batch (Using `Pop2PianoFeatureExtractor` and `Pop2PianoTokenizer`):

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoFeatureExtractor, Pop2PianoTokenizer

>>> # feel free to change the sr to a suitable value.
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)  
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
>>> tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = feature_extractor(
...     audio=[audio1, audio2], 
...     sampling_rate=[sr1, sr2], 
...     return_attention_mask=True, 
...     return_tensors="pt",
... )
>>> # Since we now generating in batch(2 audios) we must pass the attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = tokenizer.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # Since we now have 2 generated MIDI files
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```


## Pop2PianoConfig

[[autodoc]] Pop2PianoConfig

## Pop2PianoFeatureExtractor

[[autodoc]] Pop2PianoFeatureExtractor
    - __call__

## Pop2PianoForConditionalGeneration

[[autodoc]] Pop2PianoForConditionalGeneration
    - forward
    - generate

## Pop2PianoTokenizer

[[autodoc]] Pop2PianoTokenizer
    - __call__

## Pop2PianoProcessor

[[autodoc]] Pop2PianoProcessor
    - __call__
