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

# Moshi

## Overview

The Moshi model was proposed in [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf) by Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave and Neil Zeghidour.

Moshi is a speech-text foundation model that casts spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. Moshi also predicts time-aligned text tokens as a prefix to audio tokens. This “Inner Monologue” method significantly improves the linguistic quality of generated speech and provides streaming speech recognition and text-to-speech. As a result, Moshi is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_architecture.png">
</div>

The abstract from the paper is the following:

*We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks cannot emulate the experience of real conversations. First, their complexity induces a latency of several seconds between interactions. Second, text being the intermediate modality for dialogue, non-linguistic information that modifies meaning— such as emotion or non-speech sounds— is lost in the interaction. Finally, they rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections. Moshi solves these independent issues altogether by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. We moreover extend the hierarchical semantic-to-acoustic token generation of previous work to first predict time-aligned text tokens as a prefix to audio tokens. Not only this “Inner Monologue” method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide streaming speech recognition and text-to-speech. Our resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice, and is available at github.com/kyutai-labs/moshi.* 

Moshi deals with 3 streams of information:
1. The user's audio
2. Moshi's audio
3. Moshi's textual output

Similarly to [`~MusicgenModel`], audio is represented with audio codebooks, which can be interpreted like tokens. The main difference between text tokens and audio codebooks is that audio codebooks introduce an additional dimension of information.
Text tokens are typically of dim `(batch_size, sequence_length)` but audio tokens are of dim `(batch_size, num_codebooks, sequence_length)`.

Moshi's made of 3 components:

**1. The main decoder (Helium in the paper)**

It corresponds to [`MoshiForCausalLM`]. It is strictly a classic text LLM, that uses an architecture similar to [` ~GemmaForCausalLM`]. In other words, it takes text tokens, embeds them, pass them through the decoder and a language head, to get text logits.

**2. The depth decoder**

On its own, it's also a classic LLM, but this time, instead of generating over the time dimension, it generates over the codebook dimension.

It also means that its context length is `num_codebooks`, thus it can't generate more than `num_codebooks`.

Note that each timestamp - i.e each codebook - gets its own set of Linear Layers and Embeddings.

**3. [`MimiModel`]**

It's the audio encoder from Kyutai, that has recently been integrated to transformers, which is used to "tokenize" audio. It has the same use that [`~EncodecModel`] has in [`~MusicgenModel`].


## Tips:

The original checkpoints can be converted using the conversion script `src/transformers/models/moshi/convert_moshi_transformers.py` 


### How to use the model:

This implementation has two main aims:
1. quickly test model generation by simplifying the original API
2. simplify training. A training guide will come soon, but user contributions are welcomed!

<Tip>

It is designed for intermediate use. We strongly recommend using the original [implementation](https://github.com/kyutai-labs/moshi) to infer the model in real-time streaming.

</Tip>

**1. Model generation**

Moshi is a streaming auto-regressive model with two streams of audio. To put it differently, one audio stream corresponds to what the model said/will say and the other audio stream corresponds to what the user said/will say.

[`MoshiForConditionalGeneration.generate`] thus needs 3 inputs:
1. `input_ids` - corresponding to the text token history
2. `moshi_input_values` or `moshi_audio_codes`- corresponding to the model audio history
3. `user_input_values` or `user_audio_codes` - corresponding to the user audio history

These three inputs must be synchronized. Meaning that their lengths must correspond to the same number of tokens.

You can dynamically use the 3 inputs depending on what you want to test:
1. Simply check the model response to an user prompt - in that case, `input_ids` can be filled with pad tokens and `user_input_values` can be a zero tensor of the same shape than the user prompt.
2. Test more complex behaviour - in that case, you must be careful about how the input tokens are synchronized with the audios.

<Tip>

The original model is synchronized text with audio by padding the text in between each token enunciation.

To follow the example of the following image, `"Hello, I'm Moshi"` could be transformed to `"Hello,<pad><unk>I'm Moshi"`.

</Tip>

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_text_sync.png">
</div>


[`MoshiForConditionalGeneration.generate`] then auto-regressively feeds to itself its own audio stream, but since it doesn't have access to the user input stream while using `transformers`, it will thus **assume that the user is producing blank audio**.



```python 
>>> from datasets import load_dataset, Audio
>>> import torch, math
>>> from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


>>> # prepare user input audio 
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> user_input_values = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(device=device, dtype=dtype)

>>> # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
>>> moshi_input_values = torch.zeros_like(user_input_values.input_values)

>>> # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
>>> num_tokens = math.ceil(moshi_input_values.shape[-1] * waveform_to_token_ratio)
>>> input_ids = torch.ones((1, num_tokens), device=device, dtype=torch.int64) * tokenizer.encode("<pad>")[0]

>>> # generate 25 new tokens (around 2s of audio)
>>> output = model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values, moshi_input_values=moshi_input_values, max_new_tokens=25)

>>> text_tokens = output.sequences
>>> audio_waveforms = output.audio_sequences
```

**2. Model training**

Most of the work has to be done during data creation/pre-processing, because of the need to align/synchronize streams.

Once it's done, you can simply forward `text_labels` and `audio_labels` to [`MoshiForConditionalGeneration.forward`], alongside the usual inputs, to get the model loss.
 
A training guide will come soon, but user contributions are welcomed!

### How does the model forward the inputs / generate:

1. The input streams are embedded and combined into `inputs_embeds`.

2. `inputs_embeds` is passed through the main decoder, which processes it like a normal LLM would.

3. The main decoder outputs `text logits` but also its `last hidden state` which is called `temporal context` in the paper.

3. The depth decoder switches the dimension on which we forward / generate (codebooks instead of time). It uses the token generated from `text logits`  and the `temporal context` to auto-regressively generate audio codebooks.


This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe).

The original code can be found [here](https://github.com/kyutai-labs/moshi).



## MoshiConfig

[[autodoc]] MoshiConfig

## MoshiDepthConfig

[[autodoc]] MoshiDepthConfig

## MoshiModel

[[autodoc]] MoshiModel
    - forward

## MoshiForCausalLM

[[autodoc]] MoshiForCausalLM
    - forward

## MoshiForConditionalGeneration

[[autodoc]] MoshiForConditionalGeneration
    - forward
    - generate
    - get_unconditional_inputs
