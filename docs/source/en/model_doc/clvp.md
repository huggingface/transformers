<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLVP

## Overview

The CLVP (Contrastive Language-Voice Pretrained Transformer) model was proposed in [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) by James Betker.

The abstract from the paper is the following:

*In recent years, the field of image generation has been revolutionized by the application of autoregressive transformers and DDPMs. These approaches model the process of image generation as a step-wise probabilistic processes and leverage large amounts of compute and data to learn the image distribution. This methodology of improving performance need not be confined to images. This paper describes a way to apply advances in the image generative domain to speech synthesis. The result is TorToise - an expressive, multi-voice text-to-speech system.*


This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/neonbjb/tortoise-tts).


## Usage tips

1. CLVP is an integral part of the Tortoise TTS model.
2. CLVP can be used to compare different generated speech candidates with the provided text, and the best speech tokens are forwarded to the diffusion model.
3. The use of the [`ClvpModelForConditionalGeneration.generate()`] method is strongly recommended for tortoise usage.
4. Note that the CLVP model expects the audio to be sampled at 22.05 kHz contrary to other audio models which expects 16 kHz. 


## Brief Explanation:

- The [`ClvpTokenizer`] tokenizes the text input, and the [`ClvpFeatureExtractor`] extracts the log mel-spectrogram from the desired audio.
- [`ClvpConditioningEncoder`] takes those text tokens and audio representations and converts them into embeddings conditioned on the text and audio.
- The [`ClvpForCausalLM`] uses those embeddings to generate multiple speech candidates.
- Each speech candidate is passed through the speech encoder ([`ClvpEncoder`]) which converts them into a vector representation, and the text encoder ([`ClvpEncoder`]) converts the text tokens into the same latent space. 
- At the end, we compare each speech vector with the text vector to see which speech vector is most similar to the text vector. 
- [`ClvpModelForConditionalGeneration.generate()`] compresses all of the logic described above into a single method.  


Example :

```python
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library).
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> sample = ds[0]["audio"]

>>> # Define processor and model.
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output.
>>> processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
>>> generated_output = model.generate(**processor_output)
```


## ClvpConfig

[[autodoc]] ClvpConfig
    - from_sub_model_configs

## ClvpEncoderConfig

[[autodoc]] ClvpEncoderConfig

## ClvpDecoderConfig

[[autodoc]] ClvpDecoderConfig

## ClvpTokenizer

[[autodoc]] ClvpTokenizer
    - save_vocabulary

## ClvpFeatureExtractor

[[autodoc]] ClvpFeatureExtractor
    - __call__

## ClvpProcessor

[[autodoc]] ClvpProcessor
    - __call__
    - decode
    - batch_decode

## ClvpModelForConditionalGeneration

[[autodoc]] ClvpModelForConditionalGeneration
    - forward
    - generate
    - get_text_features
    - get_speech_features

## ClvpForCausalLM

[[autodoc]] ClvpForCausalLM

## ClvpModel

[[autodoc]] ClvpModel

## ClvpEncoder

[[autodoc]] ClvpEncoder

## ClvpDecoder

[[autodoc]] ClvpDecoder

