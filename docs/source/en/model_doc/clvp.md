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

Tips:

1. CLVP or Contrastive Language-Voice Pretrained Transformer is an integral part of the Tortoise TTS model.
2. CLVP can be used to compare different generated speech candidates with the provided text, and the best speech tokens are forwarded to the diffusion model.
3. The use of the [`ClvpModelForConditionalGeneration.generate()`] method is strongly recommended for tortoise usage.

Brief Explanation:

Firstly, the [`ClvpTokenizer`] tokenizes the `text` and the [`ClvpFeatureExtractor`] extracts the `log-melspectrogram` from the desired `audio`. 
Next, the text tokens and audio representations are passed to the [`ClvpConditioningEncoder`], which converts them into `conditioning_embeds` that preserve the text information as well as the voice properties. The [`ClvpForCausalLM`] uses these embeds to generate multiple `speech candidates`. 
The speech encoder converts each speech candidate into a vector representation, and the text encoder converts the text tokens into the same latent space. 
At the end, we compare each speech vector with the text vector to see which speech vector is most similar to the text vector. 


This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/neonbjb/tortoise-tts).


Example :

```python
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

>>> # Define processor and model
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output 
>>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
>>> generated_output = model.generate(input_ids=processor_output["input_ids"], input_features=processor_output["input_features"], num_beams=4, num_return_sequences=4)
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

