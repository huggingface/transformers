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

# DAC

## Overview


The DAC model was proposed in [Descript Audio Codec: High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) by Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar.

The Descript Audio Codec (DAC) model is a powerful tool for compressing audio data, making it highly efficient for storage and transmission. By compressing 44.1 KHz audio into tokens at just 8kbps bandwidth, the DAC model enables high-quality audio processing while significantly reducing the data footprint. This is particularly useful in scenarios where bandwidth is limited or storage space is at a premium, such as in streaming applications, remote conferencing, and archiving large audio datasets.

The abstract from the paper is the following:

*Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves ~90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling.*

This model was contributed by [Kamil Akesbi](https://huggingface.co/kamilakesbi).
The original code can be found [here](https://github.com/descriptinc/descript-audio-codec/tree/main?tab=readme-ov-file).


## Model structure

The Descript Audio Codec (DAC) model is structured into three distinct stages:

1. Encoder Model: This stage compresses the input audio, reducing its size while retaining essential information.
2. Residual Vector Quantizer (RVQ) Model: Working in tandem with the encoder, this model quantizes the latent codes of the audio, refining the compression and ensuring high-quality reconstruction.
3. Decoder Model: This final stage reconstructs the audio from its compressed form, restoring it to a state that closely resembles the original input.

## Usage example 

Here is a quick example of how to encode and decode an audio using this model: 

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import DacModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = DacModel.from_pretrained("descript/dac_16khz")
>>> processor = AutoProcessor.from_pretrained("descript/dac_16khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"])
>>> # Get the intermediate audio codes
>>> audio_codes = encoder_outputs.audio_codes
>>> # Reconstruct the audio from its quantized representation
>>> audio_values = model.decode(encoder_outputs.quantized_representation)
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"]).audio_values
```

## DacConfig

[[autodoc]] DacConfig

## DacFeatureExtractor

[[autodoc]] DacFeatureExtractor
    - __call__

## DacModel

[[autodoc]] DacModel
    - decode
    - encode
    - forward
