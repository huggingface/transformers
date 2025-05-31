<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# FastSpeech2Conformer

[FastSpeech2Conformer](https://huggingface.co/papers/2010.13956) is a non-autoregressive text-to-speech (TTS) model that combines FastSpeech2's architecture with [Conformer](https://huggingface.co/papers/2005.08100) blocks for improved performance.

FastSpeech2Conformer can synthesize speech significantly faster than autoregressive models while maintaining high quality. Unlike the original FastSpeech, it trains directly on ground-truth data and incorporates speech variation information like pitch, energy, and duration for more natural-sounding output. The Conformer architecture combines the strengths of convolution and attention mechanisms to better capture both local and global dependencies in speech.

You can find all the original FastSpeech2Conformer checkpoints under the [ESPnet](https://huggingface.co/espnet?search_models=fastspeech2) organization.

> [!TIP]
> Click on the FastSpeech2Conformer models in the right sidebar for more examples of how to apply FastSpeech2Conformer to different text-to-speech tasks.

The example below demonstrates how to generate audio from text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline
import soundfile as sf

# Initialize the TTS pipeline
synthesizer = pipeline("text-to-speech", model="espnet/fastspeech2_conformer")

# Generate speech
speech = synthesizer("Hello, my dog is cute.")

# Save the audio to a file
sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

</hfoption>
<hfoption id="AutoModel">

```py
# pip install -U -q g2p-en
import soundfile as sf
import torch
from transformers import AutoTokenizer, FastSpeech2ConformerWithHifiGan

tokenizer = AutoTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan", torch_dtype=torch.float16, device_map="auto")

inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]

output_dict = model(input_ids, return_dict=True)
waveform = output_dict["waveform"]
sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

</hfoption>
</hfoptions>

For a more streamlined experience, you can use the combined model with the HiFi-GAN vocoder in a single step:

```python
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")

inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")

output_dict = model(inputs["input_ids"], return_dict=True)
waveform = output_dict["waveform"]

sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

## Notes

- Use the matching [HiFi-GAN vocoder](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) for optimal audio quality.
- Use the combined model, [`FastSpeech2ConformerWithHiFiGan`], for faster generation speed since it handles both spectrogram generation and vocoding in a single step.
- You could also use the [`FastSpeech2ConformerModel`] and [`FastSpeech2ConformerHiFiGan`] separately as shown below.
   <add code snippet here>

## FastSpeech2ConformerConfig

[[autodoc]] FastSpeech2ConformerConfig

## FastSpeech2ConformerHifiGanConfig

[[autodoc]] FastSpeech2ConformerHifiGanConfig

## FastSpeech2ConformerWithHifiGanConfig

[[autodoc]] FastSpeech2ConformerWithHifiGanConfig

## FastSpeech2ConformerTokenizer

[[autodoc]] FastSpeech2ConformerTokenizer
    - __call__
    - save_vocabulary
    - decode
    - batch_decode

## FastSpeech2ConformerModel

[[autodoc]] FastSpeech2ConformerModel
    - forward

## FastSpeech2ConformerHifiGan

[[autodoc]] FastSpeech2ConformerHifiGan
    - forward

## FastSpeech2ConformerWithHifiGan

[[autodoc]] FastSpeech2ConformerWithHifiGan
    - forward
