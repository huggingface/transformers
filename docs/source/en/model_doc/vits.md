<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-06-11 and added to Hugging Face Transformers on 2023-09-01 and contributed by [Matthijs](https://huggingface.co/Matthijs) and [sanchit-gandhi](https://huggingface.co/sanchit-gandhi).*

# VITS

[VITS](https://huggingface.co/papers/2106.06103) is an end-to-end text-to-speech model that uses variational inference with adversarial learning. It consists of a posterior encoder, decoder, and conditional prior, enhanced with normalizing flows. The model predicts spectrogram-based acoustic features using a Transformer-based text encoder and coupling layers, and synthesizes waveforms with transposed convolutional layers similar to HiFi-GAN. A stochastic duration predictor allows for diverse rhythmic variations in speech synthesis. Trained with a combination of variational lower bound and adversarial losses, VITS achieves high-quality audio output, outperforming existing two-stage TTS systems in terms of naturalness and expressiveness.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-to-audio", model="facebook/mms-tts-eng", dtype="auto")
output = pipeline("Plants create energy through a process known as photosynthesis.")
audio = output["audio"]
```

</hfoption>
<hfoption id="VitsModel">

```py
import torch
import scipy
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng", dtype="auto")

inputs = tokenizer(text="Plants create energy through a process known as photosynthesis.", return_tensors="pt")

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
scipy.io.wavfile.write("outputwav", rate=model.config.sampling_rate, data=waveform)
```

</hfoption>
</hfoptions>

## Usage tips

- Set a seed for reproducibility because VITS synthesizes speech non-deterministically.
- Install the `uroman` package for languages with non-Roman alphabets (Korean, Arabic, etc.) to preprocess text inputs to the Roman alphabet. Check if the tokenizer requires uroman as shown below.
- The tokenizer automatically applies uroman to text inputs when your language requires it. Python >= 3.10 skips additional preprocessing steps.

## VitsConfig

[[autodoc]] VitsConfig

## VitsTokenizer

[[autodoc]] VitsTokenizer
    - __call__
    - save_vocabulary

## VitsModel

[[autodoc]] VitsModel
    - forward

