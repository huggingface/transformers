<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# VITS

[VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)](https://hf.co/papers/2106.06103) is a end-to-end speech synthesis model, simplifying the traditional two-stage text-to-speech (TTS) systems. It's unique because it directly synthesizes speech from text using variational inference, adversarial learning, and normalizing flows to produce natural and expressive speech with diverse rhythms and intonations.

You can explore all official VITS checkpoints [here](https://huggingface.co/facebook?search_models=mms-tts).

> [!TIP]
> - VITS synthesizes speech non-deterministically. For consistent results, set a fixed random seed.
> - For languages with non-Roman alphabets (e.g., Korean or Arabic), you might need the [`uroman`](https://github.com/isi-nlp/uroman) package.
> - Check if your tokenizer requires uroman preprocessing by inspecting the `is_uroman` attribute.
> - Click on the VITS models in the right sidebar for more examples and use-cases!

The following examples show how easy it is to generate speech from text using `Pipeline` and `AutoModel`:

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline, set_seed
from scipy.io.wavfile import write

set_seed(555)

pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-eng",
    torch_dtype=torch.float16,
    device=0
)

speech = pipe("Hello, my dog is cute")

# Extract audio data and sampling rate
audio_data = speech["audio"]
sampling_rate = speech["sampling_rate"]

# Save as WAV file
write("hello.wav", sampling_rate, audio_data.squeeze())
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

set_seed(555)

model_id = "facebook/mms-tts-eng"
tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]

# To save the generated waveform as an audio file:

import scipy
scipy.io.wavfile.write("hello.wav", rate=model.config.sampling_rate, data=waveform.cpu().numpy())

# Or play it in a notebook:

from IPython.display import Audio
Audio(waveform.cpu().numpy(), rate=model.config.sampling_rate)
```

</hfoption>
</hfoptions>

## VitsConfig

[[autodoc]] VitsConfig

## VitsTokenizer

[[autodoc]] VitsTokenizer
- __call__
- save_vocabulary

## VitsModel

[[autodoc]] VitsModel
- forward

