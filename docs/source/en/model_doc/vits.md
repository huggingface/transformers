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

You can find all the original VITS checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mms-tts) organization.

> [!TIP]
> Click on the VITS models in the right sidebar for more examples of how to apply VITS.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

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
    dtype=torch.float16,
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
import scipy
from IPython.display import Audio
from transformers import AutoTokenizer, VitsModel, set_seed

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng", dtype=torch.float16).to("cuda")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")

set_seed(555)

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]
scipy.io.wavfile.write("hello.wav", rate=model.config.sampling_rate, data=waveform)

# display in Colab notebook
Audio(waveform, rate=model.config.sampling_rate)
```

</hfoption>
</hfoptions>

## Notes

- Set a seed for reproducibility because VITS synthesizes speech non-deterministically.
- For languages with non-Roman alphabets (Korean, Arabic, etc.), install the [uroman](https://github.com/isi-nlp/uroman) package to preprocess the text inputs to the Roman alphabet. You can check if the tokenizer requires uroman as shown below.

   ```py
   # pip install -U uroman
   from transformers import VitsTokenizer

   tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
   print(tokenizer.is_uroman)
   ```

   If your language requires uroman, the tokenizer automatically applies it to the text inputs. Python >= 3.10 doesn't require any additional preprocessing steps. For Python < 3.10, follow the steps below.

   ```bash
   git clone https://github.com/isi-nlp/uroman.git
   cd uroman
   export UROMAN=$(pwd)
   ```

   Create a function to preprocess the inputs. You can either use the bash variable `UROMAN` or pass the directory path directly to the function.

   ```py
   import torch
   from transformers import VitsTokenizer, VitsModel, set_seed
   import os
   import subprocess

   tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
   model = VitsModel.from_pretrained("facebook/mms-tts-kor")

   def uromanize(input_string, uroman_path):
       """Convert non-Roman strings to Roman using the `uroman` perl package."""
       script_path = os.path.join(uroman_path, "bin", "uroman.pl")

       command = ["perl", script_path]

       process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       # Execute the perl command
       stdout, stderr = process.communicate(input=input_string.encode())

       if process.returncode != 0:
           raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

       # Return the output as a string and skip the new-line character at the end
       return stdout.decode()[:-1]

   text = "이봐 무슨 일이야"
   uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

   inputs = tokenizer(text=uromanized_text, return_tensors="pt")

   set_seed(555)  # make deterministic
   with torch.no_grad():
      outputs = model(inputs["input_ids"])

   waveform = outputs.waveform[0]
   ```

## VitsConfig

[[autodoc]] VitsConfig

## VitsTokenizer

[[autodoc]] VitsTokenizer
- __call__
- save_vocabulary

## VitsModel

[[autodoc]] VitsModel
- forward

