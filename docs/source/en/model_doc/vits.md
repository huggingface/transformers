<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VITS

## Overview

The VITS model was proposed in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son.

VITS (**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech) is an end-to-end 
speech synthesis model that predicts a speech waveform conditional on an input text sequence. It is a conditional variational 
autoencoder (VAE) comprised of a posterior encoder, decoder, and conditional prior.

A set of spectrogram-based acoustic features are predicted by the flow-based module, which is formed of a Transformer-based
text encoder and multiple coupling layers. The spectrogram is decoded using a stack of transposed convolutional layers,
much in the same style as the HiFi-GAN vocoder. Motivated by the one-to-many nature of the TTS problem, where the same text 
input can be spoken in multiple ways, the model also includes a stochastic duration predictor, which allows the model to 
synthesise speech with different rhythms from the same input text. 

The model is trained end-to-end with a combination of losses derived from variational lower bound and adversarial training. 
To improve the expressiveness of the model, normalizing flows are applied to the conditional prior distribution. During 
inference, the text encodings are up-sampled based on the duration prediction module, and then mapped into the 
waveform using a cascade of the flow module and HiFi-GAN decoder. Due to the stochastic nature of the duration predictor,
the model is non-deterministic, and thus requires a fixed seed to generate the same speech waveform.

The abstract from the paper is the following:

*Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.*

This model can also be used with TTS checkpoints from [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516) 
as these checkpoints use the same architecture and a slightly modified tokenizer.

This model was contributed by [Matthijs](https://huggingface.co/Matthijs) and [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). The original code can be found [here](https://github.com/jaywalnut310/vits).

## Usage examples

Both the VITS and MMS-TTS checkpoints can be used with the same API. Since the flow-based model is non-deterministic, it 
is good practice to set a seed to ensure reproducibility of the outputs. For languages with a Roman alphabet, 
such as English or French, the tokenizer can be used directly to pre-process the text inputs. The following code example 
runs a forward pass using the MMS-TTS English checkpoint:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
```

The resulting waveform can be saved as a `.wav` file:

```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
```

Or displayed in a Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

For certain languages with a non-Roman alphabet, such as Arabic, Mandarin or Hindi, the [`uroman`](https://github.com/isi-nlp/uroman) 
perl package is required to pre-process the text inputs to the Roman alphabet.

You can check whether you require the `uroman` package for your language by inspecting the `is_uroman` attribute of 
the pre-trained `tokenizer`:

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

If required, you should apply the uroman package to your text inputs **prior** to passing them to the `VitsTokenizer`, 
since currently the tokenizer does not support performing the pre-processing itself.  

To do this, first clone the uroman repository to your local machine and set the bash variable `UROMAN` to the local path:

```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

You can then pre-process the text input using the following code snippet. You can either rely on using the bash variable 
`UROMAN` to point to the uroman repository, or you can pass the uroman directory as an argument to the `uromaize` function:

```python
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
uromaized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromaized_text, return_tensors="pt")

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
