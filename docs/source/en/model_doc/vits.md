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

The abstract from the paper is the following:

*Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.*

This model can also be used with MMS-TTS checkpoints as those use the same architecture (but a different tokenizer).

This model was contributed by [Matthijs](https://huggingface.co/Matthijs). The original code can be found [here](https://github.com/jaywalnut310/vits).

## Model Usage

Both the VITS and MMS-TTS checkpoints can be used with the same API. Since the flow-based model is non-deterministic, it 
is good practice to set a seed to ensure reproducibility of the outputs. For languages with a Roman alphabet, 
such as English or French, the tokenizer can be used directly to pre-process the text inputs. The following code example 
runs a forward pass using the MMS-TTS English checkpoint:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("sanchit-gandhi/mms-tts-eng")
model = VitsModel.from_pretrained("sanchit-gandhi/mms-tts-eng")

inputs = tokenizer(text="Hello, my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

outputs.audio.shape
```

For certain languages with a non-Roman alphabet, such as Arabic, Mandarin or Hindi, the [`uroman`](https://github.com/isi-nlp/uroman) 
perl package is required to pre-process the text inputs to the Roman alphabet. First, clone the `uroman` package:

```bash
git clone https://github.com/isi-nlp/uroman.git
```

Then specify the path to the `uroman` package when you call the `tokenizer`. The following example generates 
speech using the MMS-TTS Korean checkpoint and the `uroman` package:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("sanchit-gandhi/mms-tts-kor")
model = VitsModel.from_pretrained("sanchit-gandhi/mms-tts-kor")

inputs = tokenizer(text="이봐 무슨 일이야", uroman_path="./uroman", return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

outputs.audio.shape
```

You can check whether you require the `uroman` package for your language by inspecting the `is_uroman` attribute of 
the pre-trained `tokenizer`:

```python
tokenizer.is_uroman
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
