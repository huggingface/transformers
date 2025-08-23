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

# MMS

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
</div>

## Overview

The MMS model was proposed in [Scaling Speech Technology to 1,000+ Languages](https://huggingface.co/papers/2305.13516) 
by Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli

The abstract from the paper is the following:

*Expanding the language coverage of speech technology has the potential to improve access to information for many more people. 
However, current speech technology is restricted to about one hundred languages which is a small fraction of the over 7,000
languages spoken around the world. 
The Massively Multilingual Speech (MMS) project increases the number of supported languages by 10-40x, depending on the task. 
The main ingredients are a new dataset based on readings of publicly available religious texts and effectively leveraging
self-supervised learning. We built pre-trained wav2vec 2.0 models covering 1,406 languages, 
a single multilingual automatic speech recognition model for 1,107 languages, speech synthesis models 
for the same number of languages, as well as a language identification model for 4,017 languages. 
Experiments show that our multilingual speech recognition model more than halves the word error rate of 
Whisper on 54 languages of the FLEURS benchmark while being trained on a small fraction of the labeled data.*

Here are the different models open sourced in the MMS project. The models and code are originally released [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms). We have add them to the `transformers` framework, making them easier to use.

### Automatic Speech Recognition (ASR)

The ASR model checkpoints  can be found here : [mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102), [mms-1b-l1107](https://huggingface.co/facebook/mms-1b-l1107), [mms-1b-all](https://huggingface.co/facebook/mms-1b-all). For best accuracy, use the `mms-1b-all` model. 

Tips:

- All ASR models accept a float array corresponding to the raw waveform of the speech signal. The raw waveform should be pre-processed with [`Wav2Vec2FeatureExtractor`].
- The models were trained using connectionist temporal classification (CTC) so the model output has to be decoded using
  [`Wav2Vec2CTCTokenizer`].
- You can load different language adapter weights for different languages via [`~Wav2Vec2PreTrainedModel.load_adapter`]. Language adapters only consists of roughly 2 million parameters 
  and can therefore be efficiently loaded on the fly when needed.

#### Loading

By default MMS loads adapter weights for English. If you want to load adapter weights of another language 
make sure to specify `target_lang=<your-chosen-target-lang>` as well as `"ignore_mismatched_sizes=True`.
The `ignore_mismatched_sizes=True` keyword has to be passed to allow the language model head to be resized according
to the vocabulary of the specified language.
Similarly, the processor should be loaded with the same target language

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor

model_id = "facebook/mms-1b-all"
target_lang = "fra"

processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True)
```

<Tip>

You can safely ignore a warning such as:

```text
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/mms-1b-all and are newly initialized because the shapes did not match:
- lm_head.bias: found shape torch.Size([154]) in the checkpoint and torch.Size([314]) in the model instantiated
- lm_head.weight: found shape torch.Size([154, 1280]) in the checkpoint and torch.Size([314, 1280]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

</Tip>

If you want to use the ASR pipeline, you can load your chosen target language as such:

```py
from transformers import pipeline

model_id = "facebook/mms-1b-all"
target_lang = "fra"

pipe = pipeline(model=model_id, model_kwargs={"target_lang": "fra", "ignore_mismatched_sizes": True})
```

#### Inference

Next, let's look at how we can run MMS in inference and change adapter layers after having called [`~PretrainedModel.from_pretrained`]
First, we load audio data in different languages using the [Datasets](https://github.com/huggingface/datasets).

```py
from datasets import load_dataset, Audio

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# French
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
fr_sample = next(iter(stream_data))["audio"]["array"]
```

Next, we load the model and processor

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

Now we process the audio data, pass the processed audio data to the model and transcribe the model output,
just like we usually do for [`Wav2Vec2ForCTC`].

```py
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'
```

We can now keep the same model in memory and simply switch out the language adapters by
calling the convenient [`~Wav2Vec2ForCTC.load_adapter`] function for the model and [`~Wav2Vec2CTCTokenizer.set_target_lang`] for the tokenizer.
We pass the target language as an input - `"fra"` for French.

```py
processor.tokenizer.set_target_lang("fra")
model.load_adapter("fra")

inputs = processor(fr_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# "ce dernier est volé tout au long de l'histoire romaine"
```

In the same way the language can be switched out for all other supported languages. Please have a look at:

```py
processor.tokenizer.vocab.keys()
```

to see all supported languages.

To further improve performance from ASR models, language model decoding can be used. See the documentation [here](https://huggingface.co/facebook/mms-1b-all) for further details.  

### Speech Synthesis (TTS)

MMS-TTS uses the same model architecture as VITS, which was added to 🤗 Transformers in v4.33. MMS trains a separate 
model checkpoint for each of the 1100+ languages in the project. All available checkpoints can be found on the Hugging 
Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts), and the inference 
documentation under [VITS](https://huggingface.co/docs/transformers/main/en/model_doc/vits).

#### Inference

To use the MMS model, first update to the latest version of the Transformers library:

```bash
pip install --upgrade transformers accelerate
```

Since the flow-based model in VITS is non-deterministic, it is good practice to set a seed to ensure reproducibility of 
the outputs. 

- For languages with a Roman alphabet, such as English or French, the tokenizer can be used directly to 
pre-process the text inputs. The following code example runs a forward pass using the MMS-TTS English checkpoint:

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

scipy.io.wavfile.write("synthesized_speech.wav", rate=model.config.sampling_rate, data=waveform)
```

Or displayed in a Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

For certain languages with non-Roman alphabets, such as Arabic, Mandarin or Hindi, the [`uroman`](https://github.com/isi-nlp/uroman) 
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
`UROMAN` to point to the uroman repository, or you can pass the uroman directory as an argument to the `uromanize` function:

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
uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromanized_text, return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

**Tips:**

* The MMS-TTS checkpoints are trained on lower-cased, un-punctuated text. By default, the `VitsTokenizer` *normalizes* the inputs by removing any casing and punctuation, to avoid passing out-of-vocabulary characters to the model. Hence, the model is agnostic to casing and punctuation, so these should be avoided in the text prompt. You can disable normalisation by setting `normalize=False` in the call to the tokenizer, but this will lead to un-expected behaviour and is discouraged.
* The speaking rate can be varied by setting the attribute `model.speaking_rate` to a chosen value. Likewise, the randomness of the noise is controlled by `model.noise_scale`:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

# make deterministic
set_seed(555)  

# make speech faster and more noisy
model.speaking_rate = 1.5
model.noise_scale = 0.8

with torch.no_grad():
   outputs = model(**inputs)
```

### Language Identification (LID)

Different LID models are available based on the number of languages they can recognize - [126](https://huggingface.co/facebook/mms-lid-126), [256](https://huggingface.co/facebook/mms-lid-256), [512](https://huggingface.co/facebook/mms-lid-512), [1024](https://huggingface.co/facebook/mms-lid-1024), [2048](https://huggingface.co/facebook/mms-lid-2048), [4017](https://huggingface.co/facebook/mms-lid-4017). 

#### Inference
First, we install transformers and some other libraries

```bash
pip install torch accelerate datasets[audio]
pip install --upgrade transformers
````

Next, we load a couple of audio samples via `datasets`. Make sure that the audio data is sampled to 16000 kHz.

```py
from datasets import load_dataset, Audio

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# Arabic
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
ar_sample = next(iter(stream_data))["audio"]["array"]
```

Next, we load the model and processor

```py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
```

Now we process the audio data, pass the processed audio data to the model to classify it into a language, just like we usually do for Wav2Vec2 audio classification models such as [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)

```py
# English
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'eng'

# Arabic
inputs = processor(ar_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'ara'
```

To see all the supported languages of a checkpoint, you can print out the language ids as follows:
```py
processor.id2label.values()
```

### Audio Pretrained Models

Pretrained models are available for two different sizes - [300M](https://huggingface.co/facebook/mms-300m) , 
[1Bil](https://huggingface.co/facebook/mms-1b). 

<Tip>

The MMS for ASR architecture is based on the Wav2Vec2 model, refer to [Wav2Vec2's documentation page](wav2vec2) for further 
details on how to finetune with models for various downstream tasks.

MMS-TTS uses the same model architecture as VITS, refer to [VITS's documentation page](vits) for API reference.
</Tip>
