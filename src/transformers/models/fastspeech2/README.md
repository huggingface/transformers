---
language: en
datasets:
- LJSpeech
- LibriTTS
tags:
- audio
- TTS
license: apache-2.0
---
# ontocord/fastspeech2-en
Modified version of the text-to-speech system [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech] (https://arxiv.org/abs/2006.04558v1).
## Installation
```
git clone https://github.com/ontocord/fastspeech2_hf
pip install transformers torchaudio

```
## Usage
The model can be used directly as follows:

```
# load the model and tokenizer
from fastspeech2_hf.modeling_fastspeech2 import FastSpeech2ForPretraining, FastSpeech2Tokenizer
model = FastSpeech2ForPretraining.from_pretrained("ontocord/fastspeech2-en")
tokenizer = FastSpeech2Tokenizer.from_pretrained("ontocord/fastspeech2-en")

# some helper routines
from IPython.display import Audio as IPAudio, display as IPdisplay
import torch
import torchaudio

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()
  if len(waveform.shape)==1:
    IPdisplay(IPAudio(waveform, rate=sample_rate))
    return 
  num_channels, num_frames = waveform.shape
  if num_channels <= 1:
    IPdisplay(IPAudio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    IPdisplay(IPAudio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

# set the g2p module for the tokenizer
tokenizer.set_g2p(model.fastspeech2.g2p)
# you can run in half mode on gpu.
model = model.cuda().half()
sentences = [
        "Advanced text to speech models such as Fast Speech can synthesize speech significantly faster than previous auto regressive models with comparable quality. The training of Fast Speech model relies on an auto regressive teacher model for duration prediction and knowledge distillation, which can ease the one to many mapping problem in T T S. However, Fast Speech has several disadvantages, 1, the teacher student distillation pipeline is complicated, 2, the duration extracted from the teacher model is not accurate enough, and the target mel spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality. ",
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition "
        "in being comparatively modern. ",
        "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process "
        "produced the block books, which were the immediate predecessors of the true printed book, "
        "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing. ",
        "And it is worth mention in passing that, as an example of fine typography, "
        "the earliest book printed with movable types, the Gutenberg, or \"forty-two line Bible\" of about 1455, "
        "has never been surpassed. ",
        "Printing, then, for our purpose, may be considered as the art of making books by means of movable types. "
        "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,",
       ]
batch = tokenizer(sentences, return_tensors="pt", padding=True)

model.eval()
with torch.no_grad():
  out = model(use_postnet=False, **batch)
wav =out[-2]
for line, phone, w in zip(sentences, tokenizer.batch_decode(batch['input_ids']), wav):
  print ("txt:", line)
  print ("phoneme:", phone)
  play_audio(w.type(torch.FloatTensor), model.config.sampling_rate)


```

##Github Code Repo 
Current code for this model can be found [here](https://github.com/ontocord/fastspeech2_hf)

This is a work in progress (WIP) port of the model and code from  
[this repo] (https://github.com/ming024/FastSpeech2).

The datasets on which this model was trained:
- LJSpeech: a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- LibriTTS: a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

