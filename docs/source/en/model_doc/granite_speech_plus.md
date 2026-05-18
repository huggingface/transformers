<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-23 and added to Hugging Face Transformers on 2026-04-29.*

# Granite Speech Plus


## Overview

Granite Speech Plus is a variant of [Granite Speech](./granite_speech) whose projector consumes the concatenation of
the encoder's final hidden states with an arbitrary subset of its intermediate hidden states (along the feature
dimension). The selected intermediate layers are controlled by the `cat_hidden_layers` config field on
[`GraniteSpeechPlusEncoderConfig`]; when it is `None`, the model behaves identically to Granite Speech. When it is set, the
projector's `encoder_hidden_size` must equal `encoder_config.hidden_dim * (len(cat_hidden_layers) + 1)`.

The rest of the architecture — speech encoder, query transformer projector, language model, and optional LoRA adapter
— is inherited unchanged from Granite Speech. See the [Granite Speech documentation](./granite_speech) for usage
examples; the same [`GraniteSpeechProcessor`] and [`GraniteSpeechFeatureExtractor`] are used here.

## Usage

Granite Speech Plus is a multimodal speech-to-text model that can transcribe audio, provide speaker annotation and word level timestamps by responding to text prompts. Here's how to use the different functions:


**Setup** — load the model and a test audio clip:

```python
import re
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000
MODEL_NAME = "ibm-granite/granite-speech-4.1-2b-plus"
```

Define the prompts used for the different tasks:

```python
SYSTEM_PROMPT = "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
ASR_PROMPT = "<|audio|> can you transcribe the speech into a written format?"
SAA_PROMPT = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
TS_PROMPT = "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
```

Load the model and define a general function for decoding the audio:

```python
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, device_map="auto")

@torch.inference_mode()
def transcribe(audio, prompt, max_new_tokens=2000, prefix_text=None):
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    extra = {"prefix_text": prefix_text} if prefix_text is not None else {}
    prompt_text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, **extra)
    inputs = processor(prompt_text, audio, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    output_text = processor.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    return output_text
```

Load some example audio data from the AMI dataset

```python
ds = load_dataset("diarizers-community/ami", "ihm", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, num_channels=1))

TEST_SAMPLE = 0
START_TIME, END_TIME = 5 * 60, 6 * 60
audio = ds["audio"][TEST_SAMPLE].get_samples_played_in_range(START_TIME, END_TIME)
```

**Task 1: ASR** — plain speech-to-text transcription:

```python
asr_text = transcribe(audio.data, ASR_PROMPT)
print(asr_text)
```

**Task 2: Speaker Attributed ASR** — transcription with speaker labels:

```python
saa_text = transcribe(audio.data, SAA_PROMPT)
for segment in re.split(r"(\[Speaker \d+\]:)", saa_text):
    print(segment.strip())
```

**Task 3: Word-level timestamps** — transcription with per-word timing:

The timestamps are given in centiseconds and are modulo 1000 (=10 seconds)
so we need to unwrap them by adding multiples of 10 seconds.

```python
ts_text = transcribe(audio.data, TS_PROMPT, max_new_tokens=10000)
ts_words = re.split(r"\[T:(\d+)\]", ts_text)
last_word_end_time = 0
offset_time = 0
for word, ts in zip(ts_words[::2], ts_words[1::2]):
    word_end_time = float(ts) / 100
    while word_end_time + offset_time < last_word_end_time:
        offset_time += 10
    last_word_end_time = word_end_time + offset_time
    print(f"{word}\t{last_word_end_time:.2f}s")
```

**Task 4: Incremental decoding** — transcribe segments while accumulating audio context:

```python
NUM_SEGMENTS = 3
previous_transcript = ""
all_audio = None

for k in range(NUM_SEGMENTS):
    t1 = START_TIME + (END_TIME - START_TIME) * k / NUM_SEGMENTS
    t2 = START_TIME + (END_TIME - START_TIME) * (k + 1) / NUM_SEGMENTS
    new_audio = ds["audio"][TEST_SAMPLE].get_samples_played_in_range(t1, t2)
    all_audio = new_audio.data if all_audio is None else torch.cat([all_audio, new_audio.data], dim=-1)
    saa_text = transcribe(all_audio, SAA_PROMPT, prefix_text=previous_transcript)
    print(f"{t1:06.2f}-{t2:06.2f}:\t{saa_text}")
    previous_transcript = (previous_transcript + " " + saa_text).strip()
```

## GraniteSpeechPlusConfig

[[autodoc]] GraniteSpeechPlusConfig

## GraniteSpeechPlusEncoderConfig

[[autodoc]] GraniteSpeechPlusEncoderConfig

## GraniteSpeechPlusForConditionalGeneration

[[autodoc]] GraniteSpeechPlusForConditionalGeneration
    - forward
    - get_audio_features
