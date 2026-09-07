<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-27.*

# MOSS-Transcribe-Diarize

## Overview

**MOSS-Transcribe-Diarize 0.9B** is an end-to-end audio understanding model for long-form multi-speaker transcription,
speaker diarization, and timestamps. It combines a Whisper-style encoder, a 4× frame merge step, a multi-modal
projector, and a Qwen3 language model.

The model checkpoint is available at:
[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)

Key capabilities include:

* **Joint transcription and diarization** in a single pass, with segments formatted as
  `[start][S01]text[end]`.
* **Long-form audio** via Whisper-window chunking upstream in the processor; the model reassembles chunks per sample
  using `audio_chunk_mapping`.
* **Promptable transcription** through `apply_transcription_request` or the chat template.

This model was contributed by the Hugging Face team. See the
[model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) and the
[OpenMOSS repository](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) for more details.

## Usage

### Basic transcription and diarization

<hfoptions id="usage">
<hfoption id="AutoModel">

```py runnable:test_basic
from transformers import AutoModelForSeq2SeqLM, AutoProcessor


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = AutoModelForSeq2SeqLM.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto")

inputs = processor.apply_transcription_request(
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
)

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)

decoded = processor.decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
assert decoded == [  # nodoc
    "[0.00][S01] Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel.[3.00]"
]  # nodoc
print(decoded)
```

</hfoption>
</hfoptions>

### Advanced usage with the chat template

`apply_transcription_request` without a prompt is equivalent to a user turn that contains only audio:

```py runnable:test_advanced
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request(audio_url)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
        ],
    },
]

manual_inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)

for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
    assert manual_inputs[key].equal(inputs[key])  # nodoc

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)

decoded = processor.decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(decoded)
```

### Speaker-diarized transcription

Because each speaker turn is decoded as a `[start][S0N]text[end]` segment, `processor.decode` can also return the
parsed segments (`return_format="parsed"`) or just the concatenated transcription (`return_format="transcription_only"`),
so users have a clear view of the diarization output.

```py runnable:test_parsed_diarization
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav"
inputs = processor.apply_transcription_request(audio_url)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids = outputs[:, inputs.input_ids.shape[1] :]

parsed = processor.decode(generated_ids, return_format="parsed")[0]
assert [segment["Speaker"] for segment in parsed] == [1, 1, 1, 2, 2, 1, 1, 2, 2]  # nodoc
for segment in parsed:
    print(segment)

"""
{'Start': 0.26, 'End': 3.06, 'Speaker': 1, 'Content': "Hello everyone and welcome to the Vibe Voice Podcast. I'm your host Alex and today"}
{'Start': 3.06, 'End': 5.81, 'Speaker': 1, 'Content': "we're getting into one of the biggest debates in all sports, who's the greatest basketball"}
{'Start': 5.81, 'End': 7.77, 'Speaker': 1, 'Content': "player of all time. I'm so excited to have Sam here to talk about it with me."}
{'Start': 7.77, 'End': 9.61, 'Speaker': 2, 'Content': "Thanks so much for having me Alex. You're absolutely right, this question always brings"}
{'Start': 9.61, 'End': 10.61, 'Speaker': 2, 'Content': 'out some seriously strong feelings.'}
{'Start': 10.61, 'End': 13.16, 'Speaker': 1, 'Content': "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips"}
{'Start': 13.16, 'End': 15.73, 'Speaker': 1, 'Content': 'to finals, six championships. That kind of perfection is just incredible.'}
{'Start': 15.73, 'End': 17.94, 'Speaker': 2, 'Content': 'Oh man, the first thing that always pops in my head is that shot against the Cleveland Cavaliers'}
{'Start': 17.94, 'End': 20.32, 'Speaker': 2, 'Content': 'back in 89. Jordan just rises, hangs in the air forever, and just sinks it.'}
"""
```

### Custom prompt and hotwords

`apply_transcription_request` accepts a custom `prompt` to override [`~MossTranscribeDiarizeProcessor.default_transcription_prompt`].
Append a `热词提示：`("hotword hint") suffix with domain-specific terms (e.g. proper nouns) to nudge the model toward the
correct spelling.

```py runnable:test_hotwords
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav"

# Without hotwords
inputs = processor.apply_transcription_request(audio_url)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
without_hotwords = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(f"WITHOUT HOTWORDS: {without_hotwords[0]}")

# With hotwords: append a hint to the default prompt
hotword_prompt = processor.default_transcription_prompt + "热词提示：VibeVoice"
inputs = processor.apply_transcription_request(audio_url, prompt=hotword_prompt)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
with_hotwords = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
assert with_hotwords != without_hotwords  # nodoc
print(f"WITH HOTWORDS   : {with_hotwords[0]}")

"""
WITHOUT HOTWORDS: Vive's is a framework designed for generating expressive long-form multi-speaker conversational audio.
WITH HOTWORDS   : ViveVoice is a new framework designed for generating expressive long-form multi-speaker conversational audio
"""
```

### Batch inference

Pass a list of audio, and optionally a matching list of prompts, to transcribe a batch in one call. Set an entry to
`None` to fall back to the default prompt for that sample.

```py runnable:test_batch
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)

audio = [
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
]
prompts = [None, processor.default_transcription_prompt + "热词提示：VibeVoice"]

inputs = processor.apply_transcription_request(audio, prompt=prompts)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
assert len(transcription) == 2  # nodoc
print(transcription)

"""
['Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel.', 'ViveVoice is a new framework designed for generating expressive long-form multi-speaker conversational audio']
"""
```

### Training

MOSS-Transcribe-Diarize can be trained with the loss returned by the model when `labels` are provided. Build a
conversation with the audio in the user turn and the target transcription as the assistant turn, then set
`output_labels=True` when tokenizing so the processor builds the labels for you (audio and padding tokens are masked
out with `-100`).

```py runnable:test_training
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)
model.train()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
                {"type": "text", "text": processor.default_transcription_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "[0.00][S01] Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel.[3.00]",
                },
            ],
        },
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device, dtype=model.dtype)

loss = model(**inputs).loss
assert loss.item() > 0  # nodoc
print("Loss:", loss.item())
loss.backward()
```

### Torch compile

The model can be compiled for faster inference/training. `device_map="auto"` wraps the forward pass with `accelerate`
hooks that cause repeated recompilation, so load the model directly on a single device instead.

```py runnable:test_compile
import time

import torch

from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


num_warmup = 3
num_runs = 10

processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", dtype=torch.bfloat16
).to("cuda")

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request([audio_url] * 4).to(model.device, torch.bfloat16)

print("Warming up without compile...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)
torch.accelerator.synchronize()

print("Benchmarking without torch.compile...")
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.accelerator.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Average time without compile: {no_compile_time:.4f}s")

print("Compiling model...")
model.forward = torch.compile(model.forward)

print("Warming up with compile (includes graph capture)...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)
torch.accelerator.synchronize()

print("Benchmarking with torch.compile...")
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.accelerator.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"Average time with compile: {compile_time:.4f}s")

speedup = no_compile_time / compile_time
assert speedup > 1  # nodoc
print(f"Speedup: {speedup:.2f}x")
```

## MossTranscribeDiarizeConfig

[[autodoc]] MossTranscribeDiarizeConfig

## MossTranscribeDiarizeProcessor

[[autodoc]] MossTranscribeDiarizeProcessor
    - __call__
    - apply_chat_template
    - apply_transcription_request

## MossTranscribeDiarizeModel

[[autodoc]] MossTranscribeDiarizeModel
    - forward
    - get_audio_features

## MossTranscribeDiarizeForConditionalGeneration

[[autodoc]] MossTranscribeDiarizeForConditionalGeneration
    - forward
    - get_audio_features
