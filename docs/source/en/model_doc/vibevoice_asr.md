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
*This model was released on 2026-01-26 and added to Hugging Face Transformers on 2026-03-02.*

# VibeVoice ASR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

VibeVoice ASR is an automatic speech recognition model from Microsoft that combines acoustic and semantic audio tokenizers with a causal language model for robust speech-to-text transcription. The model uses VibeVoice's acoustic and semantic tokenizers that process audio at 24kHz, paired with a Qwen2-based language decoder for generating transcriptions. See the [technical report](https://huggingface.co/papers/2601.18184) for more details.

The model checkpoint is available at: [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)

Highlights:

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.
  
- **🌍 Multilingual & Code-Switching Support**:
  It supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Language distribution can be found [here](#language-distribution).

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Usage

The model supports various automatic speech recognition functionalities.

### Speaker-timestamped transcription

A notable feature of VibeVoice ASR is its ability to transcribe multi-speaker content, denoting who spoke and when.

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Prepare inputs using `apply_transcription_request`
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
).to(model.device, model.dtype)

# Apply model
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids)[0]
print("\n" + "=" * 60)
print("RAW OUTPUT")
print("=" * 60)
print(transcription)

transcription = processor.decode(generated_ids, return_format="parsed")[0]
print("\n" + "=" * 60)
print("TRANSCRIPTION (list of dicts)")
print("=" * 60)
for speaker_transcription in transcription:
    print(speaker_transcription)

# Remove speaker labels, only get raw transcription
transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
print("\n" + "=" * 60)
print("TRANSCRIPTION ONLY")
print("=" * 60)
print(transcription)

"""
============================================================
RAW OUTPUT
============================================================
<|im_start|>assistant
[{"Start":0,"End":15.43,"Speaker":0,"Content":"Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me."},{"Start":15.43,"End":21.05,"Speaker":1,"Content":"Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings."},{"Start":21.05,"End":31.66,"Speaker":0,"Content":"Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible."},{"Start":31.66,"End":40.93,"Speaker":1,"Content":"Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."}]<|im_end|>
<|endoftext|>

============================================================
TRANSCRIPTION (list of dicts)
============================================================
{'Start': 0, 'End': 15.43, 'Speaker': 0, 'Content': "Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me."}
{'Start': 15.43, 'End': 21.05, 'Speaker': 1, 'Content': "Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings."}
{'Start': 21.05, 'End': 31.66, 'Speaker': 0, 'Content': "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible."}
{'Start': 31.66, 'End': 40.93, 'Speaker': 1, 'Content': "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."}

============================================================
TRANSCRIPTION ONLY
============================================================
Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me. Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings. Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible. Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it.
"""
```

The VibeVoice ASR model is trained to generate a string that resembles a JSON structure. The flag `return_format="parsed"` tries to return the generated output as a list of dicts, while `return_format="transcription_only"` tries to extract only the transcribed audio. If they fail, the generated output is returned as-is.

### Providing context

It is also possible to provide context. This can be useful if certain words cannot be transcribed correctly, such as proper nouns.

Below we transcribe an audio where the speaker (with a German accent) talks about VibeVoice, comparing with and without the context "About VibeVoice".

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Without context
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
print(f"WITHOUT CONTEXT: {transcription}")

# With context
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    prompt="About VibeVoice",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
print(f"WITH CONTEXT   : {transcription}")

"""
WITHOUT CONTEXT: Revevoices is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio.
WITH CONTEXT   : VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio.
"""
```

### Batch inference

Batch inference is possible by passing a list of audio and, if provided, a list of prompts. The number of audio inputs and prompts should match (for prompts, you can set an entry to `None` if not needed for a given audio).

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
audio = [
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav"
]
prompts = ["About VibeVoice", None]

processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

inputs = processor.apply_transcription_request(audio, prompt=prompts).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids, return_format="transcription_only")

print(transcription)
```

### Adjusting tokenizer chunk (e.g. if out-of-memory)

A key feature of VibeVoice ASR is that it can transcribe up to 60 minutes of continuous audio. This is done by chunking audio into 60-second segments (1440000 samples at 24kHz) and caching the convolution states between each segment.

However, if chunks of 60 seconds are too large for your device, the `acoustic_tokenizer_chunk_size` argument passed to `generate` can be adjusted. *Note it should be a multiple of the hop length (3200 for the original acoustic tokenizer).*

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

acoustic_tokenizer_chunk_size = 64000    # default is 1440000 (60s @ 24kHz)
model_id = "microsoft/VibeVoice-ASR-HF"
audio = [
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav"
]
prompts = ["About VibeVoice", None]

processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

inputs = processor.apply_transcription_request(audio, prompt=prompts).to(model.device, model.dtype)
output_ids = model.generate(**inputs, acoustic_tokenizer_chunk_size=acoustic_tokenizer_chunk_size)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids, return_format="transcription_only")
print(transcription)
```

### Chat template

VibeVoice ASR also accepts chat template inputs (`apply_transcription_request` is actually a wrapper for `apply_chat_template` for convenience):
```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")

chat_template = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "About VibeVoice"},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
                },
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
).to(model.device, model.dtype)

output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.decode(generated_ids, return_format="transcription_only")
print(transcription)
```

### Training

VibeVoice ASR can be trained with the loss outputted by the model.

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

# Prepare batch of 2
# -- NOTE: the original model is trained to output transcription, speaker ID, and timestamps in JSON-like format. Below we are only using the transcription text as the label
chat_template = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello everyone and welcome to the VibeVoice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me. Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings. Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible. Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
                },
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device, model.dtype)

loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```

### Torch compile

The model can be compiled for faster inference/training.
```python
import time
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"

num_warmup = 5
num_runs = 20

# Load processor + model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,).to("cuda")

# Prepare static inputs
chat_template = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
] * 4  # batch size 4
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
).to("cuda", torch.bfloat16)

# Benchmark without compile
print("Warming up without compile...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)

torch.cuda.synchronize()

print("\nBenchmarking without torch.compile...")
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.cuda.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Average time without compile: {no_compile_time:.4f}s")

# Benchmark with compile
print("\nCompiling model...")
model = torch.compile(model)

print("Warming up with compile (includes graph capture)...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)

torch.cuda.synchronize()

print("\nBenchmarking with torch.compile...")
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.cuda.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"Average time with compile: {compile_time:.4f}s")

speedup = no_compile_time / compile_time
print(f"\nSpeedup: {speedup:.2f}x")
```

### Pipeline usage

The model can be used as a pipeline, but you will have to define your own methods for parsing the raw output.

```python
from transformers import pipeline

model_id = "microsoft/VibeVoice-ASR-HF"
pipe = pipeline("any-to-any", model=model_id, device_map="auto")
chat_template = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "About VibeVoice"},
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
            },
        ],
    }
]
outputs = pipe(text=chat_template, return_full_text=False)

print("\n" + "=" * 60)
print("RAW PIPELINE OUTPUT")
print("=" * 60)
print(outputs[0]["generated_text"])

print("\n" + "=" * 60)
print("DICT OUTPUT")
print("=" * 60)
dict_output = pipe.processor.extract_speaker_dict(outputs[0]["generated_text"])
print(dict_output)

print("\n" + "=" * 60)
print("TRANSCRIPT OUTPUT")
print("=" * 60)
transcription = pipe.processor.extract_transcription(outputs[0]["generated_text"])
print(transcription)
```

## VibeVoiceAsrConfig

[[autodoc]] VibeVoiceAsrConfig

## VibeVoiceAsrProcessor

[[autodoc]] VibeVoiceAsrProcessor
    - __call__
    - apply_transcription_request
    - decode

## VibeVoiceAsrForConditionalGeneration

[[autodoc]] VibeVoiceAsrForConditionalGeneration
    - forward
    - get_audio_features
