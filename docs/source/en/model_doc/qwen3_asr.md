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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-24.*

# Qwen3 ASR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Qwen3 ASR is an automatic speech recognition model from Alibaba's Qwen team that combines a Whisper-style audio encoder with a Qwen3 language model decoder for speech-to-text transcription. The model supports automatic language detection and multilingual transcription.

A forced aligner model is also included. It can be used the timestamp a provided transcript and its audio. It uses the same audio encoder model with a classification head that predicts a word's length. This model can be used with the transcript from any ASR model (see the example below with Parakeet CTC).

Available checkpoints:
- [bezzam/Qwen3-ASR-1.7B](https://huggingface.co/bezzam/Qwen3-ASR-1.7B)
- [bezzam/Qwen3-ASR-0.6B](https://huggingface.co/bezzam/Qwen3-ASR-0.6B)
- [bezzam/Qwen3-ForcedAligner-0.6B](https://huggingface.co/bezzam/Qwen3-ForcedAligner-0.6B)

The following languages are supported:
- `Qwen3-ASR-1.7B` and `Qwen3-ASR-0.6B`: Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de), French (fr), Spanish (es), Portuguese (pt), Indonesian (id), Italian (it), Korean (ko), Russian (ru), Thai (th), Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi), Malay (ms), Dutch (nl), Swedish (sv), Danish (da), Finnish (fi), Polish (pl), Czech (cs), Filipino (fil), Persian (fa), Greek (el), Hungarian (hu), Macedonian (mk), Romanian (ro)
- `Qwen3-ForcedAligner-0.6B`: Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

See the original repository at [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) and the [report](https://huggingface.co/papers/2601.21337) for more details.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam) and [Muhammed Tariq](https://huggingface.co/mbtariq82).

## Usage

### Simple transcription

The simplest way to transcribe audio is with `apply_transcription_request`, which handles the chat template formatting for you.

```python
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
).to(model.device, model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

# Raw output includes language tag and <asr_text> marker
raw = processor.decode(generated_ids)[0]
print(f"Raw: {raw}")

# Parsed output: dict with "language" and "transcription"
parsed = processor.decode(generated_ids, return_format="parsed")[0]
print(f"Parsed: {parsed}")

# Extract only the transcription text
transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
print(f"Transcription: {transcription}")

"""
Raw: language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
Parsed: {'language': 'English', 'transcription': 'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'}
Transcription: Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
"""
```

### Language hint

You can provide a language hint to guide the model.

```python
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Without language hint (auto-detect)
inputs = processor.apply_transcription_request(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
print(f"Auto-detect: {processor.decode(generated_ids, return_format='transcription_only')[0]}")

# With language hint
inputs = processor.apply_transcription_request(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    language="Chinese",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
print(f"With hint:   {processor.decode(generated_ids, return_format='transcription_only')[0]}")
```

### Batch inference

Batch inference is possible by passing a list of audios and, if provided, a list of languages.

```python
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
audio = [
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
]

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(
    audio, language=["English", "Chinese"],
).to(model.device, model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
transcriptions = processor.decode(generated_ids, return_format="transcription_only")

for i, text in enumerate(transcriptions):
    print(f"Audio {i + 1}: {text}")
```

### Chat template

Qwen3 ASR also accepts chat template inputs (`apply_transcription_request` is a convenience wrapper for `apply_chat_template`):

```python
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# With language hint as system message
chat_template = [
    [
        {"role": "system", "content": [{"type": "text", "text": "English"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
                },
            ],
        },
    ],
]

inputs = processor.apply_chat_template(
    chat_template, tokenize=True, return_dict=True,
).to(model.device, model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
transcriptions = processor.decode(generated_ids, return_format="transcription_only")
for text in transcriptions:
    print(text)
```

### Training

Qwen3 ASR can be trained with the loss outputted by the model.

```python
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

chat_template = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    chat_template, tokenize=True, return_dict=True, output_labels=True,
).to(model.device, model.dtype)

loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```

### Forced alignment (word-level timestamping)

Use `Qwen3ASRForForcedAlignment` to obtain word-level timestamps from a transcript. First transcribe with the ASR model, then align with the forced aligner.

The following languages are supported: Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish.

Japanese requires the `nagisa` library, while Korean requires the `soynlp` library:
```
pip install nagisa soynlp
```

#### English

```python
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForForcedAlignment

asr_model_id = "bezzam/Qwen3-ASR-0.6B"
aligner_model_id = "bezzam/Qwen3-ForcedAligner-0.6B"

asr_processor = AutoProcessor.from_pretrained(asr_model_id)
asr_model = Qwen3ASRForConditionalGeneration.from_pretrained(asr_model_id, device_map="auto")

aligner_processor = AutoProcessor.from_pretrained(aligner_model_id)
aligner_model = Qwen3ASRForForcedAlignment.from_pretrained(
    aligner_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"

# Step 1: Transcribe
inputs = asr_processor.apply_transcription_request(audio=audio_url).to(asr_model.device, asr_model.dtype)
output_ids = asr_model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
transcript = parsed["transcription"]
language = parsed["language"] or "English"

# Step 2: Prepare alignment inputs
aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=audio_url, transcript=transcript, language=language,
)
aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)

# Step 3: Run forced aligner
with torch.inference_mode():
    outputs = aligner_model(**aligner_inputs)

# Step 4: Decode timestamps
timestamps = aligner_processor.decode_forced_alignment(
    logits=outputs.logits,
    input_ids=aligner_inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner_model.config.timestamp_token_id,
)[0]

for item in timestamps:
    print(f"{item['text']:<20} {item['start_time']:>8.3f}s → {item['end_time']:>8.3f}s")

"""
Word                  Start (s)    End (s)
------------------------------------------
Mr                        0.560      0.800
Quilter                   0.800      1.280
is                        1.280      1.440
the                       1.440      1.520
apostle                   1.520      2.080
...
"""
```

#### Chinese

For Chinese text, each character is aligned individually.

```python
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForForcedAlignment

asr_model_id = "bezzam/Qwen3-ASR-0.6B"
aligner_model_id = "bezzam/Qwen3-ForcedAligner-0.6B"

asr_processor = AutoProcessor.from_pretrained(asr_model_id)
asr_model = Qwen3ASRForConditionalGeneration.from_pretrained(asr_model_id, device_map="auto")

aligner_processor = AutoProcessor.from_pretrained(aligner_model_id)
aligner_model = Qwen3ASRForForcedAlignment.from_pretrained(
    aligner_model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"

# Step 1: Transcribe with language hint
inputs = asr_processor.apply_transcription_request(
    audio=audio_url, language="Chinese",
).to(asr_model.device, asr_model.dtype)
output_ids = asr_model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
transcript = parsed["transcription"]

# Step 2–4: Align and decode
aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=audio_url, transcript=transcript, language="Chinese",
)
aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)

with torch.inference_mode():
    outputs = aligner_model(**aligner_inputs)

timestamps = aligner_processor.decode_forced_alignment(
    logits=outputs.logits,
    input_ids=aligner_inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner_model.config.timestamp_token_id,
)[0]

for item in timestamps:
    print(f"{item['text']:<4} {item['start_time']:>8.3f}s → {item['end_time']:>8.3f}s")

"""
Char        Start (s)    End (s)
--------------------------------
甚               0.400      0.720
至               0.720      0.960
出               0.960      1.120
现               1.120      1.520
...
"""
```

#### With another ASR model

The forced aligner is model-agnostic, meaning the transcripts from any ASR system can be provided. Below is an example using [NVIDIA Parakeet CTC](https://huggingface.co/nvidia/parakeet-ctc-1.1b) for transcription.

**Single sample:**

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor, Qwen3ASRForForcedAlignment

# Load Parakeet CTC for transcription
parakeet_processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
parakeet_model = AutoModelForCTC.from_pretrained(
    "nvidia/parakeet-ctc-1.1b", torch_dtype="auto", device_map="cuda",
)

# Load Qwen3 Forced Aligner for timestamping
aligner_model_id = "bezzam/Qwen3-ForcedAligner-0.6B"
aligner_processor = AutoProcessor.from_pretrained(aligner_model_id)
aligner_model = Qwen3ASRForForcedAlignment.from_pretrained(
    aligner_model_id, torch_dtype=torch.bfloat16, device_map="cuda",
)

# Load audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=parakeet_processor.feature_extractor.sampling_rate))
audio_array = ds[0]["audio"]["array"]
sr = ds[0]["audio"]["sampling_rate"]

# Step 1: Transcribe with Parakeet
inputs = parakeet_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(
    parakeet_model.device, dtype=parakeet_model.dtype
)
with torch.inference_mode():
    outputs = parakeet_model.generate(**inputs)
transcript = parakeet_processor.decode(outputs)[0]
print(f"Transcript: {transcript}")

# Step 2: Align with Qwen3 Forced Aligner (expects 16kHz audio)
aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=audio_array, transcript=transcript, language="English",
)
aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)

with torch.inference_mode():
    aligner_outputs = aligner_model(**aligner_inputs)

timestamps = aligner_processor.decode_forced_alignment(
    logits=aligner_outputs.logits,
    input_ids=aligner_inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner_model.config.timestamp_token_id,
)[0]

for item in timestamps:
    print(f"{item['text']:<20} {item['start_time']:>8.3f}s → {item['end_time']:>8.3f}s")
```

**Batch:**

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor, Qwen3ASRForForcedAlignment

parakeet_processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
parakeet_model = AutoModelForCTC.from_pretrained(
    "nvidia/parakeet-ctc-1.1b", torch_dtype="auto", device_map="cuda",
)

aligner_model_id = "bezzam/Qwen3-ForcedAligner-0.6B"
aligner_processor = AutoProcessor.from_pretrained(aligner_model_id)
aligner_model = Qwen3ASRForForcedAlignment.from_pretrained(
    aligner_model_id, torch_dtype=torch.bfloat16, device_map="cuda",
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=parakeet_processor.feature_extractor.sampling_rate))
audio_arrays = [ds[i]["audio"]["array"] for i in range(3)]
sr = ds[0]["audio"]["sampling_rate"]

# Batch transcribe with Parakeet
inputs = parakeet_processor(audio_arrays, sampling_rate=sr, return_tensors="pt", padding=True).to(
    parakeet_model.device, dtype=parakeet_model.dtype
)
with torch.inference_mode():
    outputs = parakeet_model.generate(**inputs)
transcripts = parakeet_processor.decode(outputs)

# Batch align with Qwen3 Forced Aligner
aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=audio_arrays, transcript=transcripts, language="English",
)
aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)

with torch.inference_mode():
    aligner_outputs = aligner_model(**aligner_inputs)

batch_timestamps = aligner_processor.decode_forced_alignment(
    logits=aligner_outputs.logits,
    input_ids=aligner_inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner_model.config.timestamp_token_id,
)

for i, (transcript, timestamps) in enumerate(zip(transcripts, batch_timestamps)):
    print(f"\n[Sample {i}] {transcript}")
    for item in timestamps[:5]:
        print(f"  {item['text']:<20} {item['start_time']:>8.3f}s → {item['end_time']:>8.3f}s")
    if len(timestamps) > 5:
        print(f"  ... ({len(timestamps) - 5} more words)")
```

### Torch compile

Both the ASR and forced aligner models support `torch.compile` for faster inference. The forced aligner is an especially good fit for compilation because it runs a single forward pass (no autoregressive decoding). This makes it ideal for **bulk audio timestamping**: transcribe with any ASR model, then batch-align with the compiled forced aligner for maximum throughput.

#### Compiling the forced aligner

```python
import time
import torch
from transformers import AutoProcessor, Qwen3ASRForForcedAlignment

model_id = "bezzam/Qwen3-ForcedAligner-0.6B"
num_warmup, num_runs = 5, 20

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForForcedAlignment.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

# Prepare a batch of 4 samples
audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
transcript = "Mr. Quilter is the apostle of the middle classes."

aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
    audio=[audio_url] * 4,
    transcript=[transcript] * 4,
    language=["English"] * 4,
)
aligner_inputs = aligner_inputs.to("cuda", torch.bfloat16)

# Without compile
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**aligner_inputs)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**aligner_inputs)
torch.cuda.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Without compile: {no_compile_time:.4f}s")

# With compile
model = torch.compile(model)
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**aligner_inputs)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**aligner_inputs)
torch.cuda.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"With compile:    {compile_time:.4f}s")
print(f"Speedup: {no_compile_time / compile_time:.2f}x")
# ~2.5x speedup observed on A100
```

#### Compiling the ASR model (generate)

For autoregressive transcription, `torch.compile` accelerates the per-token forward passes inside `generate`.

```python
import time
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
num_warmup, num_runs = 3, 10
max_new_tokens = 256

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request(
    audio=[audio_url] * 4,  # batch of 4
).to("cuda", torch.bfloat16)

# Without compile
with torch.inference_mode():
    for _ in range(num_warmup):
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
torch.cuda.synchronize()
start = time.time()
with torch.inference_mode():
    for _ in range(num_runs):
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
torch.cuda.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Without compile: {no_compile_time:.4f}s")

# With compile
model = torch.compile(model)
with torch.inference_mode():
    for _ in range(num_warmup):
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
torch.cuda.synchronize()
start = time.time()
with torch.inference_mode():
    for _ in range(num_runs):
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
torch.cuda.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"With compile:    {compile_time:.4f}s")
print(f"Speedup: {no_compile_time / compile_time:.2f}x")
# ~2.5x speedup observed on A100
```

### Pipeline usage

```python
from transformers import pipeline

model_id = "bezzam/Qwen3-ASR-1.7B"
pipe = pipeline("any-to-any", model=model_id, device_map="auto")

chat_template = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
            },
        ],
    }
]
outputs = pipe(text=chat_template, return_full_text=False)
raw_text = outputs[0]["generated_text"]
print(f"Raw: {raw_text}")

# Use processor helper to extract transcription
transcription = pipe.processor.extract_transcription(raw_text)
print(f"Transcription: {transcription}")
```

## Qwen3ASRConfig

[[autodoc]] Qwen3ASRConfig


## Qwen3ASREncoderConfig

[[autodoc]] Qwen3ASREncoderConfig


## Qwen3ASRFeatureExtractor

[[autodoc]] Qwen3ASRFeatureExtractor
    - __call__

## Qwen3ASRProcessor

[[autodoc]] Qwen3ASRProcessor
    - __call__
    - apply_transcription_request
    - prepare_forced_aligner_inputs
    - decode_forced_alignment
    - decode

## Qwen3ASREncoder

[[autodoc]] Qwen3ASREncoder

## Qwen3ASRModel

[[autodoc]] Qwen3ASRModel

## Qwen3ASRForConditionalGeneration

[[autodoc]] Qwen3ASRForConditionalGeneration
    - forward
    - get_audio_features

## Qwen3ForcedAlignerConfig

[[autodoc]] Qwen3ForcedAlignerConfig

## Qwen3ASRForForcedAlignment

[[autodoc]] Qwen3ASRForForcedAlignment
    - forward
    - get_audio_features
