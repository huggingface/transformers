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

# Qwen3 ASR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Qwen3 ASR is an automatic speech recognition model from Alibaba's Qwen team that combines a Qwen3 Omni-style audio encoder with a Qwen3 language model decoder for speech-to-text transcription. The model supports automatic language detection and multilingual transcription.

Available checkpoints:
- [bezzam/Qwen3-ASR-1.7B](https://huggingface.co/bezzam/Qwen3-ASR-1.7B)
- [bezzam/Qwen3-ASR-0.6B](https://huggingface.co/bezzam/Qwen3-ASR-0.6B)

See the original repository at [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) for more details.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

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

### Torch compile

The model can be compiled with `torch.compile` for faster inference.

```python
import time
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

model_id = "bezzam/Qwen3-ASR-1.7B"
num_warmup, num_runs = 5, 20

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

chat_template = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Mr. Quilter is the apostle of the middle classes.",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
            ],
        }
    ],
] * 4  # batch of 4
inputs = processor.apply_chat_template(
    chat_template, tokenize=True, return_dict=True,
).to("cuda", torch.bfloat16)

# Without compile
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.cuda.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Without compile: {no_compile_time:.4f}s")

# With compile
model = torch.compile(model)
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.cuda.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"With compile:    {compile_time:.4f}s")
print(f"Speedup: {no_compile_time / compile_time:.2f}x")
# ~1.70x speedup observed on A100
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

## Qwen3ASRProcessor

[[autodoc]] Qwen3ASRProcessor
    - __call__
    - apply_transcription_request
    - decode

## Qwen3ASRForConditionalGeneration

[[autodoc]] Qwen3ASRForConditionalGeneration
    - forward
    - get_audio_features
