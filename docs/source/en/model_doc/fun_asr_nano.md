<!--Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
-->
*This model was published in HF papers on 2025-09-15 and contributed to Hugging Face Transformers on 2026-07-22.*

# Fun-ASR-Nano

## Overview

Fun-ASR-Nano is an 800M-parameter end-to-end speech recognition model developed by Alibaba DAMO Academy's FunAudioLLM team. It achieves state-of-the-art performance on Chinese, English, and Japanese ASR benchmarks while being significantly smaller than comparable models.

The model was proposed in [Fun-ASR: An Industrial-Grade Speech Recognition System](https://huggingface.co/papers/2509.12508).

### Architecture

Fun-ASR-Nano consists of three components:

1. **Audio Encoder** (SenseVoiceEncoderSmall): A 70-layer SANM (Self-Attention with FSMN Memory) encoder that combines multi-head self-attention with Feedforward Sequential Memory Networks for efficient speech feature extraction.

2. **Audio Adaptor**: A 2-layer Transformer that projects encoder outputs (512-dim) to the LLM dimension (1024-dim).

3. **Language Model** (Qwen3-0.6B): A 28-layer causal language model that generates transcription text autoregressively.

### Key Features

- **Chinese, English, and Japanese**, including 7 Chinese dialects and 26 regional accents
- **Hotword customization** for domain-specific vocabulary
- **Native punctuation** output (no separate punctuation model needed)

## Usage

### Single inference

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio_url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/en.mp3"
inputs = processor.apply_transcription_request(audio=audio_url, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

### Batch inference

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio_urls = [
    "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/zh.mp3",
    "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/en.mp3",
]
prompts = ["语音转写成中文：", "Transcribe the audio:"]
inputs = processor.apply_transcription_request(audio=audio_urls, prompt=prompts, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True))
```

### Custom prompts and hotwords

Passing `prompt` replaces the processor's default `"Transcribe the audio:"` prompt, so include the complete
transcription instruction together with any hotwords.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio_url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/zh.mp3"
hotword_prompt = (
    "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
    "**上下文信息：**\n\n\n"
    "热词列表：[开放时间]\n"
    "语音转写成中文："
)
inputs = processor.apply_transcription_request(
    audio=audio_url,
    prompt=hotword_prompt,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

### Training

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.train()

audio_url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/en.mp3"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the audio:"},
            {"type": "audio", "path": audio_url},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.",
            }
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    processor_kwargs={"output_labels": True},
).to(model.device)

loss = model(**inputs).loss
loss.backward()
```

### Inference with `torch.compile`

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, CompileConfig

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio_url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/example/en.mp3"
inputs = processor.apply_transcription_request(audio=audio_url, return_tensors="pt").to(model.device)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        cache_implementation="static",
        compile_config=CompileConfig(),
    )
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

## FunAsrNanoConfig

[[autodoc]] FunAsrNanoConfig

## FunAsrNanoEncoderConfig

[[autodoc]] FunAsrNanoEncoderConfig

## FunAsrNanoFeatureExtractor

[[autodoc]] FunAsrNanoFeatureExtractor
    - __call__

## FunAsrNanoProcessor

[[autodoc]] FunAsrNanoProcessor
    - __call__
    - apply_transcription_request

## FunAsrNanoEncoder

[[autodoc]] FunAsrNanoEncoder
    - forward

## FunAsrNanoModel

[[autodoc]] FunAsrNanoModel
    - forward

## FunAsrNanoForConditionalGeneration

[[autodoc]] FunAsrNanoForConditionalGeneration
    - forward
