<!--Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
-->
*This model was published in HF papers on 2025-09-15 and contributed to Hugging Face Transformers on 2026-06-24.*

# Fun-ASR-Nano

## Overview

Fun-ASR-Nano is an 800M-parameter end-to-end speech recognition model developed by Alibaba DAMO Academy's FunAudioLLM team. It achieves state-of-the-art performance on Chinese, English, and Japanese ASR benchmarks while being significantly smaller than comparable models.

The model was proposed in [Fun-ASR: An Industrial-Grade Speech Recognition System](https://huggingface.co/papers/2509.12508).

### Architecture

Fun-ASR-Nano consists of four components:

1. **Audio Encoder** (SenseVoiceEncoderSmall): A 70-layer SANM (Self-Attention with FSMN Memory) encoder that combines multi-head self-attention with Feedforward Sequential Memory Networks for efficient speech feature extraction.

2. **Audio Adaptor**: A 2-layer Transformer that projects encoder outputs (512-dim) to the LLM dimension (1024-dim).

3. **Language Model** (Qwen3-0.6B): A 28-layer causal language model that generates transcription text autoregressively.

### Key Features

- **31 languages**: Chinese (including 7 dialects and 26 regional accents), English, Japanese, and 20+ European languages
- **Hotword customization** for domain-specific vocabulary
- **Native punctuation** output (no separate punctuation model needed)
- **Streaming support** for chunk-by-chunk inference

## Usage

### Single inference

```python
import torch
import librosa
from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio, _ = librosa.load("audio.wav", sr=16000)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the audio:"},
            {"type": "audio"},
        ],
    },
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Batch inference

```python
import torch
import librosa
from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

audio_zh, _ = librosa.load("zh.wav", sr=16000)
audio_en, _ = librosa.load("en.wav", sr=16000)

conversations = [
    [{"role": "user", "content": [{"type": "text", "text": "语音转写成中文："}, {"type": "audio"}]}],
    [{"role": "user", "content": [{"type": "text", "text": "Transcribe the audio:"}, {"type": "audio"}]}],
]
texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
inputs = processor(text=texts, audio=[audio_zh, audio_en], sampling_rate=16000, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

### Training

```python
import torch
import librosa
from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.train()

audio, _ = librosa.load("audio.wav", sr=16000)
conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Transcribe the audio:"}, {"type": "audio"}]},
    {"role": "assistant", "content": "The transcription of the audio."},
]
text = processor.apply_chat_template(conversation, tokenize=False)
inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt").to(model.device)
# Build labels from the input ids (mask out the prompt positions with -100 in your data pipeline).
inputs["labels"] = inputs["input_ids"].clone()

loss = model(**inputs).loss
loss.backward()
```

### Inference with `torch.compile`

```python
import torch
import librosa
from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = FunAsrNanoForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.forward = torch.compile(model.forward)

audio, _ = librosa.load("audio.wav", sr=16000)
conversation = [{"role": "user", "content": [{"type": "text", "text": "Transcribe the audio:"}, {"type": "audio"}]}]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=200)
generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
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

## FunAsrNanoEncoder

[[autodoc]] FunAsrNanoEncoder
    - forward

## FunAsrNanoModel

[[autodoc]] FunAsrNanoModel
    - forward

## FunAsrNanoForConditionalGeneration

[[autodoc]] FunAsrNanoForConditionalGeneration
    - forward
