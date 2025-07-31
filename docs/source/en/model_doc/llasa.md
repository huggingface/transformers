<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Llasa TTS

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

Llasa comprises a set of open-source text-to-speech (TTS) model developed by researchers at [Prof. Wei Xue's lab](https://huggingface.co/HKUSTAudio) at the Hong Kong University of Science and Technology (HKUST).
It was proposed in their paper [Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis](https://huggingface.co/papers/2502.04128).
Three models are available in different sizes: 1B, 3B, and 8B.

**Model Architecture:**
Llasa is designed with the standard text LLM paradigm in mind, consisting of two main components: (1) a tokenizer and (2) a single Transformer-based LLM.

1. The tokenizer combines a standard text LLM tokenizer (for the input text) and a speech tokenizer for representing waveforms as speech tokens. To this end, the authors introduced [X-Codec2](./xcodec2), which employs a single codebook (unlike [DAC](./dac) and [EnCodec](./encodec)) for convenient conversion between speech tokens and audio waveforms.
2. The Transformer-based LLM is trained to handle both conventional text and speech tokens. It outputs speech tokens, which can be decoded by the speech tokenizer. The Llasa LLM is initialized with parameters from:
   - [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for Llasa-1B  
   - [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) for Llasa-3B  
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for Llasa-8B

Dia is an encoder-decoder transformer based on the original transformer architecture. However, some more modern features such as
rotational positional embeddings (RoPE) are also included. For its text portion (encoder), a byte tokenizer is utilized while
for the audio portion (decoder), a pretrained codec model [DAC](./dac.md) is used - DAC encodes speech into discrete codebook
tokens and decodes them back into audio.

## Usage Tips

### Generation with Text

```python
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

text = ["[S1] Dia is an open weights text to dialogue model."]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# save audio to a file
outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.wav")

```

### Generation with Text and Audio (Voice Cloning)

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio + additional text you want as new audio
text = ["[S1] I know. It's going to save me a lot of money, I hope. [S2] I sure hope so for you."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt").to(torch_device)
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# retrieve actually generated audio and save to a file
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, "example_with_audio.wav")
```

### Training

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio
text = ["[S1] I know. It's going to save me a lot of money, I hope."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(
    text=text,
    audio=audio,
    generation=False,
    output_labels=True,
    padding=True,
    return_tensors="pt"
).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
out = model(**inputs)
out.loss.backward()
```


This model was contributed by [Jaeyong Sung](https://huggingface.co/buttercrab), [Arthur Zucker](https://huggingface.co/ArthurZ),
and [Anton Vlasjuk](https://huggingface.co/AntonV). The original code can be found [here](https://github.com/nari-labs/dia/).


## DiaConfig

[[autodoc]] DiaConfig

## DiaDecoderConfig

[[autodoc]] DiaDecoderConfig

## DiaEncoderConfig

[[autodoc]] DiaEncoderConfig

## DiaTokenizer

[[autodoc]] DiaTokenizer
    - __call__

## DiaFeatureExtractor

[[autodoc]] DiaFeatureExtractor
    - __call__

## DiaProcessor

[[autodoc]] DiaProcessor
    - __call__
    - batch_decode
    - decode

## DiaModel

[[autodoc]] DiaModel
    - forward

## DiaForConditionalGeneration

[[autodoc]] DiaForConditionalGeneration
    - forward
    - generate
