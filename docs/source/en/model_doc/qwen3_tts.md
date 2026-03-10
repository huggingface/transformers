<!--Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-22 and added to Hugging Face Transformers on 2026-03-09.*

# Qwen3-TTS

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Qwen3-TTS is a series of text-to-speech models proposed in [Qwen3-TTS Technical Report](https://huggingface.co/papers/2601.15621) by the Qwen team, Alibaba Group.

The abstract from the paper is the following:

*We release Qwen3-TTS, a series of powerful speech generation models offering comprehensive support for voice clone, voice design, ultra-high-quality human-like speech generation, and natural language-based voice control. Powered by the self-developed Qwen3-TTS-Tokenizer-12Hz, it achieves efficient acoustic compression and high-dimensional semantic modeling of speech signals. Utilizing a discrete multi-codebook LM architecture, it realizes full-information end-to-end speech modeling that completely bypasses the information bottlenecks and cascading errors inherent in traditional LM+DiT schemes. It covers 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian) and supports streaming generation with end-to-end latency as low as 97ms.*

`Qwen3-TTS` checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/collections/Qwen/qwen3-tts).

## Usage Tips

### Basic Text-to-Speech

The standard workflow uses `Qwen3TTSForConditionalGeneration` to generate speech codes from text, then decodes them to audio using `Qwen3TTSTokenizerV2Model`:

```python
import torch
import soundfile as sf
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor, Qwen3TTSTokenizerV2Model

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
    model_id, subfolder="speech_tokenizer", device_map="auto"
)

text = "Hello, how are you doing today?"
inputs = processor(text=text, return_tensors="pt").to(model.device)

codes_list, _ = model.generate(
    input_ids=[inputs["input_ids"][0]],
    languages=["auto"],
)

audio_codes = codes_list[0].unsqueeze(0).to(speech_tokenizer.device)
with torch.no_grad():
    audio = speech_tokenizer.decode(audio_codes, return_dict=True).audio_values[0]

sf.write("output.wav", audio.cpu().float().numpy(), speech_tokenizer.output_sample_rate)
```

### Built-in Voice Presets

CustomVoice models ship with built-in voice presets. Pass a `speakers` list to `generate()`. Use `model.get_supported_speakers()` to list available voices for the loaded checkpoint.

```python
import torch
import soundfile as sf
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor, Qwen3TTSTokenizerV2Model

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
    model_id, subfolder="speech_tokenizer", device_map="auto"
)

text = "Welcome to the future of voice technology."
inputs = processor(text=text, return_tensors="pt").to(model.device)

codes_list, _ = model.generate(
    input_ids=[inputs["input_ids"][0]],
    languages=["English"],
    speakers=["Ryan"],
)

audio_codes = codes_list[0].unsqueeze(0).to(speech_tokenizer.device)
with torch.no_grad():
    audio = speech_tokenizer.decode(audio_codes, return_dict=True).audio_values[0]

sf.write("output_ryan.wav", audio.cpu().float().numpy(), speech_tokenizer.output_sample_rate)
```

### Batch Inference

Pass a list of `input_ids` and corresponding `languages` for batch generation:

```python
import torch
import soundfile as sf
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor, Qwen3TTSTokenizerV2Model

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
    model_id, subfolder="speech_tokenizer", device_map="auto"
)

texts = ["The weather is nice today.", "I enjoy listening to music."]
inputs_list = [processor(text=t, return_tensors="pt").to(model.device) for t in texts]

codes_list, _ = model.generate(
    input_ids=[inp["input_ids"][0] for inp in inputs_list],
    languages=["auto", "auto"],
)

for i, codes in enumerate(codes_list):
    audio_codes = codes.unsqueeze(0).to(speech_tokenizer.device)
    with torch.no_grad():
        audio = speech_tokenizer.decode(audio_codes, return_dict=True).audio_values[0]
    sf.write(f"output_{i}.wav", audio.cpu().float().numpy(), speech_tokenizer.output_sample_rate)
```

### Voice Design with Natural Language Instructions

VoiceDesign models accept a natural language description of the desired voice via `instruct_ids`:

```python
import torch
import soundfile as sf
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor, Qwen3TTSTokenizerV2Model

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
    model_id, subfolder="speech_tokenizer", device_map="auto"
)

instruct_text = "A warm, friendly female voice with a slight British accent, speaking at a calm pace."
text = "Good morning! Today is a beautiful day."

instruct_inputs = processor(text=instruct_text, return_tensors="pt").to(model.device)
text_inputs = processor(text=text, return_tensors="pt").to(model.device)

codes_list, _ = model.generate(
    input_ids=[text_inputs["input_ids"][0]],
    instruct_ids=[instruct_inputs["input_ids"][0]],
    languages=["English"],
)

audio_codes = codes_list[0].unsqueeze(0).to(speech_tokenizer.device)
with torch.no_grad():
    audio = speech_tokenizer.decode(audio_codes, return_dict=True).audio_values[0]

sf.write("output_voice_design.wav", audio.cpu().float().numpy(), speech_tokenizer.output_sample_rate)
```

### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
from transformers import Qwen3TTSForConditionalGeneration

model = Qwen3TTSForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

## Qwen3TTSConfig

[[autodoc]] Qwen3TTSConfig

## Qwen3TTSTalkerConfig

[[autodoc]] Qwen3TTSTalkerConfig

## Qwen3TTSTalkerCodePredictorConfig

[[autodoc]] Qwen3TTSTalkerCodePredictorConfig

## Qwen3TTSTokenizerV2Config

[[autodoc]] Qwen3TTSTokenizerV2Config

## Qwen3TTSTokenizerV2Code2WavConfig

[[autodoc]] Qwen3TTSTokenizerV2Code2WavConfig

## Qwen3TTSTokenizerV1Config

[[autodoc]] Qwen3TTSTokenizerV1Config

## Qwen3TTSProcessor

[[autodoc]] Qwen3TTSProcessor
    - __call__

## Qwen3TTSForConditionalGeneration

[[autodoc]] Qwen3TTSForConditionalGeneration
    - generate

## Qwen3TTSTalkerForConditionalGeneration

[[autodoc]] Qwen3TTSTalkerForConditionalGeneration
    - forward
    - generate

## Qwen3TTSTokenizerV2Model

[[autodoc]] Qwen3TTSTokenizerV2Model
    - encode
    - decode

## Qwen3TTSTokenizerV1Model

[[autodoc]] Qwen3TTSTokenizerV1Model
    - encode
    - decode
