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
*This model was released on 2026-01-22 and added to Hugging Face Transformers on 2026-06-28.*

# Qwen3-TTS

## Overview

Qwen3-TTS is a series of text-to-speech models proposed in [Qwen3-TTS Technical Report](https://huggingface.co/papers/2601.15621) by the Qwen team, Alibaba Group.

The abstract from the paper is the following:

*We release Qwen3-TTS, a series of powerful speech generation models offering comprehensive support for voice clone, voice design, ultra-high-quality human-like speech generation, and natural language-based voice control. Powered by the self-developed Qwen3-TTS-Tokenizer-12Hz, it achieves efficient acoustic compression and high-dimensional semantic modeling of speech signals. Utilizing a discrete multi-codebook LM architecture, it realizes full-information end-to-end speech modeling that completely bypasses the information bottlenecks and cascading errors inherent in traditional LM+DiT schemes. It covers 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian) and supports streaming generation with end-to-end latency as low as 97ms.*

`Qwen3-TTS` checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/collections/Qwen/qwen3-tts).

## Usage Tips

### Basic Text-to-Speech

The processor bundles the text tokenizer, the speaker feature extractor, and the audio tokenizer. Build a conversation
with [`~Qwen3TTSProcessor.apply_chat_template`], generate the speech codes, then decode them to audio with
[`~Qwen3TTSProcessor.batch_decode`]:

```python
import torch
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Hello, how are you doing today?"}]},
]
inputs = processor.apply_chat_template(conversation)

codes, _ = model.generate(**inputs)
audio = processor.batch_decode(codes)
processor.save_audio(audio, "output.wav")
```

### Built-in Voice Presets

CustomVoice models ship with built-in voice presets. Set a `speaker` on the `user` message and a `language`. Use
`model.get_supported_speakers()` to list available voices for the loaded checkpoint.

```python
import torch
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Welcome to the future of voice technology."}],
        "language": "English",
        "speaker": "Ryan",
    },
]
inputs = processor.apply_chat_template(conversation)

codes, _ = model.generate(**inputs)
audio = processor.batch_decode(codes)
processor.save_audio(audio, "output_ryan.wav")
```

### Batch Inference

Pass a list of conversations to generate a batch:

```python
import torch
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

conversations = [
    [{"role": "user", "content": [{"type": "text", "text": "The weather is nice today."}]}],
    [{"role": "user", "content": [{"type": "text", "text": "I enjoy listening to music."}]}],
]
inputs = processor.apply_chat_template(conversations)

codes, _ = model.generate(**inputs)
audios = processor.batch_decode(codes)
processor.save_audio(audios, ["output_0.wav", "output_1.wav"])
```

### Voice Design with Natural Language Instructions

VoiceDesign models accept a natural language description of the desired voice as a `system` message:

```python
import torch
from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "A warm, friendly female voice with a slight British accent, speaking at a calm pace."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Good morning! Today is a beautiful day."}],
        "language": "English",
    },
]
inputs = processor.apply_chat_template(conversation)

codes, _ = model.generate(**inputs)
audio = processor.batch_decode(codes)
processor.save_audio(audio, "output_voice_design.wav")
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


## Qwen3TTSProcessor

[[autodoc]] Qwen3TTSProcessor
    - __call__
    - apply_chat_template
    - batch_decode
    - save_audio

## Qwen3TTSForConditionalGeneration

[[autodoc]] Qwen3TTSForConditionalGeneration
    - forward
    - generate


## Qwen3TTSFeatureExtractor

[[autodoc]] Qwen3TTSFeatureExtractor

## Qwen3TTSTokenizerMultiCodebookConfig

[[autodoc]] Qwen3TTSTokenizerMultiCodebookConfig

## Qwen3TTSTokenizerMultiCodebookCode2WavConfig

[[autodoc]] Qwen3TTSTokenizerMultiCodebookCode2WavConfig

## Qwen3TTSTokenizerMultiCodebookModel

[[autodoc]] Qwen3TTSTokenizerMultiCodebookModel
    - encode
    - decode

## Qwen3TTSTokenizerSingleCodebookConfig

[[autodoc]] Qwen3TTSTokenizerSingleCodebookConfig

## Qwen3TTSTokenizerSingleCodebookDiTConfig

[[autodoc]] Qwen3TTSTokenizerSingleCodebookDiTConfig

## Qwen3TTSTokenizerSingleCodebookEncoderConfig

[[autodoc]] Qwen3TTSTokenizerSingleCodebookEncoderConfig

## Qwen3TTSTokenizerSingleCodebookDecoderConfig

[[autodoc]] Qwen3TTSTokenizerSingleCodebookDecoderConfig

## Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig

[[autodoc]] Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig

## Qwen3TTSTokenizerSingleCodebookModel

[[autodoc]] Qwen3TTSTokenizerSingleCodebookModel
    - encode
    - decode

## Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel

[[autodoc]] Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel

## Qwen3TTSTokenizerSingleCodebookDecoderDiTModel

[[autodoc]] Qwen3TTSTokenizerSingleCodebookDecoderDiTModel
