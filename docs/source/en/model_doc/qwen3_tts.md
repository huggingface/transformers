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

The model checkpoints can be found [here](https://huggingface.co/collections/Qwen/qwen3-tts-hf).

<!-- TODO: contact Qwen to add checkpoints to their org! -->

Qwen3-TTS generates codes for a separate audio codec, which decodes them to a waveform. That codec is its own
model, documented in [Qwen3-TTS Tokenizer](./qwen3_tts_tokenizer). Its configuration is stored in
`model.config.audio_config`, defaulting to [`Qwen3TTSTokenizerConfig`].
The processor loads the codec alongside the text tokenizer and the feature extractor, so
[`~Qwen3TTSProcessor.decode`] works without setting it up yourself. Codec weights remain separate from the
TTS model weights; loading the TTS model does not instantiate a second codec.

This model was contributed by [Vandit Shah](https://huggingface.co/shahvandit).

## Usage Tips

### Basic Text-to-Speech

The processor bundles the text tokenizer, the speaker feature extractor, and the audio tokenizer. Build a conversation
with [`~Qwen3TTSProcessor.apply_chat_template`], generate the speech codes, then decode them to audio with
[`~Qwen3TTSProcessor.decode`]:

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "shahvandit/qwen3-tts-base-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Hello, how are you doing today?"}]},
]
inputs = processor.apply_chat_template(conversation)

codes = model.generate(**inputs).sequences
audio = processor.decode(codes)
processor.save_audio(audio, "output.wav")
```

### Built-in Voice Presets

CustomVoice models ship with built-in voice presets. Set a `speaker` on the `user` message and a `language`. Use
`model.get_supported_speakers()` to list available voices for the loaded checkpoint; it is empty on Base
checkpoints, which have no presets.

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "shahvandit/qwen3-tts-customvoice-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Welcome to the future of voice technology."}],
        "language": "English",
        "speaker": "Ryan",
    },
]
inputs = processor.apply_chat_template(conversation)

codes = model.generate(**inputs).sequences
audio = processor.decode(codes)
processor.save_audio(audio, "output_ryan.wav")
```

### Batch Inference

Pass a list of conversations to generate a batch:

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "shahvandit/qwen3-tts-base-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

conversations = [
    [{"role": "user", "content": [{"type": "text", "text": "The weather is nice today."}]}],
    [{"role": "user", "content": [{"type": "text", "text": "I enjoy listening to music."}]}],
]
inputs = processor.apply_chat_template(conversations)

codes = model.generate(**inputs).sequences
audios = processor.decode(codes)
processor.save_audio(audios, ["output_0.wav", "output_1.wav"])
```

### Voice Design with Natural Language Instructions

VoiceDesign models accept a natural language description of the desired voice as a `system` message.

```python
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "shahvandit/qwen3-tts-voicedesign-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

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

codes = model.generate(**inputs).sequences
audio = processor.decode(codes)
processor.save_audio(audio, "output_voice_design.wav")
```

### Voice Cloning

Base checkpoints carry a speaker encoder, which turns a reference recording into speaker conditioning.
Include mono reference audio at 24 kHz in the conversation. The processor extracts mel features,
and the model computes the speaker embedding internally.

```python
from datasets import Audio, load_dataset

from transformers import AutoProcessor, AutoModelForTextToWaveform


model_id = "shahvandit/qwen3-tts-base-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
reference_audio = dataset[0]["audio"]["array"]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This sentence is spoken in the voice of the reference clip."},
            {"type": "audio", "audio": reference_audio},
        ],
        "language": "English",
    },
]
inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(model.device, model.dtype)

codes = model.generate(**inputs).sequences
audio = processor.decode(codes)
processor.save_audio(audio, "output_cloned.wav")
```

Generation options can be supplied to `generate` or stored on `model.generation_config`; call-time values take
precedence. The residual code predictor has separate `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`,
and `subtalker_temperature` options. For example, `model.generate(**inputs, do_sample=True, top_p=0.9,
subtalker_top_p=0.9)` samples both stages with nucleus sampling.

### Pipeline usage

Qwen3-TTS is also available through the `text-to-speech` pipeline, which builds the inputs, generates the codes,
and decodes them to a waveform in a single call. Pass a conversation the same way as
[`~Qwen3TTSProcessor.apply_chat_template`]. The per-message `language` key selects the spoken language;
CustomVoice checkpoints additionally support the `speaker` key for built-in voice presets.

```python
import soundfile as sf
from transformers import pipeline

pipe = pipeline("text-to-speech", model="shahvandit/qwen3-tts-customvoice-hf", device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Welcome to the future of voice technology."}],
        "language": "English",
        "speaker": "Ryan",
    },
]
generate_kwargs = {"do_sample": True, "top_p": 0.9, "subtalker_top_p": 0.9}
output = pipe(conversation, generate_kwargs=generate_kwargs)
sf.write("output.wav", output["audio"], output["sampling_rate"])
```

### Training

Qwen3-TTS Base checkpoints can be trained with the loss returned by the model. Pass the target speech and its
transcript in a conversation with `output_labels=True`. The processor encodes the target speech into codec
labels and extracts its speaker features; the model computes the frozen speaker conditioning internally.
For larger datasets, cache the audio codes and speaker embeddings and pass them to the processor's
`audio_codes` and `speaker_embeddings` arguments instead of re-encoding the audio at every step.

```python
from datasets import Audio, load_dataset

from transformers import AutoModelForTextToWaveform, AutoProcessor


model_id = "shahvandit/qwen3-tts-base-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")
model.train()
# Enable activation recomputation; the model's support flag does not enable it automatically.
model.gradient_checkpointing_enable()
# Freeze reference-speaker conditioning while training the talker and code predictor.
model.speaker_encoder.requires_grad_(False)
model.speaker_encoder.eval()

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
example = dataset[0]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": example["text"]},
            {"type": "audio", "audio": example["audio"]["array"]},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    processor_kwargs={"output_labels": True},
).to(model.device, model.dtype)

outputs = model(**inputs)
outputs.loss.backward()
```

Use FP32 or BF16 for training because the released checkpoint's residual predictor can overflow in FP16.

### Torch compile

Generation is driven by two autoregressive transformers — the talker (`model.model`) and the code
predictor (`model.code_predictor.model`). Both can be compiled with `torch.compile`;
`generate` is then called as usual and uses the compiled graphs. Initial calls include compilation
overhead, and later calls may trigger recompilation. Benchmark after warmup on your target hardware;
compilation does not guarantee lower latency.

```python
import torch
from transformers import AutoProcessor, AutoModelForTextToWaveform

model_id = "shahvandit/qwen3-tts-base-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# compile the two decoder sub-models the generation loop drives
model.model = torch.compile(model.model)
model.code_predictor.model = torch.compile(model.code_predictor.model)

conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Hello, how are you doing today?"}]},
]
inputs = processor.apply_chat_template(conversation)

# the first generate() call triggers compilation (slow); later calls reuse the compiled graphs
codes = model.generate(**inputs).sequences
audio = processor.decode(codes)
processor.save_audio(audio, "output.wav")
```

## Qwen3TTSConfig

[[autodoc]] Qwen3TTSConfig

## Qwen3TTSSpeakerEncoderConfig

[[autodoc]] Qwen3TTSSpeakerEncoderConfig


## Qwen3TTSTalkerConfig

[[autodoc]] Qwen3TTSTalkerConfig

## Qwen3TTSTalkerCodePredictorConfig

[[autodoc]] Qwen3TTSTalkerCodePredictorConfig


## Qwen3TTSProcessor

[[autodoc]] Qwen3TTSProcessor
    - __call__
    - apply_chat_template
    - decode
    - save_audio

## Qwen3TTSForConditionalGeneration

[[autodoc]] Qwen3TTSForConditionalGeneration
    - forward
    - generate


## Qwen3TTSFeatureExtractor

[[autodoc]] Qwen3TTSFeatureExtractor
