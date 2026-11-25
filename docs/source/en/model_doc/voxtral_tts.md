<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-03-26 and contributed to Hugging Face Transformers on 2026-11-25.*

# VoxtralTts

[Voxtral TTS](https://huggingface.co/papers/2603.25551) is a multilingual text-to-speech model from
[Mistral AI](https://mistral.ai), released with 20 preset voices in English, French, Spanish, German, Italian,
Portuguese, Dutch, Arabic and Hindi.

The model has three parts:

- a Mistral language model (initialized from Ministral 3B), prompted with the embeddings of a voice preset and the text
  to read. Every audio frame (80 ms) it generates is fed back as the sum of the embeddings of its 37 codes.
- a flow-matching acoustic transformer. From the hidden state of the language model, a linear head greedily picks the
  semantic code of the next frame, and a small bidirectional transformer integrates a flow (7 Euler steps, with
  classifier-free guidance) to predict its 36 acoustic values, quantized to 21 levels.
- a causal codec decoder (convolutions and sliding-window transformers with ALiBi) that turns the codes into a 24 kHz
  waveform.

The released checkpoint does not include the codec encoder, so voices can only be selected among the presets (or
passed as precomputed voice embeddings); voice cloning from a reference recording is not supported.

You can find the original checkpoint under the [Mistral AI](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
organization. The checkpoint and the voice presets are released under the CC BY-NC 4.0 license.

## Usage

```python
import soundfile as sf
import torch

from transformers import AutoProcessor, VoxtralTtsForTextToSpeech

model_id = "mistralai/Voxtral-4B-TTS-2603"
processor = AutoProcessor.from_pretrained(model_id)
model = VoxtralTtsForTextToSpeech.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

print(processor.available_voices)
inputs = processor("Paris is a beautiful city!", voice="casual_male").to(model.device)
outputs = model.generate(**inputs)
sf.write("paris.wav", outputs.audio[0].float().cpu().numpy(), processor.sampling_rate)
```

### Batched generation

Prompts are left-padded, and each one can use a different voice.

```python
inputs = processor(
    ["The quick brown fox jumps over the lazy dog.", "Bonjour, je m'appelle Marie et j'habite à Lyon."],
    voice=["neutral_female", "fr_male"],
).to(model.device)
outputs = model.generate(**inputs, generator=torch.Generator(model.device).manual_seed(0))
for i, audio in enumerate(outputs.audio):
    sf.write(f"sample_{i}.wav", audio.float().cpu().numpy(), processor.sampling_rate)
```

`generate` returns the waveforms (`outputs.audio`), the generated codes (`outputs.audio_codes`) and the number of audio
frames of each sample (`outputs.audio_lengths`). The classifier-free guidance scale of the flow matching can be set
with `cfg_alpha` (defaults to 1.2), and `generator` makes the generation reproducible.

## VoxtralTtsConfig

[[autodoc]] VoxtralTtsConfig

## VoxtralTtsAcousticTransformerConfig

[[autodoc]] VoxtralTtsAcousticTransformerConfig

## VoxtralTtsCodecConfig

[[autodoc]] VoxtralTtsCodecConfig

## VoxtralTtsProcessor

[[autodoc]] VoxtralTtsProcessor
    - __call__

## VoxtralTtsModel

[[autodoc]] VoxtralTtsModel
    - forward

## VoxtralTtsForTextToSpeech

[[autodoc]] VoxtralTtsForTextToSpeech
    - forward
    - generate

## VoxtralTtsAcousticTransformer

[[autodoc]] VoxtralTtsAcousticTransformer
    - forward
    - generate_frame

## VoxtralTtsCodecModel

[[autodoc]] VoxtralTtsCodecModel
    - decode
