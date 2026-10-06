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
*This model was contributed to Hugging Face Transformers on 2026-10-06.*

# Nemotron 3.5 Transcribe

## Overview

Nemotron 3.5 Transcribe is a speech language model for multilingual transcription, speaker-attributed transcription,
speech translation and audio question answering. Its audio encoder runs two models in parallel on the same log-mel
spectrogram:

- a speech recognition encoder ([`Nemotron3_5TranscribeAsrEncoder`]), a Transformer over stacked spectrogram frames with
  the queries and keys of each attention head normalized,
- a speaker diarization model ([Nemotron 3 Diarization](./nemotron3_diarization)), which predicts the activity of up
  to eight speakers.

The speaker activities are added to the speech recognition encoder outputs as fixed speaker embeddings, so that each
audio embedding also says which speakers are active. A linear projector maps one audio embedding every 80 ms into a
Nemotron-H language model (hybrid Mamba-2, attention and mixture-of-experts layers), which generates the text.

The model supports the original single-window inference path for audio up to 40 seconds. It is not bit-exact with the
original implementation: Transformers computes the feature extractor constants in float32 instead of loading their
bfloat16-rounded checkpoint values, retains the released Sortformer boundary behavior at the final padded frame, and
uses the standard Nemotron-H bfloat16 parameter dtypes. Long-form windowing and the Sortformer speaker cache are not
included yet.

## Usage

### Transcription

```python
import torch
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration

model_id = "nvidia/Nemotron-3.5-Transcribe"
processor = AutoProcessor.from_pretrained(model_id)
model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

inputs = processor.apply_transcription_request(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
).to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

### Speaker-attributed transcription

The model starts each speaker turn with a `<spk:N>` tag when the prompt asks for it. Keep the special tokens when you
decode.

```python
prompt = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no punctuation "
    "and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; "
    "assign one stable tag per speaker and keep speaker identities consistent. Output only the speaker-tagged transcript."
)
inputs = processor.apply_transcription_request(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    prompt=prompt,
).to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False))
```

### Chat template

The processor chat template reproduces the training prompt format: an empty system turn, the text and audio of a
user turn joined by a space, and thinking disabled. Do not add a system prompt.

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Translate the audio into German."},
            {"type": "audio", "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"},
        ],
    }
]
inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)
```

## Nemotron3_5TranscribeConfig

[[autodoc]] Nemotron3_5TranscribeConfig

## Nemotron3_5TranscribeAsrEncoderConfig

[[autodoc]] Nemotron3_5TranscribeAsrEncoderConfig

## Nemotron3_5TranscribeEncoderConfig

[[autodoc]] Nemotron3_5TranscribeEncoderConfig

## Nemotron3_5TranscribeProcessor

[[autodoc]] Nemotron3_5TranscribeProcessor
    - __call__
    - apply_transcription_request

## Nemotron3_5TranscribeAsrEncoder

[[autodoc]] Nemotron3_5TranscribeAsrEncoder
    - forward

## Nemotron3_5TranscribeEncoder

[[autodoc]] Nemotron3_5TranscribeEncoder
    - forward

## Nemotron3_5TranscribeModel

[[autodoc]] Nemotron3_5TranscribeModel
    - forward
    - get_audio_features

## Nemotron3_5TranscribeForConditionalGeneration

[[autodoc]] Nemotron3_5TranscribeForConditionalGeneration
    - forward
