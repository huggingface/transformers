<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-27.*

# MOSS-Transcribe-Diarize

## Overview

**MOSS-Transcribe-Diarize 0.9B** is an end-to-end audio understanding model for long-form multi-speaker transcription,
speaker diarization, and timestamps. It combines a Whisper-style encoder, a 4× frame merge step, a multi-modal
projector, and a Qwen3 language model.

The model checkpoint is available at:
[itazap/MOSS-Transcribe-Diarize-HF](https://huggingface.co/itazap/MOSS-Transcribe-Diarize-HF)

Key capabilities include:

* **Joint transcription and diarization** in a single pass, with segments formatted as
  `[start][S01]text[end]`.
* **Long-form audio** via Whisper-window chunking upstream in the processor; the model reassembles chunks per sample
  using `audio_chunk_mapping`.
* **Promptable transcription** through `apply_transcription_request` or the chat template.

This model was contributed by the Hugging Face team. See the
[model card](https://huggingface.co/itazap/MOSS-Transcribe-Diarize-HF) and the
[OpenMOSS repository](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) for more details.

## Usage

### Basic transcription and diarization

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = AutoModelForSeq2SeqLM.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF", device_map="auto")

inputs = processor.apply_transcription_request(
    "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
)

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)

decoded = processor.decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(decoded)
```

</hfoption>
</hfoptions>

### Advanced usage with the chat template

`apply_transcription_request` without a prompt is equivalent to a user turn that contains only audio:

```python
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "itazap/MOSS-Transcribe-Diarize-HF", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
decoded = processor.decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(decoded)
```

### Speaker-diarized transcription

Because each speaker turn is decoded as a `[start][S0N]text[end]` segment, `processor.decode` can also return the
parsed segments (`return_format="parsed"`) or just the concatenated transcription (`return_format="transcription_only"`),
so users have a clear view of the diarization output.

```python
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "itazap/MOSS-Transcribe-Diarize-HF", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/podcast_sample.wav"
inputs = processor.apply_transcription_request(audio_url)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids = outputs[:, inputs.input_ids.shape[1] :]

parsed = processor.decode(generated_ids, return_format="parsed")[0]
for segment in parsed:
    print(segment)
```

### Custom prompt and hotwords

`apply_transcription_request` accepts `keywords` to bias transcription toward specific terms (e.g. proper nouns or
domain-specific vocabulary), rendered as a `热词列表：`("hotword list") in the chat template. A custom `prompt` can be
passed alongside it for additional context.

```python
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "itazap/MOSS-Transcribe-Diarize-HF", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/languages_sample.wav"
inputs = processor.apply_transcription_request(audio_url, keywords=["Mandarin", "Yoruba"])
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(transcription[0])
```

### Batch inference

Pass a list of audio, and optionally a matching list of prompts and/or keywords, to transcribe a batch in one call.
Set an entry to `None` to skip it for that sample.

```python
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "itazap/MOSS-Transcribe-Diarize-HF", device_map="auto"
)

audio = [
    "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav",
    "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/languages_sample.wav",
]
keywords = [None, ["Mandarin", "Yoruba"]]

inputs = processor.apply_transcription_request(audio, keywords=keywords)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(transcription)
```

### Training

MOSS-Transcribe-Diarize can be trained with the loss returned by the model when `labels` are provided. Build a
conversation with the audio in the user turn and the target transcription as the assistant turn, then set
`output_labels=True` when tokenizing so the processor builds the labels for you (audio and padding tokens are masked
out with `-100`).

```python
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

processor = AutoProcessor.from_pretrained("itazap/MOSS-Transcribe-Diarize-HF")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "itazap/MOSS-Transcribe-Diarize-HF", device_map="auto"
)
model.train()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "[0.00][S01] Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel.[3.00]",
                },
            ],
        },
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device, dtype=model.dtype)

loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```

### Torch compile

The model can be compiled for faster inference/training. `device_map="auto"` wraps the forward pass with `accelerate`
hooks that cause repeated recompilation, so load the model directly on a single device instead.

On a B200, we observed a speed-up of ~2.1x for a batch size of 4.

```python
import torch
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
num_warmup = 3

processor = AutoProcessor.from_pretrained(model_id)
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
inputs = processor.apply_transcription_request([audio_url] * 4).to(model.device, torch.bfloat16)

# Warm-up and apply model
model.forward = torch.compile(model.forward)
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)
with torch.no_grad():
    _ = model(**inputs)
```

## MossTranscribeDiarizeConfig

[[autodoc]] MossTranscribeDiarizeConfig

## MossTranscribeDiarizeProcessor

[[autodoc]] MossTranscribeDiarizeProcessor
    - __call__
    - apply_chat_template
    - apply_transcription_request

## MossTranscribeDiarizeModel

[[autodoc]] MossTranscribeDiarizeModel
    - forward
    - get_audio_features

## MossTranscribeDiarizeForConditionalGeneration

[[autodoc]] MossTranscribeDiarizeForConditionalGeneration
    - forward
    - get_audio_features
