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
[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)

Key capabilities include:

* **Joint transcription and diarization** in a single pass, with segments formatted as
  `[start][S01]text[end]`.
* **Long-form audio** via Whisper-window chunking upstream in the processor; the model reassembles chunks per sample
  using `audio_chunk_mapping`.
* **Promptable transcription** through `apply_transcription_request` or the chat template.

This model was contributed by the Hugging Face team. See the
[model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) and the
[OpenMOSS repository](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) for more details.

## Usage

### Basic transcription and diarization

<hfoptions id="usage">
<hfoption id="AutoModel">

```py runnable:test_basic
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = AutoModelForSeq2SeqLM.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto")

inputs = processor.apply_transcription_request(
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
)

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)

decoded = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
assert len(decoded) == 1  # nodoc
print(decoded)
```

</hfoption>
</hfoptions>

### Advanced usage with the chat template

`apply_transcription_request` without a prompt is equivalent to a user turn that contains only audio:

```py runnable:test_advanced
# pytest-decorator: transformers.testing_utils.slow, transformers.testing_utils.require_torch
from transformers import AutoProcessor, MossTranscribeDiarizeForConditionalGeneration


processor = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize")
model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize", device_map="auto"
)

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request(audio_url)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
        ],
    },
]

manual_inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)

for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
    assert manual_inputs[key].equal(inputs[key])  # nodoc

inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)

decoded = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(decoded)
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
