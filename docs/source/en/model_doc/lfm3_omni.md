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
*This model was contributed to Hugging Face Transformers on 2026-11-03.*


# LFM3-Omni

## Overview

LFM3-Omni, by [Liquid AI](https://liquid.ai/), combines the [LFM3](./lfm3.md) language model, the [LFM3-Audio](./lfm3_audio.md) speech encoder and
codec depth decoder, and the [LFM3-VL](./lfm3_vl.md) image encoder. It accepts interleaved text, images,
and audio, and generates text, speech, or interleaved text and speech.

## Usage

Images, videos and 16 kHz audio clips enter through the chat template, which wraps every media placeholder in
`<|vision_start|>` / `<|vision_end|>` and lays each video out as one timestamped group of placeholders per temporal
patch. Speech output, batching and live transcription follow [LFM3-Audio](./lfm3_audio.md): `generation_mode`
opens the answer with `<|audio_start|>` (speech) or `<|mixed_start|>` (interleaved text and speech), and the
processor decodes `audio_sequences` to 24 kHz waveforms with the [Liquid codec](./liquid_codec.md).

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "LiquidAI/LFM3-Omni-2B"
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")

samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "audio", "url": samples + "librispeech_mr_quilter.wav"},
            {"type": "text", "text": "What is in the image, and what is said in the audio?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))

# answer in speech
inputs = processor.apply_chat_template(
    [
        {"role": "system", "content": "Perform TTS in American English using a female voice."},
        {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
    ],
    add_generation_prompt=True,
    generation_mode="audio",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, max_new_tokens=256)
audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)
processor.save_audio(audio, "tts.wav")
```

A video's soundtrack is opt-in: with `use_audio_in_video=True` (and `load_audio_from_video=True` in
`apply_chat_template` to read it from the file) the first clip after each video is interleaved after its frame
groups, otherwise every clip is its own media span.

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
            {"type": "text", "text": "What is said in this video?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    load_audio_from_video=True,
    use_audio_in_video=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)
```

For training, `output_labels=True` returns text `labels` (placeholders, `<|audio_start|>` and padding at `-100`) and,
with `audio_out`, frame-wise `audio_labels`; the loss averages text tokens and speech frames with their configured
weights.

## Lfm3OmniConfig

[[autodoc]] Lfm3OmniConfig

## Lfm3OmniAudioConfig

[[autodoc]] Lfm3OmniAudioConfig

## Lfm3OmniDepthDecoderConfig

[[autodoc]] Lfm3OmniDepthDecoderConfig

## Lfm3OmniAudioModel

[[autodoc]] Lfm3OmniAudioModel
    - forward

## Lfm3OmniDepthDecoderModel

[[autodoc]] Lfm3OmniDepthDecoderModel
    - forward

## Lfm3OmniDepthDecoderForMaskedLM

[[autodoc]] Lfm3OmniDepthDecoderForMaskedLM
    - forward

## Lfm3OmniProcessor

[[autodoc]] Lfm3OmniProcessor
    - __call__

## Lfm3OmniModel

[[autodoc]] Lfm3OmniModel
    - forward

## Lfm3OmniForConditionalGeneration

[[autodoc]] Lfm3OmniForConditionalGeneration
    - forward
    - generate
