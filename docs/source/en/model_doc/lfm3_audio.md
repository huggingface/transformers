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
*This model was contributed to Hugging Face Transformers on 2026-11-19.*


# LFM3-Audio

## Overview

LFM3-Audio is the speech-and-text model built on [LFM3](./lfm3.md) by [Liquid AI](https://liquid.ai/). One LFM3 language
model reads speech through a Liquid speech encoder and writes speech as frames of 12 codebook codes through a depth
decoder, alongside text. Speech frames are turned into audio by the [Liquid codec](./liquid_codec.md).

## Architecture

- **Speech encoder**: three stride-2 convolutions over the 128-bin log-mel spectrogram, then a causal 17-layer LFM3
  model with a sliding attention window of 32 frames. Every 8 spectrogram frames (80 ms) become one language token.
- **Language model**: LFM3, with the text head tied to the input embeddings.
- **Speech output**: every frame is 12 codes at 12.5 Hz. Their embeddings are summed on the way in. On the way out a
  depth decoder projects the language model's hidden state into one slot per codebook and a 3-layer bidirectional
  LFM3 model fills the codebooks in a random order over 3 passes (MaskGIT): in training a random subset of each
  frame's codebooks is hidden and predicted from the others.
- **Codec**: the [Liquid codec](./liquid_codec.md), 12 codebooks of 2016 codes at 12.5 frames per second, 24 kHz.

## Transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")

audio_url = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
)
conversation = [
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "audio", "url": audio_url}]},
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=128)
transcription = processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(transcription)
```

## Text to speech

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")

conversation = [
    {"role": "system", "content": "Perform TTS in American English using a female voice."},
    {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
]
# the answer opens with <|audio_start|>: speech until an end-of-audio frame
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    generation_mode="audio",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)
processor.save_audio(audio, "out/tts.wav")
```

## Speech conversation

The answer interleaves text and speech; the text is the transcript of the speech.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")

audio_url = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
)
conversation = [
    {"role": "system", "content": "Respond with interleaved text and audio."},
    {"role": "user", "content": [{"type": "audio", "url": audio_url}]},
]
# the answer opens with <|mixed_start|>: blocks of text, then of speech
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    generation_mode="interleaved",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=1024)
text = processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(text)
audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)
processor.save_audio(audio, "out/answer.wav")
```

## Batching

A batch can mix outputs: pass one `generation_mode` per conversation. Pad prompts on the left. With `"text"`, the
model may still switch to speech; to forbid it, pass
`suppress_tokens=[model.config.audio_start_token_id, model.config.mixed_start_token_id]` to `generate`.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")

audio_url = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
)
transcribe = [
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "audio", "url": audio_url}]},
]
speak = [
    {"role": "system", "content": "Perform TTS in American English using a female voice."},
    {"role": "user", "content": "Hello there!"},
]
inputs = processor.apply_chat_template(
    [transcribe, speak],
    add_generation_prompt=True,
    generation_mode=["text", "audio"],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=128)
transcription = processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(transcription)
# one waveform per speech segment: only the second row speaks
audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)
processor.save_audio(audio, ["out/batched_speech.wav"])
```

## Live transcription

Pass `input_features` as an iterator to transcribe a stream: each item is the processor's output on a 0.96 s chunk,
and the text of each chunk goes to the streamer as it is written. Mark the final chunk with `is_last_audio_chunk=True`.
One context holds at most 30 s: pass the returned caches to the next `generate` call to continue. Streams are not
batched.

```python
import itertools

from transformers import AutoModelForMultimodalLM, AutoProcessor, TextStreamer
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision)

audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
    sampling_rate=16000,
)
window, hop = processor.num_samples_per_audio_chunk, processor.audio_chunk_hop_length


def input_features_generator():
    # a microphone would yield its samples as they come; consecutive windows overlap by `window - hop` samples
    starts = range(0, len(audio) - (window - hop), hop)
    for start in starts:
        yield processor(
            audio=audio[start : start + window],
            sampling_rate=16000,
            is_last_audio_chunk=start == starts[-1],
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)


prompt = processor.apply_chat_template(
    [{"role": "system", "content": "Perform live ASR."}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

# one context holds at most 30 s of audio, as in training: past it, a new context continues the stream
chunks = input_features_generator()
caches = {}
while segment := list(itertools.islice(chunks, processor.max_audio_chunks_per_context)):
    outputs = model.generate(
        **prompt, input_features=iter(segment), streamer=streamer, return_dict_in_generate=True, **caches
    )
    caches = {"padding_cache": outputs.padding_cache, "encoder_cache": outputs.encoder_cache}
```

## Training

The model trains on text and speech answers at once. Put the target in the assistant turn, speech as an `audio_out`
block holding the 24 kHz speech (or its codes), and pass `output_labels=True`: the codec encodes the speech, and the
processor returns the text targets as `labels` and the speech frames as `audio_labels`.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")
model.train()

audio_url = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
)
transcript = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
transcription = [
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "audio", "url": audio_url}]},
    {"role": "assistant", "content": transcript},
]
text_to_speech = [
    {"role": "system", "content": "Perform TTS in American English using a female voice."},
    {"role": "user", "content": transcript},
    {"role": "assistant", "content": [{"type": "audio_out", "url": audio_url}]},
]

inputs = processor.apply_chat_template(
    [transcription, text_to_speech],
    tokenize=True,
    return_dict=True,
    # only train on the assistant outputs: mask the labels outside the assistant turns
    return_assistant_tokens_mask=True,
    return_tensors="pt",
    processor_kwargs={"output_labels": True},
).to(model.device, dtype=model.dtype)
inputs["labels"][inputs.pop("assistant_masks") == 0] = -100

outputs = model(**inputs)
outputs.loss.backward()
print(f"loss {outputs.loss.item():.3f}")
print(f"text {outputs.text_loss.item():.3f}, speech {outputs.audio_loss.item():.3f}")
```

## Loading parts of the model

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id, revision = "LiquidAI/LFM3-Audio-2B", "refs/pr/26"

# full model: speech encoder, LM and depth decoder
model = AutoModelForMultimodalLM.from_pretrained(
    model_id, revision=revision, device_map="auto"
)
print(f"{model.num_parameters() / 1e9:.2f}B")  # 1.98B
# transcription only: no depth decoder
asr_model = AutoModelForMultimodalLM.from_pretrained(
    model_id, revision=revision, depth_decoder_config=None, device_map="auto"
)
print(f"{asr_model.num_parameters() / 1e9:.2f}B")  # 1.84B
# speech output only: no speech encoder
tts_model = AutoModelForMultimodalLM.from_pretrained(
    model_id, revision=revision, audio_config=None, device_map="auto"
)
print(f"{tts_model.num_parameters() / 1e9:.2f}B")  # 1.86B

# full codec: encoder and decoder
processor = AutoProcessor.from_pretrained(model_id, revision=revision, device_map="auto")
print(f"{processor.audio_tokenizer.num_parameters() / 1e6:.1f}M")  # 73.6M
# speech output only: codec decoder
tts_processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, device_map="auto", audio_tokenizer_kwargs={"encoder_config": None}
)
print(f"{tts_processor.audio_tokenizer.num_parameters() / 1e6:.1f}M")  # 36.4M
```

## Lfm3AudioConfig

[[autodoc]] Lfm3AudioConfig

## Lfm3AudioAudioConfig

[[autodoc]] Lfm3AudioAudioConfig

## Lfm3AudioDepthDecoderConfig

[[autodoc]] Lfm3AudioDepthDecoderConfig

## Lfm3AudioFeatureExtractor

[[autodoc]] Lfm3AudioFeatureExtractor
    - __call__

## Lfm3AudioAudioModel

[[autodoc]] Lfm3AudioAudioModel
    - forward

## Lfm3AudioDepthDecoderModel

[[autodoc]] Lfm3AudioDepthDecoderModel
    - forward

## Lfm3AudioDepthDecoderForMaskedLM

[[autodoc]] Lfm3AudioDepthDecoderForMaskedLM
    - forward

## Lfm3AudioProcessor

[[autodoc]] Lfm3AudioProcessor
    - __call__
    - apply_chat_template
    - decode
    - save_audio

## Lfm3AudioModel

[[autodoc]] Lfm3AudioModel
    - forward
    - get_audio_features

## Lfm3AudioGenerateOutput

[[autodoc]] models.lfm3_audio.generation_lfm3_audio.Lfm3AudioGenerateOutput

## Lfm3AudioForConditionalGeneration

[[autodoc]] Lfm3AudioForConditionalGeneration
    - forward
    - generate
