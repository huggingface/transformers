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
*This model was contributed to Hugging Face Transformers on 2026-10-23.*


# LFM3-Audio

## Overview

LFM3-Audio is the speech-and-text model built on [LFM3](lfm3) by [Liquid AI](https://liquid.ai/). One LFM3 language
model reads speech through a Liquid speech encoder and writes speech as frames of 12 codebook codes through a depth
decoder, alongside text. Speech frames are turned into audio by the [Liquid codec](liquid_codec).

## Architecture

- **Speech encoder**: three stride-2 convolutions over the 128-bin log-mel spectrogram, then a causal 17-layer LFM2
  model with a sliding attention window of 32 frames. Every 8 spectrogram frames (80 ms) become one language token.
- **Language model**: LFM3, with the text head tied to the input embeddings.
- **Speech output**: every frame is 12 codes at 12.5 Hz. Their embeddings are summed on the way in. On the way out a
  depth decoder projects the language model's hidden state into one slot per codebook and a 3-layer bidirectional
  LFM3 model fills the codebooks in a random order over 3 passes (MaskGIT): in training a random subset of each
  frame's codebooks is hidden and predicted from the others.
- **Codec**: the [Liquid codec](liquid_codec), 12 codebooks of 2016 codes at 12.5 frames per second, 24 kHz.

Generation is text by default. `<|audio_start|>` switches to speech until an end-of-audio frame. `<|mixed_start|>`
alternates blocks of 6 text tokens and 12 speech frames; `<|text_end|>` closes the text side and the speech goes on
until an end-of-audio frame. The processor expands one `<|audio|>` placeholder per input clip into the clip's encoder
tokens and, given the codec, one `<|audio_out|>` placeholder per output clip into its speech frames.

## Usage

The LFM3-Audio-2B checkpoint bundles its codec in `audio_tokenizer/`; the processor loads it as
`processor.audio_tokenizer`. No separate codec repository is needed.

```python
import torch
from transformers import AutoProcessor, Lfm3AudioForConditionalGeneration

model_id = "LiquidAI/LFM3-Audio-2B-Preview"
model = Lfm3AudioForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
processor = AutoProcessor.from_pretrained(model_id)
codec = processor.audio_tokenizer.to("cuda")

conversation = [
    {"role": "system", "content": "Respond with interleaved text and audio."},
    {"role": "user", "content": [{"type": "audio", "audio": "question.wav"}]},
    {"role": "assistant", "content": "<|mixed_start|>"},
]
inputs = processor.apply_chat_template(conversation, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)

text = processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
codes = outputs.audio_codes[0, :, : outputs.audio_lengths[0]]
codes = codes[:, ~(codes == model.config.audio_eos_code).all(0)]  # drop the end-of-audio frames
waveform = codec.decode(codes[None]).audio_values[0]  # 24 kHz
```

`generate` is the [`~generation.GenerationMixin`] loop with a text/speech mode per row: the text sampling settings
come from the model's generation config as usual, the speech frames from the depth decoder's own generation config
(`do_sample`, `temperature`, `top_k`, `repetition_penalty` and `repetition_window`, the number of past frames the
penalty covers). The checkpoint's `generation_config.json` carries both, the depth decoder's under the
`depth_decoder_` prefix, and sets `return_dict_in_generate` so the bare call returns the [`Lfm3AudioGenerateOutput`]
holding `audio_codes` next to `sequences`. Override per call with `depth_decoder_*` kwargs
(`model.generate(**inputs, depth_decoder_temperature=0.8)`), or force a mode with `generation_mode="text"`,
`"audio"` or `"interleaved"`; it is inferred per row from the prompt when omitted. Every speech frame occupies one
`<|audio_out|>` slot of `sequences`, end-of-audio frames included, so the output can be fed back as a prompt.

The frames of a finished turn are decoded with the bundled codec as above. `generate` returns the frames once the
loop completes (the streamer only carries token ids); the codec can then decode them a frame at a time through
`decode(..., use_streaming=True)` (see [`LiquidCodecModel`]) instead of waiting for the whole waveform.

For training, `labels` scores the text tokens and `audio_labels` the speech frames (one row per `<|audio_out|>`
placeholder, `-100` rows ignored); the depth decoder is trained to fill a random subset of each frame's codebooks.

```python
tts = [
    {"role": "system", "content": "Perform TTS in American English using a female voice."},
    {"role": "user", "content": "The quick brown fox jumps over the lazy dog"},
]
prompt = processor.apply_chat_template(tts, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt + "<|audio_start|><|audio_out|><|im_end|>\n", audio_out=[target_speech], return_tensors="pt").to("cuda")
labels = inputs["input_ids"].clone()
labels[:, : len(processor(text=prompt)["input_ids"][0])] = -100  # the checkpoint is trained on assistant turns only
labels[labels >= processor.audio_token_id] = -100
loss = model(**inputs, labels=labels, audio_labels=inputs["audio_out_codes"]).loss
```

## Batching and saving

Use left padding and pass the processor's attention mask. Each row independently switches between text, speech and
interleaved modes and stops at its own text EOS; rows that finish early are padded. The backbone runs the batch
together and MaskGIT batches the rows currently speaking; it still samples reveal orders, so batching changes
random-number consumption even with greedy code selection.

```python
processor.tokenizer.padding_side = "left"
inputs = processor(text=prompts, padding=True, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
for row, length in enumerate(outputs.audio_lengths.tolist()):
    codes = outputs.audio_codes[row, :, :length]
    codes = codes[:, ~(codes == model.config.audio_eos_code).all(0)]
    if codes.shape[-1]:
        waveform = codec.decode(codes[None]).audio_values
```

`audio_lengths` excludes the `-1` padding of `audio_codes`, so slice each row before decoding it. Beam search and
multiple returned sequences per input are not supported.

`processor.save_pretrained(directory)` saves the feature settings, tokenizer, special tokens and chat template.
When a codec is attached, its weights and config are bundled in `audio_tokenizer/`. The directory can be moved and
loaded offline with `AutoProcessor.from_pretrained(directory)`. Save the language/audio model to the same directory
with `model.save_pretrained(directory)` to produce a complete package.

## Incremental audio input

The speech encoder follows a live stream: `model.model.get_audio_features_stream(chunk, cache)` returns the audio
tokens each chunk completes and the encoder state (`Lfm3AudioEncoderStreamingCache`, holding the bounded
spectrogram context and the encoder's key/value cache); pass `is_final=True` to flush the trailing token. Feed the
projected tokens to the language model as `inputs_embeds`, with the cache and attention mask of the prompt already
prefilled, then let `generate` continue from that cache once the clip ends.

```python
features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")["input_features"].to("cuda")
cache, embeds = None, []
for start in range(0, features.shape[-1], 96):
    chunk, cache = model.model.get_audio_features_stream(features[..., start : start + 96], cache, is_final=start + 96 >= features.shape[-1])
    embeds.append(chunk)
```

## Live transcription

Checkpoints trained for live ASR take the audio inside the assistant turn in chunks of `config.interleaved_n_audio`
encoder tokens (0.96 s) and write the text of each chunk as soon as it arrives:

```
<prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|><|im_end|>
```

`transcribe_live` runs one chunk: the streaming encoder completes the chunk's audio tokens, the language model reads
the pending text (the prompt, or the `<|text_end|>` closing the previous block) together with them, and the block is
decoded greedily. The returned state carries the
encoder stream and the language-model context separately: pass `input_ids` again to start the context over, the way
training samples of at most 30 seconds are chained, while the encoder keeps following the stream.

```python
import torch

tokenizer = processor.tokenizer
chunk_frames = 8 * model.config.interleaved_n_audio
prompt = tokenizer("<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n", add_special_tokens=False)
prompt = prompt.input_ids + tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
prompt = torch.tensor([prompt], device=model.device)
suppress = tokenizer.convert_tokens_to_ids(["<|mixed_end|>", "<|endoftext|>"])
stop = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>"])
state, feature_cache, text = None, None, []
mel = torch.empty((1, processor.feature_extractor.feature_size, 0), device=model.device)


def transcribe(frames, is_final=False):
    global state
    output = model.transcribe_live(
        frames,
        state,
        input_ids=prompt if state is None else None,
        is_final=is_final,
        max_new_tokens=96 if is_final else 48,
        suppress_token_ids=suppress,
        stop_token_ids=stop,
    )
    state = output.state
    text.extend(token for token in output.sequences[0].tolist() if token != suppress[0])


for chunk in waveform_chunks:  # mono float samples at 16 kHz, any size
    features, feature_cache = processor.feature_extractor.stream(chunk, feature_cache, sampling_rate=16000)
    mel = torch.cat((mel, features.input_features.to(model.device)), dim=-1)
    while mel.shape[-1] >= chunk_frames:
        transcribe(mel[..., :chunk_frames])
        mel = mel[..., chunk_frames:]
features, feature_cache = processor.feature_extractor.stream([], feature_cache, is_final=True, sampling_rate=16000)
mel = torch.cat((mel, features.input_features.to(model.device)), dim=-1)
while mel.shape[-1] > chunk_frames:
    transcribe(mel[..., :chunk_frames])
    mel = mel[..., chunk_frames:]
transcribe(mel, is_final=True)  # the final chunk holds 0 to 12 tokens and ends the turn
print(tokenizer.decode(text, skip_special_tokens=True))
```

Before the final chunk `<|mixed_end|>`, `<|audio_start|>`, `<|mixed_start|>`, the end-of-turn token and the extra
`suppress_token_ids` are never sampled; a block that exhausts `max_new_tokens` is still closed with `<|text_end|>`.
The final block starts with `<|mixed_end|>` and ends at `<|text_end|>` or a stop token. One stream is transcribed at a
time; the state's caches are mutated in place and a finalized state cannot be reused.

## Lfm3AudioConfig

[[autodoc]] Lfm3AudioConfig

## Lfm3AudioEncoderConfig

[[autodoc]] Lfm3AudioEncoderConfig

## Lfm3AudioFeatureExtractor

[[autodoc]] Lfm3AudioFeatureExtractor
    - __call__
    - stream

## Lfm3AudioFeatureStreamingCache

[[autodoc]] Lfm3AudioFeatureStreamingCache

## Lfm3AudioEncoderStreamingCache

[[autodoc]] Lfm3AudioEncoderStreamingCache

## Lfm3AudioLiveTranscriptionState

[[autodoc]] models.lfm3_audio.generation_lfm3_audio.Lfm3AudioLiveTranscriptionState

## Lfm3AudioLiveTranscriptionOutput

[[autodoc]] models.lfm3_audio.generation_lfm3_audio.Lfm3AudioLiveTranscriptionOutput

## Lfm3AudioProcessor

[[autodoc]] Lfm3AudioProcessor

## Lfm3AudioModel

[[autodoc]] Lfm3AudioModel
    - forward
    - get_audio_features_stream

## Lfm3AudioGenerateOutput

[[autodoc]] models.lfm3_audio.generation_lfm3_audio.Lfm3AudioGenerateOutput

## Lfm3AudioForConditionalGeneration

[[autodoc]] Lfm3AudioForConditionalGeneration
    - forward
    - transcribe_live
    - generate
