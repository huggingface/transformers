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

# Qwen3-TTS Tokenizer

## Overview

The multi-codebook tokenizer is the audio codec behind [Qwen3-TTS](./qwen3_tts): the talker predicts codes in this
codec's space, and this model turns them back into a waveform. It can also be used on its own to tokenize audio.

- **Low frame rate**: audio is encoded at 12.5 frames per second (24 kHz input, a downsampling rate of 1920 samples per
  frame), so one second of speech becomes 12.5 frames.
- **Multi-codebook residual quantization**: each frame is represented by 16 codes rather than one, with a codebook size
  of 2048. Encoding therefore returns a `(time, 16)` tensor per utterance.
- **24 kHz in, 24 kHz out**: the decoder reconstructs at the input sample rate.

The encoder is a [Mimi](./mimi)-based residual vector quantizer, of whose 32 quantizer layers the first 16 are used.
The decoder is the Code2Wav vocoder shared with [Qwen3-Omni-MoE](./qwen3_omni_moe): a sliding-window transformer
followed by a causal-convolution upsampling stack.

A model checkpoint is available at
[Qwen/Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz).

This model was contributed by [Vandit Shah](https://huggingface.co/shahvandit).

## Usage

The [`Qwen3TTSTokenizerFeatureExtractor`] turns raw waveforms into the `(batch_size, num_samples)`
float tensor and `padding_mask` expected by [`~Qwen3TTSTokenizerModel.encode`], whose codes are then
reconstructed with [`~Qwen3TTSTokenizerModel.decode`]:

```python
import torch
from scipy.io import wavfile

from transformers import AutoModel, Qwen3TTSTokenizerFeatureExtractor
from transformers.audio_utils import load_audio_librosa


model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# load model and feature extractor
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = Qwen3TTSTokenizerFeatureExtractor(
    sampling_rate=model.config.input_sampling_rate
)

# load audio at the sample rate the tokenizer expects
audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=model.config.input_sampling_rate,
)
inputs = feature_extractor(audio, sampling_rate=model.config.input_sampling_rate).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([1, 222480])

with torch.no_grad():
    # encode: one frame per 1920 input samples, 16 codes per frame
    codes = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes[0]
    print("Codes shape:", codes.shape)
    # Codes shape: torch.Size([116, 16])

    # decode: `decode` is batched, so add back the batch dimension
    audio_values = model.decode(codes.unsqueeze(0)).audio_values[0]
    print("Reconstructed audio shape:", audio_values.shape)
    # Reconstructed audio shape: torch.Size([222720])

# save audio
output_fp = "qwen3_tts_tokenizer_reconstructed.wav"
wavfile.write(output_fp, model.config.output_sampling_rate, audio_values.float().cpu().numpy())
print(f"Reconstructed audio saved to: {output_fp}")
```

The reconstruction is padded up to a whole number of frames, so it can be slightly longer than the input (here
116 × 1920 = 222720 samples for 222480 input samples).

## Batched inputs

The feature extractor pads waveforms of different lengths to a common length and builds the `padding_mask` marking the
real samples, so trailing padding is not encoded. [`~Qwen3TTSTokenizerModel.encode`] then returns a *list*
of code tensors, one per utterance, each already trimmed to its own length:

```python
import torch

from transformers import AutoModel, Qwen3TTSTokenizerFeatureExtractor
from transformers.audio_utils import load_audio_librosa


model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = Qwen3TTSTokenizerFeatureExtractor(
    sampling_rate=model.config.input_sampling_rate
)

audios = [
    load_audio_librosa(url, sampling_rate=model.config.input_sampling_rate)
    for url in [
        "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
        "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Carter_man.wav",
    ]
]

inputs = feature_extractor(audios, sampling_rate=model.config.input_sampling_rate).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([2, 665600])

with torch.no_grad():
    codes_list = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes
    print("Codes shapes:", [tuple(codes.shape) for codes in codes_list])
    # Codes shapes: [(116, 16), (347, 16)]

    # `decode` reads each utterance's length back from the codes, where -1 marks padding
    padded_codes = torch.nn.utils.rnn.pad_sequence(codes_list, batch_first=True, padding_value=-1)
    audio_values = model.decode(padded_codes).audio_values
    print("Reconstructed shapes:", [tuple(audio.shape) for audio in audio_values])
    # Reconstructed shapes: [(222720,), (666240,)]
```

### Speed-up with `torch.compile`

You can speed up inference with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html). The first
few calls are slower due to compilation overhead, but subsequent calls with the same input shape are faster. Because the
model exposes `encode` and `decode` rather than a single forward, compile each of them.

```python
import torch
from transformers import AutoModel, Qwen3TTSTokenizerFeatureExtractor
from transformers.audio_utils import load_audio_librosa

model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = Qwen3TTSTokenizerFeatureExtractor(
    sampling_rate=model.config.input_sampling_rate
)

audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=model.config.input_sampling_rate,
)
inputs = feature_extractor(audio, sampling_rate=model.config.input_sampling_rate).to(model.device, model.dtype)

model.encode = torch.compile(model.encode)
model.decode = torch.compile(model.decode)

# warmup (the first calls include compilation)
for _ in range(3):
    with torch.inference_mode():
        codes = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes[0]
        _ = model.decode(codes.unsqueeze(0)).audio_values[0]

with torch.inference_mode():
    codes = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes[0]
    audio_values = model.decode(codes.unsqueeze(0)).audio_values[0]
print("Reconstructed audio shape:", audio_values.shape)
```

## Qwen3TTSTokenizerConfig

[[autodoc]] Qwen3TTSTokenizerConfig

## Qwen3TTSTokenizerCode2WavConfig

[[autodoc]] Qwen3TTSTokenizerCode2WavConfig

## Qwen3TTSTokenizerFeatureExtractor

[[autodoc]] Qwen3TTSTokenizerFeatureExtractor

## Qwen3TTSTokenizerModel

[[autodoc]] Qwen3TTSTokenizerModel
    - encode
    - decode
