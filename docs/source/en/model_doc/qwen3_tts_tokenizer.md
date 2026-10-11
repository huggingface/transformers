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

The feature extractor, loaded with [`AutoFeatureExtractor`], turns raw waveforms into the `(batch_size, 1, num_samples)`
float tensor and `padding_mask` expected by [`~Qwen3TTSTokenizerModel.encode`], whose codes are then
reconstructed with [`~Qwen3TTSTokenizerModel.decode`]:

```python
import torch
from scipy.io import wavfile

from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio_librosa


model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# load model and feature extractor
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# load audio at the sample rate the tokenizer expects
audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=model.config.input_sampling_rate,
)
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([1, 1, 222480])

with torch.no_grad():
    # encode: one frame per 1920 input samples, 16 codes per frame
    audio_codes = model.encode(**inputs).audio_codes
    print("Audio codes shape:", audio_codes.shape)
    # Audio codes shape: torch.Size([1, 116, 16])

    audio_values = model.decode(audio_codes).audio_values
    print("Reconstructed audio shape:", audio_values.shape)
    # Reconstructed audio shape: torch.Size([1, 222720])

    # Equivalently, you can do encoding and decoding in one step
    model_output = model(**inputs)
    audio_codes = model_output.audio_codes
    audio_values = model_output.audio_values
    print("Forward audio shape:", audio_values.shape)
    # Forward audio shape: torch.Size([1, 222480])

# save audio
output_fp = "qwen3_tts_tokenizer_reconstructed.wav"
wavfile.write(output_fp, model.config.output_sampling_rate, audio_values[0].float().cpu().numpy())
print(f"Reconstructed audio saved to: {output_fp}")
```

The output of [`~Qwen3TTSTokenizerModel.decode`] is padded up to a whole number of frames, so it can be slightly longer
than the input (here 116 × 1920 = 222720 samples for 222480 input samples). The forward pass truncates it back to the
input length.

## Batched inputs

The feature extractor pads waveforms of different lengths to a common length and builds the `padding_mask` marking the
real samples. [`~Qwen3TTSTokenizerModel.encode`] returns the codes as one padded tensor, together with an
`audio_codes_mask` marking the valid frames of each utterance, and the padded codes can be passed directly to
[`~Qwen3TTSTokenizerModel.decode`]:

```python
import torch

from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio_librosa


model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

audios = [
    load_audio_librosa(url, sampling_rate=model.config.input_sampling_rate)
    for url in [
        "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
        "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Carter_man.wav",
    ]
]

inputs = feature_extractor(
    audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([2, 1, 665600])

with torch.no_grad():
    encoder_outputs = model.encode(**inputs)
    audio_codes = encoder_outputs.audio_codes
    print("Audio codes shape:", audio_codes.shape)
    # Audio codes shape: torch.Size([2, 347, 16])
    print("Valid frames:", encoder_outputs.audio_codes_mask.sum(dim=-1).tolist())
    # Valid frames: [116, 347]

    audio_values = model.decode(audio_codes).audio_values
    print("Reconstructed audio shape:", audio_values.shape)
    # Reconstructed audio shape: torch.Size([2, 666240])
```

### Speed-up with `torch.compile`

You can speed up inference with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html). The first
few calls are slower due to compilation overhead, but subsequent calls with the same input shape are faster. Because the
model exposes `encode` and `decode` rather than a single forward, compile each of them.

```python
import torch
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio_librosa

model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=model.config.input_sampling_rate,
)
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)

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

## Qwen3TTSTokenizerEncoderConfig

[[autodoc]] Qwen3TTSTokenizerEncoderConfig

## Qwen3TTSTokenizerQuantizerConfig

[[autodoc]] Qwen3TTSTokenizerQuantizerConfig

## Qwen3TTSTokenizerModel

[[autodoc]] Qwen3TTSTokenizerModel
    - encode
    - decode
