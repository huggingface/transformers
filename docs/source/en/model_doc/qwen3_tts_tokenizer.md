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
from datasets import Audio, load_dataset
from scipy.io import wavfile

from transformers import AutoFeatureExtractor, AutoModel


model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# load model and feature extractor
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# load audio at the sample rate the tokenizer expects
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([1, 1, 140520])

with torch.no_grad():
    # encode: one frame per 1920 input samples, 16 codes per frame
    audio_codes = model.encode(**inputs).audio_codes
    print("Audio codes shape:", audio_codes.shape)
    # Audio codes shape: torch.Size([1, 74, 16])

    audio_values = model.decode(audio_codes).audio_values
    print("Reconstructed audio shape:", audio_values.shape)
    # Reconstructed audio shape: torch.Size([1, 142080])

    # Equivalently, you can do encoding and decoding in one step
    model_output = model(**inputs)
    audio_codes = model_output.audio_codes
    audio_values = model_output.audio_values
    print("Forward audio shape:", audio_values.shape)
    # Forward audio shape: torch.Size([1, 140520])

# save audio
output_fp = "qwen3_tts_tokenizer_reconstructed.wav"
wavfile.write(output_fp, model.config.output_sampling_rate, audio_values[0].float().cpu().numpy())
print(f"Reconstructed audio saved to: {output_fp}")
```

The output of [`~Qwen3TTSTokenizerModel.decode`] is padded up to a whole number of frames, so it can be slightly longer
than the input (here 74 × 1920 = 142080 samples for 140520 input samples). The forward pass truncates it back to the
input length.

## Batched inputs

The feature extractor pads waveforms of different lengths to a common length and builds the `padding_mask` marking the
real samples. [`~Qwen3TTSTokenizerModel.encode`] returns the codes as one padded tensor, together with an
`audio_codes_mask` marking the valid frames of each utterance, and the padded codes can be passed directly to
[`~Qwen3TTSTokenizerModel.decode`]:

```python
import torch
from datasets import Audio, load_dataset

from transformers import AutoFeatureExtractor, AutoModel


batch_size = 2
model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audios = [dataset[i]["audio"]["array"] for i in range(batch_size)]
inputs = feature_extractor(
    audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)
print("Input audio shape:", inputs["input_values"].shape)
# Input audio shape: torch.Size([2, 1, 140520])

with torch.no_grad():
    encoder_outputs = model.encode(**inputs)
    audio_codes = encoder_outputs.audio_codes
    print("Audio codes shape:", audio_codes.shape)
    # Audio codes shape: torch.Size([2, 74, 16])
    print("Valid frames:", encoder_outputs.audio_codes_mask.sum(dim=-1).tolist())
    # Valid frames: [74, 61]

    audio_values = model.decode(audio_codes).audio_values
    print("Reconstructed audio shape:", audio_values.shape)
    # Reconstructed audio shape: torch.Size([2, 142080])
```

### Speed-up with `torch.compile`

You can speed up inference with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html). The first
few calls are slower due to compilation overhead, but subsequent calls with the same input shape are faster.

```python
import torch
from datasets import Audio, load_dataset

from transformers import AutoFeatureExtractor, AutoModel

model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
).to(model.device, model.dtype)

compiled_model = torch.compile(model)

# Warmup (includes compilation on first call)
for _ in range(3):
    with torch.inference_mode():
        _ = compiled_model(**inputs)

with torch.inference_mode():
    output = compiled_model(**inputs)
print("Audio values shape:", output.audio_values.shape)
# Audio values shape: torch.Size([1, 140520])
```

## Separate encoder and decoder

The tokenizer builds its encoder and decoder through `AutoModel.from_config`. The encoder uses
`model.config.encoder_config`; the decoder settings remain in the main config and are exposed as a decoder-only
`model.config.decoder_config`. The quantizer has its own `model.config.quantizer_config`.

The loaded components are available as `model.encoder` and `model.decoder`. Their `forward` methods operate on
continuous representations; the top-level `encode` and `decode` methods handle quantization separately.
You can save and reload the trained components independently:

```python
model.encoder.save_pretrained("qwen3_tts_encoder")
model.decoder.save_pretrained("qwen3_tts_decoder")

encoder = AutoModel.from_pretrained("qwen3_tts_encoder")
decoder = AutoModel.from_pretrained("qwen3_tts_decoder")
```

## Quantizer configuration

`quantizer_config.semantic_quantizer_config` configures the semantic residual vector quantizer independently.
The outer `quantizer_config.n_codebooks` counts all codebooks; `semantic_quantizer_config.n_codebooks` assigns
the first codebooks to semantic codes, and the remaining codebooks are acoustic.

```python
from transformers import Qwen3TTSTokenizerConfig

config = Qwen3TTSTokenizerConfig(
    quantizer_config={
        "n_codebooks": 16,
        "semantic_quantizer_config": {"n_codebooks": 1, "codebook_size": 2048},
    }
)
```

The semantic config also exposes `codebook_dim`, `hidden_size`, and `vector_quantization_hidden_dimension`.
Unspecified semantic dictionary fields inherit the outer quantizer settings. You can also pass a
[`Qwen3TTSTokenizerSemanticQuantizerConfig`] instance with its own settings. Keep dimensions and codebook sizes
consistent with the weights when loading a pretrained checkpoint.

Older configs with `num_semantic_quantizers` are migrated on loading. Saved configs store the semantic count
only as `semantic_quantizer_config.n_codebooks`.

## Qwen3TTSTokenizerConfig

[[autodoc]] Qwen3TTSTokenizerConfig

## Qwen3TTSTokenizerEncoderConfig

[[autodoc]] Qwen3TTSTokenizerEncoderConfig

## Qwen3TTSTokenizerDecoderConfig

[[autodoc]] Qwen3TTSTokenizerDecoderConfig

## Qwen3TTSTokenizerQuantizerConfig

[[autodoc]] Qwen3TTSTokenizerQuantizerConfig

## Qwen3TTSTokenizerSemanticQuantizerConfig

[[autodoc]] Qwen3TTSTokenizerSemanticQuantizerConfig

## Qwen3TTSTokenizerModel

[[autodoc]] Qwen3TTSTokenizerModel
    - encode
    - decode

## Qwen3TTSTokenizerEncoderModel

[[autodoc]] Qwen3TTSTokenizerEncoderModel
    - forward

## Qwen3TTSTokenizerDecoderModel

[[autodoc]] Qwen3TTSTokenizerDecoderModel
    - forward
