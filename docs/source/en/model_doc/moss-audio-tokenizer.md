<!--Copyright 2026 OpenMOSS and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-05.*

# MOSS Audio Tokenizer

[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) is the neural audio codec used by
MOSS-TTS. It encodes waveforms into discrete audio codebook tokens and decodes those tokens back into waveform audio.

## Checkpoint conversion

The original OpenMOSS checkpoint uses its native config and weight names. Convert it to the Transformers format before
loading it with [`AutoModelForAudioTokenization`].

```bash
export PYTHONPATH="$PWD/src"

python src/transformers/models/moss_audio_tokenizer/convert_moss_audio_tokenizer_to_hf.py \
  --input_path_or_repo OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --output_dir /path/to/moss-audio-tokenizer-hf
```

The examples below use the converted local path. Replace it with a Hub repo id if you have pushed the converted
checkpoint.

## Single audio

```python
import torch
from datasets import Audio, load_dataset
from scipy.io.wavfile import write
from transformers import AutoFeatureExtractor, AutoModelForAudioTokenization


model_id = "/path/to/moss-audio-tokenizer-hf"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioTokenization.from_pretrained(model_id, dtype="auto", device_map="auto")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]
inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(model.device)

encoded = model.encode(**inputs, return_dict=True)
code_positions = torch.arange(encoded.audio_codes.shape[-1], device=encoded.audio_codes.device)
codes_mask = code_positions[None, :] < encoded.audio_codes_lengths[:, None]
decoded = model.decode(encoded.audio_codes, padding_mask=codes_mask, return_dict=True)

audio_length = int(decoded.audio_lengths[0])
audio_values = decoded.audio[0, 0, :audio_length].float().cpu().numpy()
write("moss_audio_tokenizer_reconstruction.wav", feature_extractor.sampling_rate, audio_values)
```

## Batch audio

```python
audios = [dataset[i]["audio"]["array"] for i in range(2)]
inputs = feature_extractor(audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(model.device)

encoded = model.encode(**inputs, return_dict=True)
code_positions = torch.arange(encoded.audio_codes.shape[-1], device=encoded.audio_codes.device)
codes_mask = code_positions[None, :] < encoded.audio_codes_lengths[:, None]
decoded = model.decode(encoded.audio_codes, padding_mask=codes_mask, return_dict=True)

first_length = int(decoded.audio_lengths[0])
second_length = int(decoded.audio_lengths[1])
first_reconstruction = decoded.audio[0, 0, :first_length]
second_reconstruction = decoded.audio[1, 0, :second_length]
```

## Fewer Quantizers

Use fewer residual quantizers to trade reconstruction quality for a lower bitrate. The full codec depth is stored in
`config.quantizer_config.n_codebooks`; `num_quantizers` selects how many quantizers to use for a specific encode or
decode call.

```python
encoded = model.encode(**inputs, num_quantizers=8, return_dict=True)
code_positions = torch.arange(encoded.audio_codes.shape[-1], device=encoded.audio_codes.device)
codes_mask = code_positions[None, :] < encoded.audio_codes_lengths[:, None]
decoded = model.decode(encoded.audio_codes, padding_mask=codes_mask, num_quantizers=8, return_dict=True)
```

## Streaming Chunks

`chunk_duration` is expressed in seconds. It must be no longer than
`config.sliding_window_duration`, and `chunk_duration * config.sampling_rate` must be divisible by
`config.hop_length`.

```python
single_inputs = feature_extractor(
    audio,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
).to(model.device)

encoded = model.encode(**single_inputs, chunk_duration=0.08, return_dict=True)
code_positions = torch.arange(encoded.audio_codes.shape[-1], device=encoded.audio_codes.device)
codes_mask = code_positions[None, :] < encoded.audio_codes_lengths[:, None]
decoded = model.decode(encoded.audio_codes, padding_mask=codes_mask, chunk_duration=0.08, return_dict=True)
```

## MossAudioTokenizerConfig

[[autodoc]] MossAudioTokenizerConfig

## MossAudioTokenizerQuantizerConfig

[[autodoc]] MossAudioTokenizerQuantizerConfig

## MossAudioTokenizerEncoderConfig

[[autodoc]] MossAudioTokenizerEncoderConfig

## MossAudioTokenizerDecoderConfig

[[autodoc]] MossAudioTokenizerDecoderConfig

## MossAudioTokenizerFeatureExtractor

[[autodoc]] MossAudioTokenizerFeatureExtractor

## MossAudioTokenizerModel

[[autodoc]] MossAudioTokenizerModel
    - encode
    - decode
    - forward
